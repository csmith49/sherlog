from sherlog.program import loads
from sherlog.inference import minibatch
from sherlog.logs import get_external
import torch.nn as nn
import torch
from math import exp, log
from statistics import mean
from . import data
from itertools import chain

logger = get_external("neural-smokers.sherlog")

# define the Sherlog model
SOURCE = """
# latent health generation per-person
health(X; dirichlet[dimension]) <- person(X).
observed_health(X; categorical[H]) <- health(X, H).

# health influences many factors (but we don't know how)
influence(X, Y; influence_nn[H1, H2]) <- health(X, H1), health(Y, H2), frined(X, Y).
asthma_risk(X; risk_nn[H]) <- health(X, H).

# probabilistic rules
stress :: smokes(X) <- person(X).
M :: smokes(X) <- influence(Y, X, M), friend(Y, X).
comorbid :: asthma(X) <- smokes(X).
R :: asthma(X) <- asthma_risk(X, R).

# ontological rules
!dependency smokes(X) | not_smokes(X) <- person(X).
!dependency asthma(X) | not_asthma(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!constraint asthma(X), not_asthma(X).
"""

HIDDEN_DIMENSIONS = 100

# compute the risk of asthma
class RiskNN(nn.Module):
    def __init__(self):
        super(RiskNN, self).__init__()
        self._nn = nn.Sequential(
            nn.Linear(data.HEALTH_DIMENSIONS, HIDDEN_DIMENSIONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSIONS, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, health):
        result = self._nn(health)
        return result[-1] # we just care about the probability of 1

# compute the influence magnitude
class InfluenceNN(nn.Module):
    def __init__(self):
        super(InfluenceNN, self).__init__()
        self._nn = nn.Sequential(
            nn.Linear(data.HEALTH_DIMENSIONS * 2, HIDDEN_DIMENSIONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSIONS, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, x, y):
        nn_input = torch.cat([x, y])
        result = self._nn(nn_input)
        return result[-1]

# convert a graph to a sherlog program
def translate_graph(graph, force_target=None):
    # compute the structure of the graph
    people = [f"person({p})." for p in graph.people()]
    friends = [f"friend({s}, {d})." for (s, d) in graph.friends(force_target=force_target)]

    structure = '\n'.join(chain(people, friends))

    # provide the observables as evidence
    smokes = [f"smokes({p})" for p in graph.smokes(True)]
    not_smokes = [f"not_smokes({p})" for p in graph.smokes(False)]
    asthma = [f"asthma({p})" for p in graph.asthma(True)]
    not_asthma = [f"not_asthma({p})" for p in graph.asthma(False)]
    health = [f"observed_health({p}, {h})" for p, h in graph.observed_health()]

    evidence = f"!evidence {', '.join(chain(smokes, not_smokes, asthma, not_asthma, health))}."

    # the raw source
    result = f"{SOURCE}\n{structure}\n{evidence}"
    return result

# the model
class SherlogModel:
    def __init__(self):
        self._stress = torch.tensor(0.5, requires_grad=True)
        self._comorbid = torch.tensor(0.5, requires_grad=True)
        self._risk_nn = RiskNN()
        self._influence_nn = InfluenceNN()

        self._namespace = {
            "dimension" : torch.ones(data.HEALTH_DIMENSIONS) * 0.5,
            "stress" : self._stress,
            "comorbid" : self._comorbid,
            "risk_nn" : self._risk_nn,
            "influence_nn" : self._influence_nn
        }

    def parameters(self):
        yield self._stress
        yield self._comorbid
        yield from self._risk_nn.parameters()
        yield from self._influence_nn.parameters()

    def clamp(self):
        with torch.no_grad():
            self._stress.clamp_(0, 1)
            self._comorbid.clamp_(0, 1)

    def fit(self, train, test, epochs : int = 1, learning_rate : float = 0.1, batch_size : int = 10, **kwargs):
        # we're doing everything manually here, unfortunately
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        lls = {}

        for epoch in range(epochs):
            for batch in minibatch(train, batch_size):

                optimizer.zero_grad()
                objective = torch.tensor(0.0)

                for graph in batch:
                    program, evidence = loads(translate_graph(graph), namespace=self._namespace)
                    log_likelihood = program.likelihood(evidence[0], explanations=1, samples=5000, width=50).log()
                    logger.info(f"Log-likelihood: {log_likelihood.item()}")
                    # we have to make sure the gradients actually exist
                    is_nan = torch.isnan(log_likelihood).any()
                    is_inf = torch.isinf(log_likelihood).any()
                    if not is_nan and not is_inf:
                        objective -= log_likelihood

                if objective != 0.0:
                    objective.backward()
                    optimizer.step()
                    self.clamp()

            test_log_likelihood = self.average_log_likelihood(test, explanations=1, samples=5000, width=50)
            lls[epoch] = test_log_likelihood
            logger.info(f"Epoch {epoch} LL: {test_log_likelihood}")

        return lls

    def average_log_likelihood(self, test, explanations : int = 10, samples : int = 1000, **kwargs):
        lls = []
        for graph in test:
            lls.append(self.log_likelihood(graph, explanations=explanations, samples=samples, **kwargs))
        return mean(lls)

    def log_likelihood(self, example, explanations : int = 10, samples : int = 1000, force_target = None, **kwargs):
        program, evidence = loads(translate_graph(example, force_target=force_target), namespace=self._namespace)
        return program.likelihood(evidence[0], explanations=explanations, samples=samples, **kwargs).log().item()

    def classification_task(self, example, **kwargs):
        friends = self.log_likelihood(example, force_target=True, **kwargs)
        not_friends = self.log_likelihood(example, force_target=False, **kwargs)
        # if friends >= not_friends:
        if exp(friends - not_friends) >= 0.5:
            confidence = 1.0
        else:
            confidence = 0.0
        return confidence, example.target_classification()