from sherlog.program import loads
from sherlog.inference import minibatch
from sherlog.logs import get_external
import torch.nn as nn
import torch
from math import exp
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

HIDDEN_DIMENSIONS = 10

# compute the risk of asthma
class RiskNN(nn.Module):
    def __init__(self):
        super(RiskNN, self).__init__()
        self._nn = nn.Sequential(
            nn.Linear(data.HEALTH_DIMENSIONS, HIDDEN_DIMENSIONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSIONS, 2),
            nn.Softmax()
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
            nn.Softmax()
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
            "dimension" : torch.ones(data.HEALTH_DIMENSIONS),
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

    def fit(self, data, epochs : int = 1, learning_rate : float = 0.01, batch_size : int = 1, **kwargs):
        # we're doing everything manually here, unfortunately
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for batch in minibatch(data, batch_size=batch_size, epochs=epochs, direct=True):
            optimizer.zero_grad()
            
            # build the objective
            objective = torch.tensor(0.0)
            for graph in batch:
                program, evidence = loads(translate_graph(graph), namespace=self._namespace)
                objective -= program.likelihood(evidence[0], explanations=3, samples=100).log()

            # and optimize
            objective.backward()
            optimizer.step()

    def log_likelihood(self, example, explanations : int = 1, samples : int = 100, force_target = None, **kwargs):
        program, evidence = loads(translate_graph(example, force_target=force_target), namespace=self._namespace)
        return program.likelihood(evidence[0], explanations=explanations, samples=samples).log().item()

    def classification_task(self, example, **kwargs):
        friends = self.log_likelihood(example, force_target=True, **kwargs)
        not_friends = self.log_likelihood(example, force_target=False, **kwargs)

        if friends >= not_friends:
            confidence = 1.0
        else:
            confidence = 0.0
        return confidence, example.target_classification()