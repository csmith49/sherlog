from .data import Graph
from typing import Iterable, Optional
from itertools import chain
from math import log, exp
from statistics import mean
from sherlog.logs import get_external
from sherlog.program import loads
from sherlog.inference import Optimizer, minibatch, Batch
from torch.optim import SGD
import torch

logger = get_external("smokers.sherlog")

SOURCE = """
# probabilistic rules
stress :: stress(X) <- person(X).
spontaneous :: asthma_spontaneous(X) <- person(X).
comorbid :: asthma_comorbid(X) <- smokes(X).
influence :: influence(X, Y) <- friend(X, Y).

# logical rules
smokes(X) <- stress(X).
smokes(X) <- influence(X, Y), smokes(Y).
asthma(X) <- asthma_spontaneous(X).
asthma(X) <- asthma_comorbid(X).

# ontological rules
!dependency smokes(X) | not_smokes(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!dependency asthma(X) | not_asthma(X) <- person(X).
!constraint asthma(X), not_asthma(X).
"""

def translate_graph(graph, force_target=None):
    # compute the structure
    people = [f"person({p})." for p in graph.people()]
    friends = [f"friend({s}, {d})." for (s, d) in graph.friends()]

    structure = '\n'.join(chain(people, friends))

    # provide observables as evidence
    smokes = [f"smokes({p})" for p in graph.smokes(True)]
    not_smokes = [f"not_smokes({p})" for p in graph.smokes(False)]
    asthma = [f"asthma({p})" for p in graph.asthma(True, force_target=force_target)]
    not_asthma = [f"not_asthma({p})" for p in graph.asthma(False, force_target=force_target)]

    evidence = f"!evidence {', '.join(chain(smokes, not_smokes, asthma, not_asthma))}."

    result = f"{SOURCE}\n{structure}\n{evidence}"
    return result

class SherlogModel:
    def __init__(self):
        self._stress = torch.tensor(0.5, requires_grad=True)
        self._spontaneous = torch.tensor(0.5, requires_grad=True)
        self._comorbid = torch.tensor(0.5, requires_grad=True)
        self._influence = torch.tensor(0.5, requires_grad=True)

        self._namespace = {
            "stress" : self._stress,
            "spontaneous" : self._spontaneous,
            "comorbid" : self._comorbid,
            "influence" : self._influence
        }

    def parameters(self):
        yield from self._namespace.values()

    def clamp(self, epsilon=0.001):
        with torch.no_grad():
            for value in self._namespace.values():
                value.clamp_(0 + epsilon, 1 - epsilon)

    def program(self, graph, force_target=None):
        return loads(translate_graph(graph, force_target=force_target), namespace=self._namespace)

    def fit(self, train, test = None, epochs : int = 1, learning_rate : float = 0.1, batch_size : int = 10, **kwargs):
        # do everything manually for now
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate * 0.01, max_lr=learning_rate)
        lls = {}

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch}...")
            for batch in minibatch(train, batch_size):
                optimizer.zero_grad()
                objective = torch.tensor(0.0)

                for graph in batch:
                    logger.info("Translating graph...")
                    program, evidence = self.program(graph)
                    logger.info("Program built...")
                    log_likelihood = program.likelihood(evidence[0], explanations=1, width=15, samples=100, depth=100, seeds=1, **kwargs).log()
                    logger.info(f"Log-likelihood: {log_likelihood}")
                    # make sure gradients exist
                    is_nan = torch.isnan(log_likelihood).any()
                    is_inf = torch.isinf(log_likelihood).any()
                    if not is_nan and not is_inf:
                        objective -= log_likelihood

                if objective != 0.0:
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()
                    # scheduler.step()
                    self.clamp()
            

                print(f"STRESS - {self._stress.item()}")
                print(f"SPONTANEOUS - {self._spontaneous.item()}")
                print(f"COMORBID - {self._comorbid.item()}")
                print(f"INFLUENCE - {self._influence.item()}")

            if test is not None:
                lls[epoch] = self.average_log_likelihood(test, explanations=1, samples=500)
                logger.info(f"Epoch {epoch} LL: {lls[epoch]}")

        return lls

    def average_log_likelihood(self, test, explanations : int = 1, samples : int = 500, **kwargs):
        lls = []
        for graph in test:
            lls.append(self.log_likelihood(graph, explanations=explanations, samples=samples, width=30, depth=100, **kwargs))
        return mean(lls)

    def log_likelihood(self, example, explanations : int = 1, samples : int = 100, force_target = None, **kwargs):
        program, evidence = self.program(example, force_target=force_target)
        return program.likelihood(evidence[0], explanations=explanations, samples=samples, **kwargs).log().item()

    def classification_task(self, example, **kwargs):
        asthma = self.log_likelihood(example, explanations=1, samples=100, force_target=True, **kwargs)
        not_asthma = self.log_likelihood(example, explanations=1, samples=100, force_target=False, **kwargs)
        if asthma >= not_asthma:
            confidence = 1.0
        else:
            confidence = 0.0
        return confidence, example.target_classification()
