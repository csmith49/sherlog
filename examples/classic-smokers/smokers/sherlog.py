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
spontaneous :: asthma(X) <- person(X).
comorbid :: asthma(X) <- smokes(X).
influence :: influence(X, Y) <- friend(X, Y).

# logical rules
smokes(X) <- stress(X).
smokes(X) <- influence(X, Y), smokes(Y).

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

    def clamp(self):
        with torch.no_grad():
            for value in self._namespace.values():
                value.clamp_(0, 1)

    def fit(self, train, test = None, epochs : int = 1, learning_rate : float = 0.1, batch_size : int = 10, **kwargs):
        # do everything manually for now
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        lls = {}

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch}...")
            for batch in minibatch(train, batch_size):
                optimizer.zero_grad()
                objective = torch.tensor(0.0)

                for graph in batch:
                    logger.info("Translating graph...")
                    program, evidence = loads(translate_graph(graph), namespace=self._namespace)
                    logger.info("Program built...")
                    log_likelihood = program.likelihood(evidence[0], explanations=1, samples=100, width=10, depth=100, **kwargs).log()
                    logger.info(f"Log-likelihood: {log_likelihood}")
                    # make sure gradients exist
                    is_nan = torch.isnan(log_likelihood).any()
                    is_inf = torch.isinf(log_likelihood).any()
                    if not is_nan and not is_inf:
                        objective -= log_likelihood

                objective.backward()
                optimizer.step()
                self.clamp()

            if test is not None:
                lls[epoch] = self.average_log_likelihood(test, explanations=10, samples=1000, width=50, depth=100)
                logger.info(f"Epoch {epoch} LL: {lls[epoch]}")

        return lls

    def average_log_likelihood(self, test, explanations : int = 10, samples : int = 500, **kwargs):
        lls = []
        for graph in test:
            lls.append(self.log_likelihood(graph, explanations=explanations, samples=samples, **kwargs))
        return mean(lls)

    def log_likelihood(self, example, explanations : int = 10, samples : int = 1000, force_target = None, **kwargs):
        program, evidence = loads(translate_graph(example, force_target=force_target), namespace=self._namespace)
        return program.likelihood(evidence[0], explanations=explanations, samples=samples, **kwargs).log().item()

    def classification_task(self, example, **kwargs):
        asthma = self.log_likelihood(example, force_target=True, **kwargs)
        not_asthma = self.log_likelihood(example, force_target=False, **kwargs)
        if asthma >= not_asthma:
            confidence = 1.0
        else:
            confidence = 0.0
        return confidence, example.target_classification()
