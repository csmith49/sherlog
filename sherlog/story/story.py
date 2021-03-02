from ..engine import Store, run, value
from ..inference import Objective
from . import stochastic
from . import scg

import torch
import storch

class Story:
    def __init__(self, model, meet, avoid, external=()):
        self.model = model
        self.meet = meet
        self.avoid = avoid
        self._external = external

    @property
    def store(self):
        return Store(external=self._external)

    def run(self, algebra, parameters={}):
        store = self.store
        for assignment in self.model.assignments:
            run(assignment, store, algebra, parameters=parameters)
        return store

    def generative_model(self):
        store = self.run(stochastic.algebra)

        # build the indexes for the nodes we'll add
        p_meet = value.Variable("sherlog:p_meet")
        p_avoid = value.Variable("sherlog:p_avoid")
        p = value.Variable("sherlog:p")

        # compute values for the nodes
        store[p_meet] = observation_similarity(self.meet, store)
        store[p_avoid] = observation_similarity(self.avoid, store)
        
        store[p] = stochastic.delta(
            p,
            store[p_meet] * (1 - store[p_avoid])
        )

        return store

    def likelihood(self, offset=1, num_samples=100):
        model = stochastic.Predictive(
            self.generative_model,
            num_samples=num_samples,
            return_sites=("sherlog:p",))
        results = model()["sherlog:p"]
        return (offset + torch.sum(results, dim=0)) / (offset + num_samples)

    def objective(self, index=0):
        store = self.run(scg.algebra)
        
        # build the indexes for the nodes we'll add
        p_meet = value.Variable(f"sherlog:p_meet:{index}")
        p_avoid = value.Variable(f"sherlog:p_avoid:{index}")
        p = value.Variable(f"sherlog:p:{index}")

        # compute values for the nodes
        store[p_meet] = observation_similarity(self.meet, store)
        store[p_avoid] = observation_similarity(self.avoid, store)
        store[p] = store[p_meet] * (1 - store[p_avoid])

        return Objective(p, store)

def observation_similarity(observation, store, epsilon=1.0):
    """Computes cosine similarity between a given observation and a store."""

    # undefined for empty vectors, so pick a number that makes the rest of the math correct
    if observation.size == 0: return torch.tensor(0.0)

    # evaluate the observation and store to get tensors
    obs_vec = observation.evaluate(store, scg.algebra)
    str_vec = [store[v] for v in observation.variables()]

    # can't stack in storch (yet), so manually compute cosine similarity
    # epsilon ensures we don't divide by zero
    # equivalent to extending each vector with 1 extra item
    dot_prod = torch.tensor(epsilon)
    mag_a, mag_b = torch.tensor(epsilon ** 2), torch.tensor(epsilon ** 2)

    for a, b in zip(obs_vec, str_vec):
        dot_prod += a * b
        mag_a += torch.pow(a, 2)
        mag_b += torch.pow(b, 2)
    
    return dot_prod / torch.sqrt(mag_a * mag_b)