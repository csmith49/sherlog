from ..engine import Store, run, value
from ..interface import io
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

        # build the site for the result
        result = value.Variable("sherlog:result")
        similarities = []
        obs_t = torch.stack(list(self.meet.evaluate(store, stochastic.algebra)))
        store_t = torch.stack([store[v] for v in self.meet.variables()])
        similarities.append(torch.cosine_similarity(obs_t, store_t, dim=0))
        store[result] = stochastic.delta(result, max(similarities))
            
        return store

    def likelihood(self, offset=1, num_samples=100):
        model = stochastic.Predictive(
            self.generative_model,
            num_samples=num_samples,
            return_sites=("sherlog:result",))
        results = model()["sherlog:result"]
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

        # add the result node as a cost
        storch.add_cost(-1 * store[p], p.name)

        return store

def observation_similarity(observation, store, epsilon=0.1):
    """Computes cosine similarity between a given observation and a store."""

    if observation.size is 0: return torch.tensor(0.0)

    obs_vec = observation.evaluate(store, scg.algebra)
    str_vec = [store[v] for v in observation.variables()]

    dot_prod = torch.tensor(epsilon)
    mag_a, mag_b = torch.tensor(epsilon), torch.tensor(epsilon)

    for a, b in zip(obs_vec, str_vec):
        dot_prod += a * b
        mag_a += torch.pow(a, 2)
        mag_b += torch.pow(b, 2)
    
    return dot_prod / torch.sqrt(mag_a * mag_b)