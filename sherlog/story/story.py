from ..engine import Store, run, value
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

    def loss(self, p=2, index=0):
        store = self.run(scg.algebra)
        
        # build the observation distances
        obs_vec = self.meet.evaluate(store, scg.algebra)
        store_vec = [store[v] for v in self.meet.variables()]
    
        total = torch.tensor(0.0)
        for o, s in zip(obs_vec, store_vec):
            total += torch.dist(o, s, p=2)

        result = value.Variable(f"sherlog:result:{index}")
        store[result] = total
        storch.add_cost(store[result], result.name)

        return store
