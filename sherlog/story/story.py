from ..engine import Store, run, value
from . import stochastic
from . import scg

import torch
import storch

class Story:
    def __init__(self, model, observations, external=()):
        self.model = model
        self.observations = observations
        self._external = external

    @property
    def store(self):
        return Store(external=self._external)

    def run(self, algebra, parameters={}):
        store = self.store
        for stmt in self.model.statements:
            run(stmt, store, algebra, parameters=parameters)
        return store

    def generative_model(self):
        store = self.run(stochastic.algebra)

        # build the site for the result
        result = value.Variable("sherlog:result")
        similarities = []
        for obs in self.observations:
            obs_t = torch.stack(list(obs.evaluate(store, stochastic.algebra)))
            store_t = torch.stack([store[v] for v in obs.variables()])
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

    def loss(self, p=2):
        store = self.run(scg.algebra)
        
        # build the observation distances
        distances = []
        for obs in self.observations:
            obs_vec = obs.evaluate(store, scg.algebra)
            store_vec = [store[v] for v in obs.variables()]
    
            total = torch.tensor(0.0)
            for o, s in zip(obs_vec, store_vec):
                total += torch.dist(o, s, p=2)
            distances.append(total)

        result = value.Variable("sherlog:result")
        store[result] = torch.min(torch.stack(distances))
        storch.add_cost(store[result], result.name)

        return store