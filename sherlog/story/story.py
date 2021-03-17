from ..engine import Store, run, value
from ..inference import Objective
from .observation import Observation
from ..engine import Model
from ..logs import get
from . import stochastic
from . import pure

import torch
import storch

logger = get("story")

class Story:
    def __init__(self, model, meet, avoid, external=()):
        logger.info(f"Story {self} built.")
        self.model = model
        self.meet = meet
        self.avoid = avoid
        self._external = external

    @classmethod
    def of_json(cls, json, external=()):
        logger.info(f"Building story from serialization: {json}...")
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        avoid = Observation.of_json(json["avoid"])
        return cls(model, meet, avoid, external=external)

    @property
    def store(self):
        return Store(external=self._external)

    def run(self, algebra, parameters={}):
        store = self.store
        for assignment in self.model.assignments:
            run(assignment, store, algebra, parameters=parameters)
        return store

    # def generative_model(self):
    #     store = self.run(stochastic.algebra)

    #     # build the indexes for the nodes we'll add
    #     p_meet = value.Variable("sherlog:p_meet")
    #     p_avoid = value.Variable("sherlog:p_avoid")
    #     p = value.Variable("sherlog:p")

    #     # compute values for the nodes
    #     store[p_meet] = observation_similarity(self.meet, store)
    #     store[p_avoid] = observation_similarity(self.avoid, store)
        
    #     store[p] = stochastic.delta(
    #         p,
    #         store[p_meet] * (1 - store[p_avoid])
    #     )

    #     return store

    # def likelihood(self, offset=1, num_samples=100):
    #     model = stochastic.Predictive(
    #         self.generative_model,
    #         num_samples=num_samples,
    #         return_sites=("sherlog:p",))
    #     results = model()["sherlog:p"]
    #     return (offset + torch.sum(results, dim=0)) / (offset + num_samples)

    def likelihood(self):
        store = self.run(pure.algebra)
        
        # build the indexes for the nodes we'll add
        p_meet = value.Variable("sherlog:p_meet")
        p_avoid = value.Variable("sherlog:p_avoid")
        p = value.Variable("sherlog:p")

        # compute values for the nodes
        store[p_meet] = self.meet.similarity(store, default=1.0)
        store[p_avoid] = self.avoid.similarity(store, default=0.0)
        store[p] = store[p_meet] * (1 - store[p_avoid])

        logger.info(f"{self} likelihood: {store[p]:f}.")

        return store[p]