from ..engine import Store, value
from ..inference import Objective
from .observation import Observation
from ..engine import Model
from ..logs import get
from . import semantics

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

    def run(self, functor, wrap_args={}, fmap_args={}, parameters={}):
        store = self.store
        for assignment in self.model.assignments:
            functor.run(assignment, store, wrap_args=wrap_args, fmap_args=fmap_args, parameters=parameters)
        return store

    def likelihood(self):
        store = self.run(semantics.tensor)
        
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