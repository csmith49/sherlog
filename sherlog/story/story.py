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
            functor.run_assignment(
                assignment, 
                store, 
                wrap_args=wrap_args, 
                fmap_args=fmap_args, 
                parameters=parameters)
        return store

    def objective(self, functor):
        # get values
        store = self.run(functor)

        # build meet and avoid
        meet = self.meet.equality(store, functor, prefix="sherlog:meet", default=1.0)
        avoid = self.avoid.equality(store, functor, prefix="sherlog:avoid", default=0.0)

        # build objective
        objective = value.Variable("sherlog:objective")
        functor.run(objective, "satisfy", [meet, avoid], store)

        # just return the computed objective - functor should track all relevant info
        return store[objective]

    def dice(self):
        objective = self.objective(semantics.dice.functor)
        score = semantics.dice.magic_box(objective.dependencies())
        result = objective.value * score
        
        return result

    def likelihood(self):
        objective = self.objective(semantics.tensor.functor)
        return objective