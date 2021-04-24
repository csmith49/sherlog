from ..engine import Store, value
from .observation import Observation
from ..engine import Model
from ..logs import get
from . import semantics

import torch

logger = get("explanation")

class Explanation:
    def __init__(self, model, meet, avoid, external=()):
        logger.info(f"Explanation {self} built.")
        self.model = model
        self.meet = meet
        self.avoid = avoid
        self._external = external

    @classmethod
    def of_json(cls, json, external=()):
        logger.info(f"Building explanation from serialization: {json}...")
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        avoid = Observation.of_json(json["avoid"])
        return cls(model, meet, avoid, external=external)

    @property
    def store(self):
        return Store(external=self._external)

    def run(self, functor, wrap_args={}, fmap_args={}, parameters={}):
        """Evaluate the explanation in the given functor.

        Parameters
        ----------
        functor : Functor

        wrap_args : Optional[Dict[string, Any]]

        fmap_args : Optional[Dict[string, Any]]

        parameters : Optional[Dict[string, Dict[string, Any]]]

        Returns
        -------
        Store[Functor.t]

        """
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
        """Use provided functor to evaluate the explanation and build optimization objective.

        Parameters
        ----------
        functor : Functor

        Returns
        -------
        Functor.t
        """
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

    def miser(self, samples=1):
        """Build a Miser surrogate objective for the story.

        Parameters
        ----------
        samples : int (default=1)
            Number of simultaneous executions to perform.

        Returns
        -------
        Tensor
        """
        # build type info for the forcing
        types = self.run(semantics.types.functor) 
        forcing = {}
        
        # add the values from meet
        for x in self.meet.variables:
            forcing[x] = self.meet[x]

        # and, if possible, add values from avoid
        for x in self.avoid.variables:
            if types[x] == semantics.types.Discrete(2):
                forcing[x] = 1 - self.avoid[x]

        logger.info(f"Forcing with observations: {forcing}")

        # build the miser functor with the computed forcings
        functor = semantics.miser.factory(samples, forcing=forcing)
        objective = self.objective(functor)

        # build surrogate
        scale = semantics.miser.forcing_scale(objective.dependencies())
        score = semantics.miser.magic_box(objective.dependencies())
        surrogate = objective.value * scale * score

        return surrogate

    def graph(self):
        """Build a graph representation of the story.

        Returns
        -------
        networkx.DiGraph
        """
        objective = self.objective(semantics.graph.functor)
        return semantics.graph.to_graph(objective)