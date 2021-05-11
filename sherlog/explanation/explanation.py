"""Explanation."""

from ..engine import Store, value
from .observation import Observation
from ..engine import Model
from ..logs import get
from . import semantics

logger = get("explanation")

class Explanation:
    def __init__(self, model, meet, avoids, external=()):
        logger.info(f"Explanation {self} built.")
        self.model = model
        self.meet = meet
        self.avoids = avoids
        self._external = external

    @classmethod
    def of_json(cls, json, external=()):
        logger.info(f"Building explanation from serialization: {json}...")
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        avoids = [Observation.of_json(obs) for obs in json["avoid"]]
        return cls(model, meet, avoids, external=external)

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
        avoids = [obs.equality(store, functor, prefix=f"sherlog:avoid:{i}", default=0.0) for i, obs in enumerate(self.avoids)]
        avoid = value.Variable("sherlog:avoid")
        if avoids:
            functor.run(avoid, "or", avoids, store)
        else:
            functor.run(avoid, "set", [0.0], store)

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

        logger.info("Evaluating types and forcing values...")
        # build type info for the forcing
        # types = self.run(semantics.types.functor)
        forcing = {}

        # add the values from meet
        for variable in self.meet.variables:
            forcing[variable] = self.meet[variable]

        # and, if possible, add values from avoid
        # for avoid in self.avoids:
        #     for variable in avoid.variables:
        #         if types[variable] == semantics.types.Discrete(2):
        #             forcing[variable] = 1 - avoid[variable]

        logger.info(f"Forcing with observations: {forcing}")

        # build the miser functor with the computed forcings
        functor = semantics.miser.factory(samples, forcing=forcing)
        objective = self.objective(functor)

        # build surrogate
        scale = semantics.miser.forcing_scale(objective.dependencies()).detach() # if we don't detach here, we "overcount" forced instances
        score = semantics.miser.magic_box(objective.dependencies(), samples)
        surrogate = objective.value * scale * score

        logger.info(f"Objective, likelihood ratio, and magic box: {objective} / {scale} / {score}")
        logger.info(f"Miser surrogate objective: {surrogate}")

        return surrogate

    def graph(self):
        """Build a graph representation of the story.

        Returns
        -------
        networkx.DiGraph
        """
        objective = self.objective(semantics.graph.functor)
        return semantics.graph.to_graph(objective)
