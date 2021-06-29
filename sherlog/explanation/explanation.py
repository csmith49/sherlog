"""Explanation."""

from ..engine import Store, value
from .observation import Observation
from ..engine import Model
from .history import History
from ..logs import get
from . import semantics
from itertools import chain
import torch

logger = get("explanation")

class Explanation:
    """Explanations are the central explanatory element in Sherlog. They capture sufficient generative properties to ensure a particular outcome."""

    def __init__(self, model, meet, history, external=()):
        """Constructs an explanation.

        Parameters
        ----------
        model : Model
        meet : Observation
        history : History
        external : Mapping[str, Any] (default={})
        """
        logger.info(f"Explanation {self} built.")
        self.model = model
        self.meet = meet
        self.history = history
        self._external = external

    @classmethod
    def of_json(cls, json, external=()):
        """Constructs an explanation from a JSON encoding.

        Parameters
        ----------
        json : JSON-like object
        external : Mapping[str, Any] (default={})

        Returns
        -------
        Explanation
        """
        logger.info(f"Building explanation from serialization: {json}...")
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        history = History.of_json(json["history"])
        return cls(model, meet, history, external=external)

    def store(self, *namespaces):
        """Store initialized with external mapping.

        Parameters
        ----------
        *namespaces : Dict[str, Any]

        Returns
        -------
        Store
        """
        return Store(external=chain(self._external, namespaces))

    def run(self, functor, wrap_args={}, fmap_args={}, parameters={}, namespace={}):
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
        store = self.store(namespace)
        for assignment in self.model.assignments:
            functor.run_assignment(
                assignment,
                store,
                wrap_args=wrap_args,
                fmap_args=fmap_args,
                parameters=parameters)
        return store

    def graph(self):
        """Build a graph representation of the story.

        Returns
        -------
        networkx.DiGraph
        """
        objective = self.objective(semantics.graph.functor)
        return semantics.graph.to_graph(objective)

    def objective(self, functor, observation, namespace={}):
        """Build computation graph for the satisfaction of the given observation in the given functor.

        Parameters
        ----------
        functor : Functor
        observation : Observation

        Returns
        -------
        Functor.t
        """
        store = self.run(functor, namespace=namespace)

        obs = observation.equality(store, functor, prefix="sherlog:observation", default=1.0)
        objective = value.Variable("sherlog:objective")
        functor.run(objective, "set", [obs], store)

        return store[objective]

    def miser(self, observation, samples=1, namespace={}):
        """Builds a Miser surrogate objective for the satisfaction of the given observation.

        Parameters
        ----------
        observation : Observation
        samples : int (default=1)

        Returns
        -------
        Miser.t (with batch dimension of size `samples`)
        """
        # force all the observed variables
        forcing = {}
        for variable in observation.variables:
            forcing[variable] = observation[variable]

        # construct the objective
        functor = semantics.miser.factory(samples, forcing=forcing)
        objective = self.objective(functor, observation, namespace=namespace)

        # scale and score appropriately
        scale = semantics.miser.forcing_scale(objective.dependencies()).detach() # if we don't detach, we seem to "overcount" forced instances
        score = semantics.miser.magic_box(objective.dependencies(), samples)
        surrogate = objective.value * scale * score

        logger.info(f"MISER cost: {objective.value}, IS score: {scale}, magic box: {score}")

        return surrogate

    def log_prob(self, parameterization, samples=1, namespace={}):
        """Computes log-likelihood of the explanation.

        Parameters
        ----------
        parameterization : Parameterization
        samples : int (default=1)

        Returns
        -------
        Tensor
        """
        p = self.miser(self.meet, samples=samples, namespace=namespace).log()
        q = self.history.log_prob(parameterization)
        logger.info(f"Log-prob: {p}, posterior scaling: {q}")
        return p - q