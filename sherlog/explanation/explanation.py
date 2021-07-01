"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..engine import Store, Functor, Identifier, Model
from ..logs import get
from .observation import Observation
from .history import History
from .semantics.target import Target
from . import semantics

from typing import Dict, Optional, TypeVar, Any
from torch import Tensor
from networkx import DiGraph

logger = get("explanation")

T = TypeVar('T')

class Explanation:
    """Explanations combine generative models with observations on identifiers."""

    def __init__(self, model : Model, meet : Observation, history : History, *builtins):
        """Construct an explanation.

        Parameters
        ----------
        model : Model

        meet : Observation

        history : History

        *builtins : Dict[str, Any]
        """
        logger.info(f"Explanation {self} built.")
        self.model, self.meet, self.history = model, meet, history
        # may not always get builtins, need a default
        self.builtins = list(builtins) if builtins else []

    @classmethod
    def of_json(cls, json, *builtins) -> 'Explanation':
        """Constructs an explanation from a JSON encoding.

        Parameters
        ----------
        json : JSON-like object

        *builtins : Dict[str, Any]

        Returns
        -------
        Explanation
        """
        logger.info(f"Building explanation from serialization: {json}...")
        # delegate to the relevant sub-component JSON parsers
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        history = History.of_json(json["history"])
        # and just pull it all together
        return cls(model, meet, history, *builtins)

    def store(self, *namespaces) -> Store:
        """Store initialized with builtin mapping.

        Parameters
        ----------
        *namespaces : Dict[str, Any]

        Returns
        -------
        Store
        """
        return Store(*self.builtins, *namespaces)

    def run(self, functor : Functor, namespace : Optional[Dict[str, Any]] = None, **kwargs) -> Store:
        """Evaluate the explanation in the given functor.

        Parameters
        ----------
        functor : Functor

        namespace : Optional[Dict[str, Any]]

        **kwargs
            Passed to the functor on execution.

        Returns
        -------
        Store
        """
        # build the store as normal
        store = self.store(namespace) if namespace else self.store()
        # and evaluate every assignment in order
        for assignment in self.model.assignments:
            functor.run_assignment(assignment, store, **kwargs)
        return store

    def objective(self, functor : Functor[T], namespace : Optional[Dict[str, Any]] = None, **kwargs) -> T:
        """
        Parameters
        ----------
        functor : Functor[T]

        namespace : Optional[Dict[str, Any]]

        **kwargs
            Passed to the functor during execution.
        
        Returns
        -------
        T
        """
        store = self.run(functor, namespace=namespace)
        observation = self.meet.target(store, functor, prefix="sherlog:observation", default=1.0)
        objective = Identifier("sherlog:objective")
        functor.run(objective, "set", [observation], store, **kwargs)
        return store[objective]

    # APPLICATIONS OF OBJECTIVE BUILDING

    def graph(self) -> DiGraph:
        """Build a directed graph representation of the explanation.
        
        Returns
        -------
        DiGraph
        """
        objective = self.objective(semantics.graph.functor)
        return semantics.graph.to_graph(objective)

    def miser(self, target : Target, namespace : Optional[Dict[str, Any]] = None, **kwargs) -> Tensor:
        """Builds a Miser surrogate objective for the satisfaction of the given observation.

        Parameters
        ----------
        target : Target

        namespace : Optional[Dict[str, Any]]

        **kwargs:
            Passed to the functor during execution.

        Returns
        -------
        Tensor
        """
        # force all the observed variables
        forcing = {}
        for identifier in self.meet.identifiers:
            forcing[identifier] = self.meet[identifier]

        # construct the objective
        functor = semantics.miser.factory(target, forcing=forcing)
        objective = self.objective(functor, namespace=namespace, **kwargs)
        
        # scaling and whatnot handled by .surrogate property
        return objective.surrogate

    def log_prob(self, namespace : Optional[Dict[str, Any]] = None) -> Tensor:
        """Compute the log-probability of the explanation.

        Parameters
        ----------
        namespace : Dict[str, Any] (default={})

        Returns
        -------
        Tensor
        """
        # log-prob done via expectation of indicator function
        target = semantics.target.EqualityIndicator()
        surrogate_log_prob = self.miser(target, namespace=namespace).log()

        return surrogate_log_prob

    def relaxed_log_prob(self, temperature : float = 1.0, namespace={}) -> Tensor:
        """Compute the relaxed log-probability of the explanation.

        Converges to `self.log_prob` as temperature tends towards 0.

        Parameters
        ----------
        namespace : Dict[str, Any] (default=[])

        temperature : float (default=1.0)

        Returns
        -------
        Tensor
        """
        target = semantics.target.RBF(sdev=temperature)
        surrogate_log_prob = self.miser(self.meet, target, namespace=namespace).log()

        return surrogate_log_prob

    def observation_loss(self, namespace={}) -> Tensor:
        """Compute the MSE between the observation and the generated values.

        Parameters
        ----------
        namespace : Dict[str, Any] (default={})

        Returns
        -------
        Tensor
        """
        target = semantics.target.MSE()
        surrogate_loss = self.miser(self.meet, target, namespace=namespace)

        return surrogate_loss