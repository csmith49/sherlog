"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..engine import Store, Functor, Identifier, Model
from ..logs import get
from .observation import Observation
from .history import History
from .semantics.target import Target
from . import semantics

from typing import Dict, Optional, TypeVar, Any
from itertools import product
from torch import Tensor, stack
from networkx import DiGraph

logger = get("explanation")

T = TypeVar('T')

class Explanation:
    """Explanations combine generative models with observations on identifiers."""

    def __init__(self, model : Model, observation : Observation, history : History):
        """Construct an explanation.

        Parameters
        ----------
        model : Model

        meet : Observation

        history : History
        """
        logger.info(f"Explanation {self} built.")
        self.model, self.observation, self.history = model, observation, history

    @classmethod
    def of_json(cls, json) -> 'Explanation':
        """Constructs an explanation from a JSON encoding.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Explanation
        """
        logger.info(f"Building explanation from serialization: {json}...")
        # delegate to the relevant sub-component JSON parsers
        model = Model.of_json(json["assignments"])
        observation = Observation.of_json(json["meet"])
        history = History.of_json(json["history"])
        # and just pull it all together
        return cls(model, observation, history)

    def run(self, functor : Functor, store : Store, **kwargs) -> Store:
        """Evaluate the explanation in the given functor.

        Parameters
        ----------
        functor : Functor

        store : Store

        **kwargs
            Passed to the functor on execution.

        Returns
        -------
        Store
        """
        # evaluate every assignment in order
        for assignment in self.model.assignments: # ordering handled by this iterable property
            functor.run_assignment(assignment, store, **kwargs)
        return store

    def objective(self, functor : Functor[T], store : Store, **kwargs) -> T:
        """
        Parameters
        ----------
        functor : Functor[T]

        store : Store

        **kwargs
            Passed to the functor during execution.
        
        Returns
        -------
        T
        """
        logger.info(f"{self} building objective in {store} using {functor}.")
        store = self.run(functor, store)
        obs = self.observation.target(functor, store, prefix="sherlog:observation", default=1.0)
        objective = Identifier("sherlog:objective")
        functor.run(objective, "set", [obs], store, **kwargs)
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

    def miser(self, target : Target, store : Store, **kwargs) -> Tensor:
        """Builds a Miser surrogate objective for the satisfaction of the given observation.

        Parameters
        ----------
        target : Target

        store : Store

        **kwargs:
            Passed to the functor during execution.

        Returns
        -------
        Tensor
        """
        # force all the observed variables
        forcing = {}
        for identifier in self.observation.identifiers:
            forcing[identifier] = self.observation[identifier]

        # construct the objective
        functor = semantics.miser.factory(target, forcing=forcing)
        objective = self.objective(functor, store, **kwargs)
        
        # scaling and whatnot handled by .surrogate property
        return objective.surrogate

    def _miser(self, target : Target, store : Store, forcing : Observation, **kwargs) -> Tensor:
        # construct the objective
        functor = semantics.miser.factory(target, forcing=forcing)
        objective = self.objective(functor, store, **kwargs)
        return objective.surrogate

    def log_prob(self, store : Store) -> Tensor:
        """Compute the log-probability of the explanation.

        Parameters
        ----------
        store : Store

        Returns
        -------
        Tensor
        """
        # get all enumerated variables
        possibilities = self.objective(semantics.enumeration.factory(), store)

        # construct the forcings from the observation and the enumerated variables
        forcings = []
        
        # construct cartesian product encoded by enums
        for arg_list in semantics.enumeration.arg_lists(possibilities):
            # and add the observed variables over - any forcings from obs are therefore guaranteed
            for id in self.observation.identifiers:
                arg_list[id] = self.observation[id]
            
            forcings.append(arg_list)
        
        logger.info(f"{self} computing log-prob in {store}: {len(forcings)} possibilities found")

        # get the miser surrogates
        target = semantics.target.EqualityIndicator()
        surrogates = [self._miser(target, store, forcing) for forcing in forcings]

        # and combine
        result = stack(surrogates).sum(dim=-1).log()
        logger.info(f"{self} log-prob in {store}: {result}")
        return result

    def relaxed_log_prob(self, store : Store, temperature : float = 1.0) -> Tensor:
        """Compute the relaxed log-probability of the explanation.

        Converges to `self.log_prob` as temperature tends towards 0.

        Parameters
        ----------
        store : Store

        temperature : float (default=1.0)

        Returns
        -------
        Tensor
        """
        target = semantics.target.RBF(sdev=temperature)
        surrogate_log_prob = self.miser(target, store).log()

        return surrogate_log_prob

    def observation_loss(self, store : Store) -> Tensor:
        """Compute the MSE between the observation and the generated values.

        Parameters
        ----------
        store : Store

        Returns
        -------
        Tensor
        """
        target = semantics.target.MSE()
        surrogate_loss = self.miser(target, store)

        return surrogate_loss