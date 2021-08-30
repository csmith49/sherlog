"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..logs import get

logger = get("explanation")

from ..pipe import Pipeline, Semantics
from .semantics.core.target import EqualityIndicator
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import tensor, Tensor

from typing import TypeVar, Mapping, Optional, Any, Callable

T = TypeVar('T')

class Explanation:
    """Explanations model observations over a generative process."""

    def __init__(self, pipeline : Pipeline, observation : Observation, history : History, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None):
        """Construct an explanation."""

        logger.info(f"Explanation {self} built.")
        self.pipeline, self.observation, self.history = pipeline, observation, history
        self.locals = locals if locals else {}

    def evaluate(self, parameters : Mapping[str, Tensor], semantics : Semantics[T], default=0.0) -> Mapping[str, T]:
        """Evaluate the explanation."""

        store = semantics(self.pipeline, parameters)
        for statement in self.observation.stub(default=default):
            store[statement.target] = semantics.evaluate(statement, store)
        return store

    def log_prob(self, parameters : Mapping[str, Tensor]) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        sem = spyglass.semantics_factory(
            forcing=self.observation.mapping,
            target=EqualityIndicator(),
            locals=self.locals
        )
        store = self.evaluate(parameters, sem)
        result = store["sherlog:target"].surrogate.log()

        return result

    # SERIALIZATION

    @classmethod
    def of_json(cls, json, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None) -> 'Explanation':
        """Construct an explanation from a JSON-like object."""

        if not json["type"] == "explanation":
            raise TypeError(f"{json} does not represent an explanation.")

        program = Pipeline.of_json(json["pipeline"])
        observation = Observation.of_json(json["observation"])
        history = History.of_json(json["history"])

        return cls(program, observation, history, locals=locals)

    def to_json(self):
        """Construct a JSON-like encoding for the explanation."""

        return {
            "type" : "explanation",
            "program" : self.program.dump(),
            "observation" : self.observation.dump(),
            "history" : self.history.dump()
        }
