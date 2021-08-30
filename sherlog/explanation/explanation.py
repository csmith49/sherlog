"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..logs import get

logger = get("explanation")

from ..pipe import Pipeline, Semantics, Statement, Literal, Identifier
from .semantics.core.target import Target, EqualityIndicator
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import tensor, Tensor

from typing import TypeVar, Mapping

T = TypeVar('T')

class Explanation:
    """Explanations model observations over a generative process."""

    def __init__(self, pipeline : Pipeline, observation : Observation, history : History):
        """Construct an explanation."""

        logger.info(f"Explanation {self} built.")
        self.pipeline, self.obervation, self.history = pipeline, observation, history

    def evaluate_observation(self, store : Mapping[str, T], semantics : Semantics[T], target : Target) -> T:
        """Evaluate the observation using the provided target."""

    def evaluate(self, parameters : Mapping[str, Tensor], semantics : Semantics[T], default=0.0) -> Mapping[str, T]:
        """Evaluate the explanation in a context defined by a given set of parameters and semantics."""

        store = semantics(self.pipeline, parameters)

        return semantics.evaluate(self.observation.stub, store)

    def log_prob(self, parameters : Mapping[str, Tensor]) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        sem = spyglass.semantics_factory(
            forcing={},
            target=EqualityIndicator()
        )

        store = self.evaluate(parameters, sem)

        return store["target"].surrogate

    # SERIALIZATION

    @classmethod
    def load(cls, json) -> 'Explanation':
        """Construct an explanation from a JSON-like object."""

        if not json["type"] == "explanation":
            raise TypeError(f"{json} does not represent an explanation.")

        program = Program.load(json["program"])
        observation = Observation.load(json["observation"])
        history = History.load(json["history"])

        return cls(program, observation, history)

    def dump(self):
        """Construct a JSON-like encoding for the explanation."""

        return {
            "type" : "explanation",
            "program" : self.program.dump(),
            "observation" : self.observation.dump(),
            "history" : self.history.dump()
        }
