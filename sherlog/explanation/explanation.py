"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..interface.logs import get

logger = get("explanation", verbose=True)

from ..pipe import Pipeline, Semantics
from .semantics.core.target import EqualityIndicator
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import Tensor, stack

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

        logger.info(f"Evaluating explanation {self} in {semantics} with parameters: {parameters}.")
        store = semantics(self.pipeline, parameters)
        
        logger.info(f"Building observation target for {self}...")
        for statement in self.observation.stub(default=default):
            store[statement.target] = semantics.evaluate(statement, store)
    
        return store

    def log_prob(self, parameters : Mapping[str, Tensor]) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        logger.info(f"Constructing spyglass semantics with forcing {self.observation}...")
        sem = spyglass.semantics_factory(
            observation=self.observation,
            target=EqualityIndicator(),
            locals=self.locals
        )
        store = self.evaluate(parameters, sem)
        result = stack([clue.surrogate for clue in store["sherlog:target"]]).mean().log()

        logger.info(f"Resulting log-prob: {result}.")
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
