"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..pipe import Pipeline, Semantics
from ..interface.instrumentation import minotaur
from .semantics.core.target import EqualityIndicator
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import Tensor, stack, tensor
from typing import TypeVar, Mapping, Optional, Callable, List

T = TypeVar('T')

class Explanation:
    """Explanations model observations over a generative process."""

    def __init__(self, pipeline : Pipeline, observations : List[Observation], history : History, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None):
        """Construct an explanation."""

        self.pipeline, self.observations, self.history = pipeline, observations, history
        self.locals = locals if locals else {}

    @minotaur("evaluate")
    def evaluate(self, parameters : Mapping[str, Tensor], semantics : Semantics[T], default=0.0) -> Mapping[str, T]:
        """Evaluate the explanation."""

        store = semantics(self.pipeline, parameters)
        
        for index, observation in enumerate(self.observations):
            for statement in observation.stub(default=default, key=index):
                store[statement.target] = semantics.evaluate(statement, store)
    
        return store

    @minotaur("explanation log-prob")
    def log_prob(self, parameters : Mapping[str, Tensor]) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        sem = spyglass.semantics_factory(
            observation=Observation({}),
            target=EqualityIndicator(),
            locals=self.locals
        )
        store = self.evaluate(parameters, sem)

        results = []
        for index, _ in enumerate(self.observations):
            result = stack([clue.surrogate for clue in store[f"sherlog:target:{index}"]]).mean()
            if result > 0:
                results.append(result.log())

        try:
            result = stack(results).max()
        except RuntimeError:
            result = tensor(0.0)

        minotaur["result"] = result.item()

        return result

    # SERIALIZATION

    @classmethod
    def of_json(cls, json, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None) -> 'Explanation':
        """Construct an explanation from a JSON-like object."""

        if not json["type"] == "explanation":
            raise TypeError(f"{json} does not represent an explanation.")

        program = Pipeline.of_json(json["pipeline"])
        observations = [Observation.of_json(ob) for ob in json["observations"]]
        history = History.of_json(json["history"])

        return cls(program, observations, history, locals=locals)

    def to_json(self):
        """Construct a JSON-like encoding for the explanation."""

        return {
            "type" : "explanation",
            "program" : self.program.dump(),
            "observations" : self.observation.dump(),
            "history" : self.history.dump()
        }


    # magic methods

    def __str__(self):
        return f"{self.pipeline}\n{self.observation}"