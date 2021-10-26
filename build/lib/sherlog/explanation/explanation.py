"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..pipe import Pipeline, Semantics, Statement
from ..interface import print
from ..interface.instrumentation import minotaur
from .semantics.core.target import EqualityIndicator
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import Tensor, stack, tensor
from typing import TypeVar, Mapping, Optional, Callable, List, Iterable
from itertools import chain

T = TypeVar('T')

class Explanation:
    """Explanations model observations over a generative process."""

    def __init__(self, pipeline : Pipeline, observations : List[Observation], history : History, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None):
        """Construct an explanation."""

        self.pipeline, self.observations, self.history = pipeline, observations, history
        self.locals = locals if locals else {}

    # EVALUATION FRAMEWORK

    def stub(self) -> Iterable[Statement]:
        """Stub for combining observations."""

        targets = [obs.target(key) for key, obs in enumerate(self.observations)]
        yield Statement("sherlog:target", "max", targets)

    @minotaur("evaluate")
    def evaluate(self, parameters : Mapping[str, Tensor], semantics : Semantics[T], default=0.0) -> Mapping[str, T]:
        """Evaluate the explanation."""

        # run the generative process marginalized to the explanation
        store = semantics(self.pipeline, parameters)

        print(store)

        # collect all further statements from the observation/explanation stubs
        statements = chain(
            *[obs.stub(key, default=default) for key, obs in enumerate(self.observations)],
            self.stub()
        )

        # evaluate the statmenets
        for statement in statements:
            print(statement)
            print(semantics.evaluate(statement, store))
            store[statement.target] = semantics.evaluate(statement, store)

        return store

    @minotaur("explanation log-prob")
    def log_prob(self, parameters : Mapping[str, Tensor], width : int = 1) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        # dyanmically construct the semantics and execute with the given parameters
        semantics = spyglass.semantics_factory(
            target=EqualityIndicator(),
            locals=self.locals,
            width=width
        )
        store = self.evaluate(parameters, semantics)

        print(store["sherlog:target"])

        # observation map
        for index, observation in enumerate(self.observations):
            result = store[f"sherlog:target:{index}"]
            print(observation)

        # PROBLEM IS HERE, AT LEAST IN PART
        results = []
        for index, _ in enumerate(self.observations):
            # SHOULD BE MAX, NOT MEAN - EXPLAINS LOW LOG-PROB IN SUCCESSFUL BENCHMARKS?
            result = stack([clue.surrogate for clue in store[f"sherlog:target:{index}"]]).max()
            if result > 0:
                results.append(result.log())

        # CAN WE RETURN SOME VALUE OTHER THAN ZERO? NO WAY TO PROP GRADIENTS WITH DICE THIS WAY
        # IDEA 1: CHANGE SCORE TO -1, 1 AND APPLY LINEAR TRANSFORMATION 0.5 X + 0.5 BEFORE TAKING LOG
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

    def __rich_repr__(self):
        yield self.pipeline
        yield from self.observations