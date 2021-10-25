"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..pipe import Pipeline, Statement, Identifier, Literal
from ..interface.instrumentation import minotaur
from .observation import Observation
from .history import History

from .semantics.core import builtin
from .semantics.core import distribution

from torch import Tensor, tensor
from typing import TypeVar, Mapping, Optional, Callable, List, Iterable
from copy import copy
from storch import Tensor as StorchTensor

import storch.method as method

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

    def statements(self) -> Iterable[Statement]:
        yield from self.pipeline.evaluation_order()
        for index, observation in enumerate(self.observations):
            yield from observation.stub(key=index, default=0.0)
        yield from self.stub()

    @minotaur("evaluate")
    def evaluate(self, parameters : Mapping[str, Tensor]) -> Mapping[str, T]:
        """Evaluate the explanation."""

        store = copy(parameters)

        for statement in self.statements():
            # evaluate the arguments in the context of the store
            def arguments():
                for argument in statement.arguments:
                    if isinstance(argument, Identifier):
                        value = store[argument.value]
                        if isinstance(value, (StorchTensor, Tensor)):
                            yield value
                        else:
                            yield tensor(value)
                    elif isinstance(argument, Literal):
                        yield tensor(argument.value)
                    else:
                        raise TypeError(f"Object not evaluable in the given parameterization. [object={argument}]")

            # case 1: function is builtin
            if statement.function in builtin.supported_builtins():
                callable = builtin.lookup(statement.function)
                store[statement.target] = callable(*arguments())

            # case 2: function is local
            elif statement.function in self.locals.keys():
                callable = self.locals[statement.function]
                store[statement.target] = callable(*arguments())

            # case 3: function is a supported distribution
            elif statement.function in distribution.supported_distributions():
                dist = distribution.lookup_constructor(statement.function)(*arguments())

                if dist.has_rsample:
                    sample = method.Reparameterization(statement.target)(dist)

                else:
                    sample = method.ScoreFunction(statement.target)(dist)

                store[statement.target] = sample

            # case 4: function cannot be found
            else:
                raise KeyError(f"No implementation found for the given function identifier. [function={statement.function}]")

        return store

    @minotaur("explanation log-prob")
    def log_prob(self, parameters : Mapping[str, Tensor], width : int = 1) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        store = self.evaluate(parameters)

        return store["sherlog:target"].log()

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