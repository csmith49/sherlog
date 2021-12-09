"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

from ..pipe import Pipeline, Statement, Value, Literal, Identifier
from ..interface import minotaur
from .semantics.core.semiring import DisjointSumSemiring
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import Tensor, stack, tensor
from typing import TypeVar, Mapping, Optional, Callable, List, Iterable

T = TypeVar('T')

class Explanation:
    """Explanations model observations over a generative process."""

    def __init__(self, pipeline : Pipeline, observations : List[Observation], history : History, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None):
        """Construct an explanation."""

        self.pipeline, self.observations, self.history = pipeline, observations, history
        self.locals = locals if locals else {}

    # evaluation framework

    def stub(self) -> Iterable[Statement]:
        """Statement stub for producing final evaluation target from observation results."""
        
        targets = [observation.target(key) for key, observation in enumerate(self.observations)]
        yield Statement("sherlog:target", "semiring:sum", targets)

    def statements(self) -> Iterable[Statement]:
        """Sequence of statements to produce final evaluation target."""

        # the program
        yield from self.pipeline.evaluation_order()
        
        # the observations
        for key, observation in enumerate(self.observations):
            yield from observation.stub(key=key, default=0.0)

        # the target
        yield from self.stub()

    @minotaur("explanation/forcing", kwargs=("force"))
    def forcing(self, parameters : Mapping[str, Tensor], force : bool = True, **kwargs) -> Mapping[str, Value]:
        """Construct the most-specific forcing possible for the explanation."""

        if len(self.observations) == 1 and force:
            observation = self.observations[0]
            result = {}
            for key, value in observation.equality.items():
                if isinstance(value, Literal):
                    result[key] = spyglass.to_tensor(value.value)
                elif isinstance(value, Identifier):
                    result[key] = parameters[value.value]
                else:
                    raise TypeError(f"Observation produced non-evaluable target. [target={value}]")
            return result
        else:
            return {}

    @minotaur("explanation/log-prob", kwargs=("samples"))
    def log_prob(self, parameters : Mapping[str, Tensor], samples : int = 1, **kwargs) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        # step 1: form the semantics
        semantics = spyglass.semantics_factory(
            forcing = self.forcing(parameters, **kwargs),
            semiring = DisjointSumSemiring(),
            locals  = self.locals
        )

        # step 2: evaluate the explanation as many times as requested
        results = []
        for _ in range(samples):
            store = semantics(self.statements(), parameters)
            result = store["sherlog:target"].surrogate
            results.append(result)

        # MC approx. of log-prob
        try:
            result = stack(results).mean().log()
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

    # magic methods

    def __str__(self):
        return f"{self.pipeline}\n{self.observation}"

    def __rich_repr__(self):
        yield self.pipeline
        yield from self.observations
