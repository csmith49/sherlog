"""Explanations are the central explanatory element in Sherlog. They capture sufficient generative constraints to ensure a particular outcome."""

<<<<<<< HEAD
from ..pipe import Pipeline, Statement
=======
from ..pipe import Pipeline, Semantics, Statement
from ..interface import print
>>>>>>> 99820b2263b8b77b9a67ce22b16fa0a8150b5ce1
from ..interface.instrumentation import minotaur
from .semantics.core.target import EqualityIndicator
from .semantics import spyglass
from .observation import Observation
from .history import History

from torch import Tensor, stack, tensor
from typing import TypeVar, Mapping, Optional, Callable, List, Iterable
<<<<<<< HEAD
=======
from itertools import chain
>>>>>>> 99820b2263b8b77b9a67ce22b16fa0a8150b5ce1

T = TypeVar('T')

class Explanation:
    """Explanations model observations over a generative process."""

    def __init__(self, pipeline : Pipeline, observations : List[Observation], history : History, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None):
        """Construct an explanation."""

        self.pipeline, self.observations, self.history = pipeline, observations, history
        self.locals = locals if locals else {}

<<<<<<< HEAD
    # evaluation framework

    def target_stub(self) -> Iterable[Statement]:
        """Statement stub for producing final evaluation target from observation results."""
        
        targets = [observation.target_identifier(key) for key, observation in enumerate(self.observations)]
        yield Statement("sherlog:target", "max", targets)

    def statements(self) -> Iterable[Statement]:
        """Sequence of statements to produce final evaluation target."""

        # the program
        yield from self.pipeline.evaluation_order()
        
        # the observations
        for key, observation in enumerate(self.observations):
            yield from observation.target_stub(key=key, default=0.0)

        # the target
        yield from self.target_stub()

    @minotaur("explanation log-prob")
    def log_prob(self, parameters : Mapping[str, Tensor], samples : int = 1) -> Tensor:
        """Compute the log-probability of the explanation generating the observations."""

        # step 1: form the semantics
        semantics = spyglass.semantics_factory(
            observation = Observation({}), # can't really force with multiple observations
            target = EqualityIndicator(),  # using 0-1 indicator value for target
            locals = self.locals
        )

        # step 2: evaluate the explanation as many times as requested
        results = []
        for _ in range(samples):
            store = semantics(self.statements(), parameters)
            result = store["sherlog:target"].surrogate
            results.append(result)

        # MC approx. of log-prob
=======
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
>>>>>>> 99820b2263b8b77b9a67ce22b16fa0a8150b5ce1
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