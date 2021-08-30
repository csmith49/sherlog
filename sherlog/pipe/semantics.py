import dataclasses
from .pipe import Pipe
from .namespace import Namespace
from .value import Value, Identifier, Literal
from .pipeline import Pipeline
from .statement import Statement

from dataclasses import dataclass
from typing import TypeVar, Generic, Mapping, Any, List, Callable
from copy import copy

T = TypeVar('T')

@dataclass
class Semantics(Generic[T]):
    """Semantics package monads and namespaces to provide a complete evaluation environemnt."""

    pipe : Pipe[T]
    namespace : Namespace[T]

    def _evaluate_value(self, value : Value, context : Mapping[str, T]) -> T:
        """Apply the semantics to a value."""

        if isinstance(value, Identifier):
            return context[value.value]

        if isinstance(value, Literal):
            return self.pipe.unit(value.value)
        
        raise TypeError(f"{value} is not an evaluatable value object.")

    def _evaluate_statement(self, statement : Statement, context : Mapping[str, T]) -> T:
        """Apply the semantics to a statement."""

        callable = self.namespace.lookup(statement)
        arguments = [self._evaluate_value(arg, context) for arg in statement.arguments]

        return self.pipe.bind(callable, arguments)

    def _evaluate_program(self, pipeline : Pipeline, context : Mapping[str, T]) -> Mapping[str, T]:
        """Apply the semantics to a program."""
        _context = copy(context) # we don't modify the passed-in context

        for statement in pipeline.evaluation_order():
            result = self._evaluate_statement(statement, _context)
            _context[statement.target] = result

        return _context

    def evaluate(self, obj : Any, context : Mapping[str, T]):
        """Apply the semantics to the provided object."""

        if isinstance(obj, Value):
            return self._evaluate_value(obj, context)
        
        if isinstance(obj, Statement):
            return self._evaluate_statement(obj, context)

        if isinstance(obj, Pipeline):
            return self._evaluate_program(obj, context)

    def run(self, pipeline : Pipeline, parameters : Mapping[str, Any]) -> Mapping[str, T]:
        """Apply the semantics to a program with the given parameterization."""

        context = {key : self.pipe.unit(value) for key, value in parameters.items()}

        return self._evaluate_program(pipeline, context)

    def __call__(self, pipeline : Pipeline, parameters : Mapping[str, Any]) -> Mapping[str, T]:
        """Alias for `self.run(pipeline, parameters)`."""

        return self.run(pipeline, parameters)

class NDSemantics(Semantics[List[T]]):
    """Semantics for non-deterministic evaluation."""

    def __init__(self, pipe : Pipe[T], namespace : Namespace[List[T]]):
        """Lift the provided pipe to support non-determinism and construct the resulting semantics."""

        # unit closure
        def unit(value : Any) -> List[T]:
            """Lift a value to a non-deterministic pipe value."""

            return [pipe.unit(value)]

        # bind closure
        def bind(callable : Callable[..., List[T]], arguments : List[List[T]]) -> List[T]:
            """Apply a non-deterministic function to a set of non-deterministic arguments."""

            # generator loops over all arguments in the cart-prod and uses pipe.bind to stitch context together
            def gen():
                for args in zip(*arguments):
                    for result in callable(*args):
                        yield pipe.bind(lambda *_: result, args)

            return list(gen())

        # store the constructed pipe locally
        super().__init__(Pipe(unit, bind), namespace)