from .pipe import Pipe
from .namespace import Namespace
from .value import Value, Identifier, Literal
from .pipeline import Pipeline
from .statement import Statement

from dataclasses import dataclass
from typing import TypeVar, Generic, Mapping, Any, List, Callable, Iterable
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
            try:
                return context[value.value]
            except KeyError:
                pass

        if isinstance(value, Literal):
            return self.pipe.unit(value.value)
        
        raise TypeError(f"{value} is not an evaluatable value object in the given context.")

    def _evaluate_statement(self, statement : Statement, context : Mapping[str, T]) -> T:
        """Apply the semantics to a statement."""

        callable = self.namespace.lookup(statement)
        arguments = [self._evaluate_value(arg, context) for arg in statement.arguments]

        result = self.pipe.bind(callable, arguments)

        return result

    def _evaluate_statements(self, statements : Iterable[Statement], context : Mapping[str, T]) -> Mapping[str, T]:
        """Apply the semantics to a program."""

        _context = copy(context) # we don't modify the passed-in context

        for statement in statements:
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
            return self._evaluate_pipeline(obj, context)

    def run(self, statements : Iterable[Statement], parameters : Mapping[str, Any]) -> Mapping[str, T]:
        """Apply the semantics to a program with the given parameterization."""

        context = {key : self.pipe.unit(value) for key, value in parameters.items()}

        return self._evaluate_statements(statements, context)

    def __call__(self, statements : Iterable[Statement], parameters : Mapping[str, Any]) -> Mapping[str, T]:
        """Alias for `self.run(pipeline, parameters)`."""

        return self.run(statements, parameters)