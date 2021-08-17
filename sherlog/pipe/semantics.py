import dataclasses
from .monad import Monad
from .namespace import Namespace
from .value import Value, Identifier, Literal
from .program import Program
from .statement import Statement

from dataclasses import dataclass
from typing import TypeVar, Generic, Mapping, Any
from copy import copy

T = TypeVar('T')

@dataclass
class Semantics(Generic[T]):
    """Semantics package monads and namespaces to provide a complete evaluation environemnt."""

    monad : Monad[T]
    namespace : Namespace[T]

    def _evaluate_value(self, value : Value, context : Mapping[str, T]) -> T:
        """Apply the semantics to a value."""

        if isinstance(value, Identifier):
            return context[value.value]

        if isinstance(value, Literal):
            return self.monad.unit(value.value)
        
        raise TypeError(f"{value} is not an evaluatable value object.")

    def _evaluate_statement(self, statement : Statement, context : Mapping[str, T]) -> T:
        """Apply the semantics to a statement."""

        callable = self.namespace.lookup(statement)
        arguments = [self._evaluate_value(arg, context) for arg in statement.arguments]

        return self.monad.bind(arguments, callable)

    def _evaluate_program(self, program : Program, context : Mapping[str, T]) -> Mapping[str, T]:
        """Apply the semantics to a program."""
        _context = copy(context) # we don't modify the passed-in context

        for statement in program.evaluation_order():
            result = self._evaluate_statement(statement, _context)
            context[statement.target] = result

        return _context

    def evaluate(self, obj : Any, context : Mapping[str, T]):
        """Apply the semantics to the provided object."""

        if isinstance(obj, Value):
            return self._evaluate_value(obj, context)
        
        if isinstance(obj, Statement):
            return self._evaluate_statement(obj, context)

        if isinstance(obj, Program):
            return self._evaluate_program(obj, context)

    def run(self, program : Program, parameters : Mapping[str, Any]) -> Mapping[str, T]:
        """Apply the semantics to a program with the given parameterization."""

        context = {key : self.monad.unit(value) for key, value in parameters.items()}

        return self._evaluate_program(program, context)

    def __call__(self, program : Program, parameters : Mapping[str, Any]) -> Mapping[str, T]:
        """Wrapper for `self.run(program, parameters)`."""

        return self.run(program, parameters)