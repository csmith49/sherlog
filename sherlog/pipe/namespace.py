from .statement import Statement

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Mapping

T = TypeVar('T')

# ABC

class Namespace(ABC, Generic[T]):
    """Abstract base class for namespaces."""

    @abstractmethod
    def lookup(self, statement : Statement) -> Callable[..., T]:
        raise NotImplementedError(f"{self} has no defined lookup method.")

# SUBCLASSES

class DirectMappingNamespace(Namespace[T]):
    """Direct mapping namespaces map the `function` attribute of statements directly to callables."""

    def __init__(self, mapping : Mapping[str, Callable[..., T]]):
        """Construct a direct mapping namespace with the provided mapping."""

        self.mapping = mapping

    def lookup(self, statement : Statement) -> Callable[..., T]:
        """Lookup a callable."""

        return self.mapping[statement.function]

class DynamicNamespace(Namespace[T]):
    """Dynamic namespaces wrap arbitrary functions that map statements to callables."""

    def __init__(self, callable_generator : Callable[[Statement], Callable[..., T]]):
        """Construct a dynamic namespace from a callable generator."""

        self.callable_generator = callable_generator

    def lookup(self, statement : Statement) -> Callable[..., T]:
        """Lookup a callable."""

        return self.callable_generator(statement)