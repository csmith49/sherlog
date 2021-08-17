from typing import Generic, TypeVar, Any, List, Callable, Mapping
from abc import ABC, abstractmethod

T = TypeVar('T')

class Monad(ABC, Generic[T]):
    """Abstract base class defining the Monad interface."""

    # PRIMARY MONAD DEFINITION

    @abstractmethod
    @staticmethod
    def unit(value : Any) -> T:
        """Embeds a value into the monad."""
        raise NotImplementedError()

    @abstractmethod
    @staticmethod
    def bind(arguments : List[T], callable : Callable[..., T]) -> T: 
        """Inserts unwrapped monadic values (`arguments`) into a monadic expression (`callable`)."""
        raise NotImplementedError()

    # UTILITY FUNCTIONS AND INTERFACE

    @classmethod
    def lift(cls, callable : Callable[..., Any]) -> Callable[..., T]:
        """Lift a callable to a monadic expression."""
        def lifted(*args):
            return cls.unit(callable(*args))
        return lifted