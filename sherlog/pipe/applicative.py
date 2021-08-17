from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, List, Any
from functools import partial

T = TypeVar("T")

@dataclass
class Applicative(Generic[T]):
    """An applicative functor with wrapped values of type `T`."""
    
    pure : Callable[..., T]
    lift : Callable[[Callable, List[T]], T]