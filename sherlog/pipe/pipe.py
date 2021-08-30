from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, List

T = TypeVar("T")

@dataclass
class Pipe(Generic[T]):
    """A monad-style context with wrapped values of type `T`.
    
    >>> unit : * -> T
    >>> bind : (* -> T) -> * -> T
    """
    
    unit : Callable[..., T]
    bind : Callable[[Callable[..., T], List[T]], T]