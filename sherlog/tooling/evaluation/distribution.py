from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable

T = TypeVar('T')

class Distribution(ABC, Generic[T]):
    @abstractmethod
    def fit(self, data : Iterable[T]): pass

    @abstractmethod
    def sample(self) -> T: pass

    @abstractmethod
    def log_likelihood(self, sample : T) -> float: pass

    @abstractmethod
    def conditional(self, event : T, condition : T) -> float: pass