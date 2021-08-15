from typing import TypeVar, Generic, Iterable, Callable, Dict, Any
from abc import ABC, abstractmethod
from torch import Tensor
from ....program import Evidence

T = TypeVar('T')

class Model(ABC, Generic[T]):
    """Abstract class defining the smallest interface needed to train and evaluate a model."""

    @abstractmethod
    def fit(self, data : Iterable[T], *args, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, datum : T, *args, **kwargs) -> Tensor:
        pass

class Task(Generic[T]):
    """An optimization task."""

    def __init__(self, evidence : Evidence, injection : Callable[[T], Dict[str, Any]]):
        self.evidence = evidence
        self.injection = injection

    def inject(self, datum : T) -> Dict[str, Any]:
        return self.injection(datum)