from typing import TypeVar, Generic, Iterable
from abc import ABC, abstractmethod
from torch import Tensor

T = TypeVar('T')

class Model(ABC, Generic[T]):
    """Abstract class defining the smallest interface needed to train and evaluate a model."""

    @abstractmethod
    def fit(self, samples : Iterable[T], *args, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, sample : T, *args, **kwargs) -> Tensor:
        pass