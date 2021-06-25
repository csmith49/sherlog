from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable

T = TypeVar('T')

class DataSource(ABC, Generic[T]):
    @abstractmethod
    def training_data(self, *args, **kwargs) -> Iterable[T]:
        pass

    @abstractmethod
    def testing_data(self, *args, **kwargs) -> Iterable[T]:
        pass
