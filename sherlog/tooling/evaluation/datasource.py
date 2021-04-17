from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sized, Iterable, Callable
from math import floor, ceil

T = TypeVar('T')

class DataSource(ABC, Generic[T]):
    @abstractmethod
    def train(self) -> Iterable[T]: pass

    @abstractmethod
    def test(self) -> Iterable[T]: pass


# data source from a fixed set of data
class SplitDataSource(DataSource):
    def __init__(self, data : Sized[T], train_ratio : float):
        split_index = len(data)
        self._train = data[:floor(split_index)]
        self._test = data[ceil(split_index):]
    
    def train(self): return self._train
    def test(self): return self._test

# data source from a pre-computed train/test split
class TrainTestDataSource(DataSource):
    def __init__(self, train : Iterable[T], test : Iterable[T]):
        self._train = train
        self._test = test

    def train(self): return self._train
    def test(self): return self._test

# data source from a distribution that can produce samples
class DistributionDataSource(DataSource):
    def __init__(self, distribution : Callable[[], T], size : int, train_ratio : float):
        self._distribution = distribution
        self._size = size
        
        train_size = floor(size * train_ratio)
        test_size = ceil(size * (1 - train_ratio))

        self._train = [distribution() for _ in range(train_size)]
        self._test = [distribution() for _ in range(test_size)]

    def train(self): return self._train
    def test(self): return self._test