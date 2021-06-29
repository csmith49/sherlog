from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable, Tuple, Optional, List
from itertools import islice

T = TypeVar('T')

class DataSource(ABC, Generic[T]):
    """Abstract class defining a source of training and testing data."""

    @abstractmethod
    def training_data(self, *args, **kwargs) -> Iterable[T]:
        pass

    @abstractmethod
    def testing_data(self, *args, **kwargs) -> Iterable[T]:
        pass

    def get(self, *args, train_size : Optional[int] = None, test_size : Optional[int] = None, **kwargs) -> Tuple[List[T], List[T]]:
        """Get train/test data with one call. The quantity of data can be controlled with the optional `train_size` and `test_size` parameters.

        `*args` and `**kwargs` are passed directly to the data generation methods.
        Parameters
        ----------
        train_size : Optional[int]
        test_size : Optional[int]

        Returns
        -------
        Tuple[List[T], List[T]]
        """
        train = list(islice(self.training_data(*args, **kwargs), train_size))
        test = list(islice(self.testing_data(*args, **kwargs), test_size))

        return (train, test)