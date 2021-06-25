from ...logs import get
from ..evaluation.datasource import DataSource
from random import randint

class List:
    def __init__(self, *args):
        self._data = list(args)

class ListDataSource(DataSource):
    def __init__(self, length=None):
        self._length = length

    def _sample(self, length):
        data = []
        for _ in range(length):
            data.append(randint(0, 9))
        return List(*data)

    def training_data(self, *args, quantity=1, **kwargs):
        if self._length:
            length = self._length
        else:
            length = randint(2, 10)
        
        for _ in range(quantity):
            yield self._sample(length)

    def testing_data(self, *args, quantity=1, **kwargs):
        if self._length:
            length = self._length
        else:
            length = randint(2, 10)
        
        for _ in range(quantity):
            yield self._sample(length)
