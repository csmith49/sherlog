from .datasource import DataSource
from typing import TypeVar, Iterable, Tuple, Callable

T = TypeVar('T')
U = TypeVar('U')

class Product(DataSource):
    def __init__(self, *sources : DataSource[T]):
        self._sources = sources

    def training_data(self, *args, **kwargs) -> Iterable[Tuple[T]]:
        yield from zip(*(s.training_data(*args, **kwargs) for s in self._sources))

    def testing_data(self, *args, **kwargs):
        yield from zip(*(s.testing_data(*args, **kwargs) for s in self._sources))

class Transform(DataSource):
    def __init__(self, callable : Callable[[T], U], source : DataSource[T]):
        self._callable = callable
        self._source = source

    def training_data(self, *args, **kwargs) -> Iterable[U]:
        for data in self._source.training_data(*args, **kwargs):
            yield self._callable(data)
        
    def testing_data(self, *args, **kwargs) -> Iterable[U]:
        for data in self._source.testing_data(*args, **kwargs):
            yield self._callable(data)

class Map(DataSource):
    def __init__(self, callable, *sources):
        self._callable = callable
        self._source = Product(*sources)

    def training_data(self, *args, **kwargs):
        for datum in self._source.training_data(*args, **kwargs):
            yield self._callable(*datum)
    
    def testing_data(self, *args, **kwargs):
        for datum in self._source.testing_data(*args, **kwargs):
            yield self._callable(*datum)