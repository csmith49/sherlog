from .datasource import DataSource
from typing import TypeVar, Iterable, Tuple, Callable

T = TypeVar('T')
U = TypeVar('U')

class Product(DataSource):
    """Data source combining other sources with a Cartesian product."""

    def __init__(self, *sources : DataSource[T]):
        """Constructs a data source by combining all provided sources with a Cartesian product.

        Parameters
        ----------
        *sources : DataSource[T]
        """
        self._sources = sources

    def training_data(self, *args, **kwargs) -> Iterable[Tuple[T]]:
        yield from zip(*(s.training_data(*args, **kwargs) for s in self._sources))

    def testing_data(self, *args, **kwargs):
        yield from zip(*(s.testing_data(*args, **kwargs) for s in self._sources))

class Transform(DataSource):
    """Data source transforming an existing source with a callable."""

    def __init__(self, callable : Callable[[T], U], source : DataSource[T]):
        """Constructs a data source by transforming an existing source.

        Parameters
        ----------
        callable : Callablue[[T], U]
        source : DataSource[T]
        """
        self._callable = callable
        self._source = source

    def training_data(self, *args, **kwargs) -> Iterable[U]:
        for data in self._source.training_data(*args, **kwargs):
            yield self._callable(data)
        
    def testing_data(self, *args, **kwargs) -> Iterable[U]:
        for data in self._source.testing_data(*args, **kwargs):
            yield self._callable(data)

class Map(DataSource):
    """Data source combining other sources with a provided callable."""

    def __init__(self, callable, *sources):
        """Constructs a data source by combining other sourecs with a provided callable.

        Parameters
        ----------
        callable : Callable[[T_1, T_2, ..., T_k], U]
        *sources : DataSource[T_i]
        """
        self._callable = callable
        self._source = Product(*sources)

    def training_data(self, *args, **kwargs):
        for datum in self._source.training_data(*args, **kwargs):
            yield self._callable(*datum)
    
    def testing_data(self, *args, **kwargs):
        for datum in self._source.testing_data(*args, **kwargs):
            yield self._callable(*datum)