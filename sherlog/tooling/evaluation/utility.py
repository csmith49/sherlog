"""Utilities for constructing and accessing data sources go here."""

from typing import Iterable, List, TypeVar, Generic
from itertools import zip_longest
from random import shuffle
from collections import namedtuple

T = TypeVar('T')

class Batch(Generic[T]):
    def __init__(self, *data, epoch=0, index=0):
        self.data = list(data)
        self.epoch, self.index = epoch, index

    def __iter__(self):
        return self.data

    @property
    def identifier(self):
        return f"batch:{self.index}:{self.epoch}"

def minibatch(data : List[T], batch_size : int, epochs : int = 1) -> Iterable[Batch[T]]:
    """Convert a list of data into batches of a given size.

    Parameters
    ----------
    data : List[T]
    batch_size : int
    epochs : int (default=1)

    Returns
    -------
    Iterable[Batch[T]]
    """
    for epoch in range(epochs):
        args = [iter(data)] * batch_size
        for index, chunk in enumerate(zip_longest(*args, fillvalue=None)):
            # make sure we remove blanks
            chunk = filter(lambda x: x is not None, chunk)
            # and yield the batch object
            yield Batch(*chunk, epoch=epoch, index=index)
