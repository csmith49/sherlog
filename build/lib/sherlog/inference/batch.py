"""Utilities for constructing and accessing data sources go here."""

from dataclasses import dataclass
from typing import List, Iterable, TypeVar, Generic
from itertools import zip_longest

from torch import Tensor, stack

from ..program import Program
from .objective import Objective
from .embedding import Embedding

T = TypeVar('T')

@dataclass
class Batch(Generic[T]):
    data : List[T]
    epoch : int
    index : int

    def __iter__(self):
        return self.data

    @property
    def identifier(self):
        return f"Batch:{self.index}:{self.epoch}"

def minibatch(data : List[T], batch_size : int, epochs : int = 1) -> Iterable[Batch[T]]:
    """Convert a list of data into batches of a given size."""

    for epoch in range(epochs):
        args = [iter(data)] * batch_size
        for index, chunk in enumerate(zip_longest(*args, fillvalue=None)):
            # remove blanks from the chunk
            chunk = filter(lambda x: x is not None, chunk)
            yield Batch(list(chunk), epoch=epoch, index=index)
