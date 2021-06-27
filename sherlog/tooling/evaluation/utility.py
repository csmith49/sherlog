"""Utilities for constructing and accessing data sources go here."""

from typing import Iterable, List, TypeVar
from itertools import zip_longest
from random import shuffle

T = TypeVar('T')

def minibatch(data : List[T], batch_size : int) -> Iterable[List[T]]:
    """Convert a list of data into batches of the same size (except for the last batch, which might be smaller).

    Parameters
    ----------
    data : List[T]
    batch_size : int

    Returns
    -------
    Iterable[List[T]]
    """
    # randomize in-place
    shuffle(data)
    # build the iterator
    args = [iter(data)] * batch_size
    for batch in zip_longest(*args, fillvalue=None):
        # make sure we remove blanks and convert to list
        yield list(filter(lambda x: x is not None, batch))