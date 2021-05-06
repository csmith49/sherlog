from .objective import Objective
from .optimizer import Optimizer
from .batch import Batch, NamespaceBatch
from typing import Iterable, TypeVar, List
from ..program import Evidence

from random import shuffle
from itertools import cycle, islice, chain, repeat, zip_longest

def minibatch(evidence : List[Evidence], batch_size : int):
    shuffle(evidence)
    args = [iter(evidence)] * batch_size
    for batch in zip_longest(*args, fillvalue=None):
        yield filter(lambda x: x is not None, batch)

T = TypeVar('T')

def namespace_minibatch(data : Iterable[T], batch_size : int, to_evidence, to_namespace, epochs : int = 1):
    data = cycle(data)
    for epoch in range(epochs):
        yield NamespaceBatch(islice(data, batch_size), to_evidence=to_evidence, to_namespace=to_namespace, index=epoch)