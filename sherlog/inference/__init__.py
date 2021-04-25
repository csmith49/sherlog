from .objective import Objective
from .optimizer import Optimizer
from .batch import Batch, NamespaceBatch
from typing import Iterable, TypeVar
from ..program import Evidence

from itertools import cycle, islice

def minibatch(evidence : Iterable[Evidence], batch_size : int, epochs : int = 1):
    data = cycle(evidence)
    for epoch in range(epochs):
        yield Batch(islice(data, batch_size), index=epoch)

T = TypeVar('T')

def namespace_minibatch(data : Iterable[T], batch_size : int, to_evidence, to_namespace, epochs : int = 1):
    data = cycle(data)
    for epoch in range(epochs):
        yield NamespaceBatch(islice(data, batch_size), to_evidence=to_evidence, to_namespace=to_namespace, index=epoch)