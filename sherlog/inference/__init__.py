from .objective import Objective
from .optimizer import Optimizer
from .batch import Batch
from typing import Iterable
from ..program import Evidence

from itertools import cycle, islice

def minibatch(evidence : Iterable[Evidence], batch_size : int, epochs : int = 1):
    data = cycle(evidence)
    for epoch in range(epochs):
        yield Batch(islice(data, batch_size), index=epoch)