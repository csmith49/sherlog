from typing import Optional, Iterable
from random import randint

from .data import Graph
from .sherlog import SherlogModel

def sample(quantity : int, size : Optional[int] = None) -> Iterable[Graph]:
    for _ in range(quantity):
        if size:
            yield Graph(size)
        else:
            yield Graph(randint(2, 10))
            