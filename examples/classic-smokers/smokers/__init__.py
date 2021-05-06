from random import randrange
from typing import Optional
from itertools import product

from .sherlog import SherlogModel
from .problog import ProblogModel
from .data import Graph, Parameterization

default_parameterization = Parameterization(
    stress=0.2,
    spontaneous=0.1,
    comorbid=0.3,
    influence=0.3
)

def dict_product(**kwargs):
    keys, values = kwargs.keys(), kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))

def sample(quantity : int, size : Optional[int] = None):
    for _ in range(quantity):
        if size is not None:
            yield Graph(size, default_parameterization)
        else:
            # sample a random size - close social circles usually b/t 5 and 15 people
            yield Graph(randrange(5, 15), default_parameterization)
