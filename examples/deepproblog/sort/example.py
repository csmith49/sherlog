from random import randint
from typing import Iterable
from itertools import permutations

class Example:
    def __init__(self, size : int, min : int, max : int):
        """A training example for the NPI task.

        Parameters
        ----------
        size : int
        min : int
        max : int
        """
        self._elts = [randint(min, max) for _ in range(size)]

    @property
    def input(self):
        return self._elts

    @property
    def output(self):
        return sorted(self._elts)

    def output_permutations(self):
        yield from permutations(self.output)

def sample(quantity : int, size : int = 5, min : int = 0, max : int = 9):
    for _ in range(quantity):
        yield Example(size, min, max)