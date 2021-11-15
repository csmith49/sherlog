from torch import Tensor, tensor
from collections import defaultdict
from typing import Iterable

from .point import Point

class Delta:
    def __init__(self, initial : Tensor):
        self.initial = initial

        self.values = defaultdict(lambda: tensor(self.initial, requires_grad=True))

    # MAGIC METHODS

    def __getitem__(self, point : Point):
        try:
            return self.values[point.key]

        except KeyError:
            result = tensor(self.initial, requires_grad=True)
            self.values[point.key] = result

            return result

    # CONVERSION OF POINTS

    def parameters(self, points : Iterable[Point]) -> Iterable[Tensor]:
        for point in points:
            yield self[point]