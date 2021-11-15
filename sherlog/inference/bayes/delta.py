from torch import Tensor, tensor
from typing import Iterable

from .point import Point

class Delta:
    def __init__(self, initial : Tensor):
        self._initial = initial
        self._values = {}

    def fresh_tensor(self) -> Tensor:
        return self._initial.clone().detach().requires_grad_(True)

    def parameter(self, point: Point):
        try:
            result = self._values[point.key]
        except KeyError:
            result = self.fresh_tensor()

        self._values[point.key] = result
        return result

    # MAGIC METHODS

    def __getitem__(self, point : Point):
        return self.parameter(point)

    # CONVERSION OF POINTS

    def parameters(self, points : Iterable[Point]) -> Iterable[Tensor]:
        for point in points:
            yield self[point]