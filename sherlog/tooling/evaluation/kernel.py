from typing import TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from math import exp

T = TypeVar('T')

class Kernel(ABC, Generic[T]):
    @abstractmethod
    def eval(self, x : T, y : T) -> float: pass

    def __call__(self, x : T, y : T) -> float:
        return self.eval(x, y)

class RBFKernel(Kernel):
    def __init__(self, metric : Callable[[T, T], float], sigma : float = 1.0):
        """Constructs a radial-basis function kernel around a metric.

        Parameters
        ----------
        metric : Callable[[T, T], float]
        sigma : float (default=1.0)
        """
        self.metric = metric
        self.sigma = sigma

    def eval(self, x : T, y : T) -> float:
        d = self.metric(x, y)
        numer = - (d ** 2)
        denom = 2 * (self.sigma ** 2)
        return exp(numer / denom)