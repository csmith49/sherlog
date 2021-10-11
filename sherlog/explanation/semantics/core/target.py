"""Contains abstract base classes and concrete implementations of *targets*: functions that convert pairs of observation tensors to an optimization objective."""

from abc import ABC, abstractmethod
from torch import Tensor, tensor, equal, dist

class Target(ABC):
    """Targets convert pairs of tensors to an optimization objective."""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @abstractmethod
    def evaluate(self, left : Tensor, right : Tensor, *args, **kwargs) -> Tensor:
        """Convert the pair `(left, right)` to an optimization objective.

        Parameters
        ----------
        left : Tensor
        right : Tensor

        Returns
        -------
        Tensor
        """
        pass

    def __call__(self, left : Tensor, right : Tensor) -> Tensor:
        """Defaults to `self.evaluate`
        
        Provides `*args` and `**kwargs` given during instance construction.

        Parameters
        ----------
        left : Tensor
        right : Tensor

        Returns
        -------
        Tensor
        """
        return self.evaluate(left, right, *self._args, **self._kwargs)

# concrete implementations
class RBF(Target):
    """Gaussian radial basis function target.
    
    Pass `sdev=...` to set the standard deviation.
    """
    def evaluate(self, left : Tensor, right : Tensor, *args, sdev : float = 1.0, **kwargs) -> Tensor:
        numerator = dist(left.float(), right.float(), p=2).pow(2)
        denominator = 2 * (sdev ** 2)
        return (-1 * (numerator / denominator)).exp()

class MSE(Target):
    """Mean squared error target."""
    def evaluate(self, left : Tensor, right : Tensor, *args, **kwargs) -> Tensor:
        return (left.float() - right.float()).pow(2).mean()

class EqualityIndicator(Target):
    """Equality indicator target.
    
    0/1-valued, and so not differentiable.
    """
    def evaluate(self, left : Tensor, right : Tensor, *args, epsilon : float = 1e-10, **kwargs) -> Tensor:
        return tensor(1.0) if dist(left, right) <= epsilon else tensor(0.0)