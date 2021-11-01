from abc import ABC, abstractmethod
from torch import Tensor, tensor, dist, stack
from typing import Callable, List

# ABSTRACT BASE CLASS

class Semiring(ABC):
    """Semirings provide semanitcs for a sum-product expression."""

    def lookup(self, function_id : str) -> Callable[..., Tensor]:
        if function_id == "semiring:sum":
            return self.sum
        elif function_id == "semiring:product":
            return self.product
        elif function_id == "semiring:one":
            return self.one
        elif function_id == "semiring:zero":
            return self.zero
        else:
            raise KeyError(f"Function not defined in semiring. [function_id={function_id}]")

    def supported(self) -> List[str]:
        return [
            "semiring:sum",
            "semiring:product",
            "semiring:one",
            "semiring:zero"
        ]

    # ABSTRACT METHODS
    # Should be implemented in all subclasses. Note the types on `zero` and `one`; they
    # correspond to evaluating `NotEqual` and `Equal` literals, respectively.

    @abstractmethod
    def sum(self, *arguments : Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def product(self, *arguments : Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def one(self, left : Tensor, right : Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def zero(self, left : Tensor, right : Tensor) -> Tensor:
        raise NotImplementedError

# EVALUATION SEMIRINGS

class PreciseSemiring(Semiring):
    """Precisely evaluates a sum-product expression using (max, min, 0, 1) to emulate (or, and, false, true)"""

    def __init__(self, epsilon : float = 1e-7):
        self.epsilon = epsilon
        super().__init__()

    def sum(self, *arguments : Tensor) -> Tensor:
        return stack(arguments).max()

    def product(self, *arguments : Tensor) -> Tensor:
        return stack(arguments).min()

    def zero(self, left : Tensor, right : Tensor) -> Tensor:
        if dist(left.float(), right.float()) > self.epsilon:
            return tensor(1.0)
        else:
            return tensor(0.0)

    def one(self, left : Tensor, right : Tensor) -> Tensor:
        if dist(left.float(), right.float()) <= self.epsilon:
            return tensor(1.0)
        else:
            return tensor(0.0)

class DisjointSumSemiring(Semiring):
    """Avoids the use of min and max, but requires all sums be disjoint."""

    def __init__(self, epsilon : float = 1e-7):
        self.epsilon = epsilon
        super().__init__()

    def sum(self, *arguments : Tensor) -> Tensor:
        return stack(arguments).sum()

    def product(self, *arguments : Tensor) -> Tensor:
        return stack(arguments).mult()

    def zero(self, left : Tensor, right : Tensor) -> Tensor:
        if dist(left.float(), right.float()) > self.epsilon:
            return tensor(1.0)
        else:
            return tensor(0.0)

    def one(self, left : Tensor, right : Tensor) -> Tensor:
        if dist(left.float(), right.float()) <= self.epsilon:
            return tensor(1.0)
        else:
            return tensor(0.0)
