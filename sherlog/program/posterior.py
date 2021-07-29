from ..explanation import Explanation

from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from torch import ones, tensor, Tensor

# ABC for posteriors
class Posterior(ABC):
    """Posteriors provide likelihoods to explanations."""
    
    @abstractmethod
    def log_prob(self, explanation : Explanation) -> Tensor:
        pass

    @abstractmethod
    def parameters(self) -> Iterable[Tensor]:
        pass

    @abstractmethod
    def parameterization(self) -> List[float]:
        pass

# Concrete implementations
class LinearPosterior:
    """Posterior that linearly combines features to compute a score."""

    def __init__(self, contexts : List[str] = [], weights : Optional[List[float]] = None):
        """Construct a linear posterior.
        
        Parameters
        ----------
        contexts : List[str] (default=[])

        weights : Optional[List[float]]
            If not provided, instantiated uniformly.
        """
        self.contexts = contexts
        if weights is None:
            self.weights = ones(len(self.contexts) + 3, requires_grad=True)
        else:
            self.weights = tensor(weights, requires_grad=True)

    def log_prob(self, explanation : Explanation) -> Tensor:
        """Compute the log-likelihood of the explanation in the posterior.
        
        Parameters
        ----------
        explanation : Explanation

        Returns
        -------
        Tensor
        """
        return explanation.history.log_prob(self.weights)

    def parameters(self) -> Iterable[Tensor]:
        """Iterates over all tuneable parameters of the posterior.
        
        Returns
        -------
        Iterable[Tensor]
        """
        yield self.weights

    def parameterization(self) -> List[float]:
        """Linearizes the parameters for serialization purposes.
        
        Returns
        -------
        List[float]
        """
        return self.weights.tolist()