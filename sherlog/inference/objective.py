from typing import TypeVar, List
from .embedding import Embedding

from ..program import Program, Evidence

import torch
from torch import Tensor
from ..logs import get

logger = get("inference.objective")

T = TypeVar('T')

class Objective:
    """Wraps a value to be optimized."""
    def __init__(self, name : str, value : Tensor):
        self.name = name
        self.value = value

        # make sure the value is actually differentiable
        if self.value.grad_fn is None:
            logger.warning(f"Objective {self} has no gradient.")

    def is_nan(self) -> bool:
        return torch.isnan(self.value).any()
    
    def is_infinite(self) -> bool:
        return torch.isinf(self.value).any()

    def __str__(self) -> str:
        return f"Objective[{self.name}, {self.value:f}]"

class BatchObjective(Objective):
    """For building objectives from a batch of evidence."""
    def __init__(self, name : str, program : Program, batch : List[Evidence], log_prob_kwargs={}):
        # we must explicitly construct the value
        marginals = [program.log_prob(evidence, **log_prob_kwargs) for evidence in batch]
        log_prob = torch.stack(marginals).sum()
        # and then just build like normal
        super().__init__(name, log_prob)

class BatchEmbeddingObjective(Objective):
    """For building objectives from a batch of values to be embedded."""
    def __init__(self, name : str, program : Program, embedding : Embedding[T], batch : List[T], log_prob_kwargs={}):
        # we must explicitly construct the value
        marginals = []
        for value in batch:
            evidence, namespace = embedding(value)
            marginal = program.log_prob(evidence, namespace=namespace, **log_prob_kwargs)
            marginals.append(marginal)
        log_prob = torch.stack(marginals).sum()
        # and then just build like normal
        super().__init__(name, log_prob)