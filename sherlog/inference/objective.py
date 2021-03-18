import torch
from ..logs import get

logger = get("inference.objective")

class Objective:
    def __init__(self, name, value):
        self.name = name
        self.value = value

        # make sure the value is actually differentiable
        if self.value.grad_fn is None:
            logger.warning(f"Objective {self} has no gradient.")

    def is_nan(self):
        return torch.isnan(self.value).any()
    
    def is_infinite(self):
        return torch.isinf(self.value).any()

    def __str__(self):
        return f"Objective[{self.name}, {self.value:f}]"