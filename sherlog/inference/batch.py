from .objective import Objective
from ..program import Program, Evidence
from typing import Iterable, Optional
import torch

class Batch:
    def __init__(self, evidence : Iterable[Evidence], index : Optional[int] = None):
        self.evidence = list(evidence)
        self.index = index

    @property
    def size(self) -> int: return len(self.evidence)

    @property
    def name(self) -> str:
        if self.index:
            return f"batch:{self.index}"
        else:
            return "batch"

    def objective(self, program : Program, **kwargs) -> Objective:
        # compute total log-prob of the provided evidence
        marginals = torch.stack([program.likelihood(evidence, **kwargs) for evidence in self.evidence])
        log_likelihood = marginals.log().sum()
        
        return Objective(self.name, log_likelihood)