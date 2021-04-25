from .objective import Objective
from ..program import Program, Evidence
from typing import Iterable, Optional, Any, TypeVar, Generic, Mapping, Callable
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

T = TypeVar('T')

class NamespaceBatch(Batch, Generic[T]):
    def __init__(self, data : Iterable[T], to_evidence : Callable[[T], Evidence], to_namespace : Callable[[T], Mapping[str, Any]], index : Optional[int] = None):
        super(NamespaceBatch, self).__init__(data, index)
        self._to_evidence = to_evidence
        self._to_namespace = to_namespace

    def marginals(self, program : Program, **kwargs):
        for evidence in self.evidence:
            yield program.likelihood(self._to_evidence(evidence), namespace=self._to_namespace(evidence), **kwargs)

    def objective(self, program : Program, **kwargs) -> Objective:
        log_likelihood = torch.stack(list(self.marginals(program, **kwargs))).log().sum()
        return Objective(self.name, log_likelihood)