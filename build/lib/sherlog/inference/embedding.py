from typing import TypeVar, Generic, Iterable, Mapping, Callable, Optional
from abc import ABC, abstractmethod
from torch import Tensor

from .objective import Objective
from ..program import Evidence
from ..interface import parse_source

T = TypeVar('T')

class Embedding(ABC, Generic[T]):
    """doc string goes here"""
    
    @abstractmethod
    def embed(self, datum : T, **kwargs) -> Objective:
        """doc string goes here"""
        
        raise NotImplementedError()

    def embed_all(self, data : Iterable[T], **kwargs) -> Iterable[Objective]:
        """Iterates over the embedded data."""

        for datum in data:
            yield self.embed(datum, **kwargs)

    def __call__(self, datum : T, **kwargs) -> Objective:
        """Dunder method wrapping `self.embed`. Embeds a datum."""

        return self.embed(datum, **kwargs)

# Specialty embeddings

# utility for converting strings to evidence objects
def parse_evidence(evidence : str):
    _, evidence = parse_source(f"!evidence {evidence}.")
    return Evidence.of_json(evidence[0])

class DirectEmbedding(Embedding[Evidence]):
    """Embedding that operates naively over evidence."""

    def embed(self, datum : Evidence, **kwargs) -> Objective:
        """Embed a piece of evidence into an objective."""

        return Objective(evidence=datum)

class UniformEmbedding(Embedding[T]):
    """Embedding that uniformly maps T to an objective.
    
    Each datum is distinguished solely by the output of a callable that produces a parameterization.
    """

    def __init__(self, evidence : str, conditional : str, callable : Callable[[T], Mapping[str, Tensor]]):
        self._evidence = parse_evidence(evidence)
        self._conditional = parse_evidence(conditional) if conditional else None
        self._callable = callable

    def embed(self, datum : T, **kwargs) -> Objective:
        return Objective(
            evidence=self.evidence,
            conditional=self._conditional,
            parameters=self._callable(datum)
        )

class PartitionEmbedding(Embedding[T]):
    """Embedding that maps datum to a finite set of evidence."""

    def __init__(self, partition : Mapping[T, str]):
        self._partition = {key : parse_evidence(value) for key, value in partition.items()}
        
    def embed(self, datum : T, **kwargs) -> Objective:
        return Objective(
            evidence=self._partition[datum]
        )

class FunctionalEmbedding(Embedding[T]):
    """Embedding that relies on external callables to convert data to objectives.
    
    Each callable acts independently.
    """

    def __init__(self,
        evidence : Callable[[T], str],
        conditional : Optional[Callable[[T], str]] = None,
        parameters : Optional[Callable[[T], Mapping[str, Tensor]]] = None
    ):
        self._evidence = evidence
        self._conditional = conditional
        self._parameters = parameters

    def embed(self, datum : T, **kwargs) -> Objective:
        return Objective(
            evidence=parse_evidence(self._evidence(datum)),
            conditional=parse_evidence(self._conditional(datum)) if self._conditional else None,
            parameters=self._parameters(datum) if self._parameters else None
        )