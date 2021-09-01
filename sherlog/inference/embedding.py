from typing import TypeVar, Generic, Iterable, Mapping, Callable
from abc import ABC, abstractmethod
from torch import Tensor

from .objective import Objective
from ..program import Evidence

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

class DirectEmbedding(Embedding[Evidence]):
    """Embedding that operates naively over evidence."""

    def embed(self, datum : Evidence, **kwargs) -> Objective:
        """Embed a piece of evidence into an objective."""

        return Objective(evidence=datum)

class UniformEmbedding(Embedding[T]):
    """Embedding that uniformly maps T to a piece of evidence."""

    def __init__(self, evidence : Evidence, embedding_callable : Callable[[T], Mapping[str, Tensor]]):
        self._evidence = evidence
        self._embedding_callable = embedding_callable

    def embed(self, datum : T, **kwargs) -> Objective:
        return Objective(
            evidence=self.evidence,
            parameters=self._embedding_callable(datum)
        )
