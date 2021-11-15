from ..program import Evidence
from ..interface import parse_source

from .bayes import Point
from .objective import Objective

from typing import Generic, TypeVar, Callable, Iterable, Optional, Mapping

T = TypeVar('T')

# UTILITY

def evidence_of_string(*evidence : str):
    _, evidence = parse_source(f"!evidence {', '.join(evidence)}.")
    return Evidence.of_json(evidence[0])

# EMBEDDING

class Embedding(Generic[T]):
    def __init__(self, evidence : Callable[[T], Iterable[str]], points : Optional[Callable[[T], Iterable[Point]]] = None):
        self._evidence = evidence
        self._points = points

    def embed(self, datum : T, **kwargs) -> Objective:
        evidence = evidence_of_string(*self._evidence(datum))
        points = self._points(datum) if self._points else ()

        return Objective(evidence=evidence, points=points)

    def embed_all(self, data : Iterable[T], **kwargs) -> Iterable[Objective]:
        for datum in data:
            yield self.embed(datum, **kwargs)

    # MAGIC METHODS

    def __call__(self, datum : T, **kwargs) -> Objective:
        return self.embed(datum, **kwargs)

# SPECIAL EMBEDDINGS

class PartitionEmbedding(Embedding[T]):
    def __init__(self, evidence : Mapping[T, str], points : Optional[Mapping[T, Iterable[Point]]] = None):
        super().__init__(lambda x : [evidence[x]], points)

class StringEmbedding(Embedding[str]):
    def __init__(self, points : Optional[Callable[[str], Iterable[Point]]] = None):
        super().__init__(lambda x : [evidence_of_string(x)], points)

class DirectEmbedding(Embedding[Evidence]):
    def __init__(self, points : Optional[Callable[[Evidence], Iterable[Point]]] = None):
        super().__init__(lambda x: [x], points)