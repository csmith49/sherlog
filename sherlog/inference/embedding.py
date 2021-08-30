from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, Tuple, Callable
from ..program import Evidence

T = TypeVar('T')

class Embedding(ABC, Generic[T]):
    """Abstract class providing the value-embedding interface."""

    @abstractmethod
    def to_evidence(self, value : T) -> Evidence:
        """Convert the provided value to a piece of evidence.

        Parameters
        ----------
        value : T

        Returns
        -------
        Evidence
        """
        pass

    @abstractmethod
    def to_namespace(self, value : T) -> Dict[str, Any]:
        """Conver the provided value to a namespace.

        Parameters
        ----------
        value : T

        Returns
        -------
        Dict[str, Any]
        """
        pass

    def embed(self, value : T) -> Tuple[Evidence, Dict[str, Any]]:
        return (self.to_evidence(value), self.to_namespace(value))

    def __call__(self, value : T) -> Tuple[Evidence, Dict[str, Any]]:
        return self.embed(value)

# types of embeddings

class FunctionalEmbedding(Embedding):
    """Embeds values into programs using provided callable objects."""

    def __init__(self, evidence : Callable[[T], Evidence], namespace : Callable[[T], Dict[str, Any]]):
        """Constructs a functional embedding with the provided callables.

        Parameters
        ----------
        evidence : Callable[[T], Evidence]
        namespace : Callable[[T], Dict[str, Any]]
        """
        self._evidence = evidence
        self._namespace = namespace

    def to_evidence(self, value):
        return self._evidence(value)

    def to_namespace(self, value):
        return self._namespace(value)

class UniformEmbedding(Embedding):
    """Embeds values into programs uniformly across evidence."""

    def __init__(self, evidence : Evidence, namespace : Callable[[T], Dict[str, Any]]):
        """COnstructs a functional embedding with the provided evidence and namespace generator.

        Parameters
        ----------
        evidence : Evidence
        namespace : Callable[[T], Dict[str, Any]]
        """
        self._evidence = evidence
        self._namespace = namespace

    def to_evidence(self, value):
        return self._evidence
    
    def to_namespace(self, value):
        return self._namespace(value)

class DirectEmbedding(Embedding):
    """Embeds values directly."""

    def __init__(self):
        pass

    def to_evidence(self, value):
        return value
    
    def to_namespace(self, _):
        return {}