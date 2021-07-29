from ..engine import Functor, Store, Value, Identifier, Literal
from ..logs import get

from typing import Dict, Iterable, TypeVar

logger = get("story.observation")

T = TypeVar('T')

class Observation:
    """Observations match identifiers with logical values."""

    def __init__(self, mapping : Dict[str, Value]):
        """Construct an observation from a map.

        Parameters
        ----------
        mapping : Dict[str, Value]
        """
        self.mapping = mapping

    @property
    def size(self) -> int:
        return len(self.mapping)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @classmethod
    def of_json(cls, json) -> 'Observation':
        """Build an observation from a JSON representation.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Observation
        """
        mapping = {}
        for key, value in json.items():
            mapping[key] = Value.of_json(value)
        return cls(mapping)

    @property
    def identifiers(self) -> Iterable[Identifier]:
        """Compute the domain of the observation.

        Returns
        -------
        Iterable[Identifier]
        """
        for key, _ in self.mapping.items():
            yield Identifier(key)

    @property
    def values(self) -> Iterable[Value]:
        """Compute the codomain of the observation.

        Returns
        -------
        Iterable[Value]
        """
        for _, value in self.mapping.items():
            yield value

    def evaluate(self, functor : Functor[T], store : Store, **kwargs) -> Iterable[T]:
        """Evaluate the observation.

        Parameters
        ----------
        functor : Functor[T]

        store : Store

        **kwargs
            Passed to functor during evaluation.

        Returns
        -------
        Iterable[T]
        """
        for _, value in self.mapping.items():
            yield functor.evaluate(value, store, **kwargs)

    def target(self, functor : Functor, store : Store, prefix : str = "sherlog", default : float = 1.0) -> Identifier:
        """Compute the optimization target of the expected and observed values.

        Returns the identifier pointing to the target in the store.
        
        Parameters
        ----------
        functor : Functor
        
        store : Store
            Modified in-place.
        
        prefix : str (default='sherlog')
        
        default : float (default=1.0)
        
        Returns
        -------
        Identifier
        """
        # build the identifiers
        keys = Identifier(f"{prefix}:keys")
        values = Identifier(f"{prefix}:values")
        target = Identifier(f"{prefix}:target")

        # if the observation is somehow empty, default
        if self.is_empty:
            functor.run(target, "set", [Literal(default)], store)
        # otherwise tensorize and compare
        else:
            functor.run(keys, "tensorize", self.identifiers, store)
            functor.run(values, "tensorize", self.values, store)
            functor.run(target, "target", [keys, values], store)
        
        # and return the identifier
        return target

    # MAGIC METHODS ------------------------------------------------------

    def __getitem__(self, key : Identifier) -> Value:
        if isinstance(key, Identifier):
            return self.mapping[key.name]
        else:
            raise TypeError(key)

    def __setitem__(self, key : Identifier, value : Value):
        if isinstance(key, Identifier):
            self.mapping[key.name] = value
        else:
            raise TypeError(key)

    def __contains__(self, key : Identifier):
        if isinstance(key, Identifier):
            return key.name in self.mapping.keys()
        else:
            raise TypeError(key)

    def __str__(self):
        return str(self.mapping)
