from typing import Dict, Iterable, Any

from ..engine import Functor, Store, Value, Variable, value
from ..engine.value import Variable
from ..logs import get

logger = get("story.observation")

class Observation:
    def __init__(self, mapping : Dict[str, Value]):
        """Observation of variable assignemnts for a story.

        Parameters
        ----------
        mapping : Dict[str, Value]
            Each key-value pair maps a string to a value

        Returns
        -------
        Observation
        """
        self.mapping = mapping

    @property
    def size(self) -> int:
        return len(self.mapping)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @classmethod
    def of_json(cls, json):
        """Build an observation from a JSON representation.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Observation
        """
        mapping = {}
        for k, v in json.items():
            mapping[k] = value.of_json(v)
        return cls(mapping)

    @property
    def variables(self) -> Iterable[Variable]:
        """Compute the domain of the observation.

        Returns
        -------
        Variable iterable
        """
        for k, _ in self.mapping.items():
            yield Variable(k)

    @property
    def values(self) -> Iterable[Value]:
        """Compute the codomain of the observation.

        Returns
        -------
        Iterable[Value]
        """
        for _, v in self.mapping.items():
            yield v

    def evaluate(self, store : Store, functor : Functor, wrap_args={}) -> Iterable[Any]:
        """Evaluate the observation.

        Parameters
        ----------
        store : Store

        functor : Functor

        Returns
        -------
        Functor.t
        """
        for _, v in self.mapping.items():
            yield functor.evaluate(v, store, wrap_args=wrap_args)

    def target(self, store : Store, functor : Functor, prefix : str = "", default : float = 1.0) -> Variable:
        """Compute the optimization target of the expected and observed values.

        Parameters
        ----------
        store : Store

        functor : Functor

        prefix : str (default="")

        default : float (default=1.0)

        Returns
        -------
        Variable
        """
        # build variables
        keys = Variable(f"{prefix}:keys")
        vals = Variable(f"{prefix}:vals")
        result = Variable(f"{prefix}:is_equal")

        # if we don't have any observations, default
        if self.is_empty:
            functor.run(result, "set", [default], store)
        else:
            # convert to tensors and evaluate
            functor.run(keys, "tensorize", self.variables, store)
            functor.run(vals, "tensorize", self.values, store)

            functor.run(result, "target", [keys, vals], store)

        # return the variable storing the result
        return result

    def join(self, other):
        mapping = {}
        for k, v in self.mapping.items():
            mapping[k] = v
        for k, v in other.mapping.items():
            mapping[k] = v
        return Observation(mapping)

    # MAGIC METHODS ------------------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.mapping[key]
        elif isinstance(key, Variable):
            return self.mapping[key.name]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(key, Variable):
            self.mapping[key.name] = value
        else:
            raise KeyError() # this isn't quite semantic, is it?

    def __contains__(self, key):
        if isinstance(key, Variable):
            return key.name in self.mapping.keys()
        else:
            return False

    def __str__(self):
        return str(self.mapping)
    
    def __add__(self, other):
        return self.join(other)