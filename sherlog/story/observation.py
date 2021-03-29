from ..engine import value
from ..logs import get
from . import semantics
import torch

logger = get("story.observation")

class Observation:
    def __init__(self, mapping):
        """Observation of variable assignemnts for a story.

        Parameters
        ----------
        mapping : dict
            Each key-value pair maps a string to a value

        Returns
        -------
        Observation
        """
        self.mapping = mapping

    @property
    def size(self):
        return len(self.mapping)

    @property
    def is_empty(self):
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
    def variables(self):
        """Compute the domain of the observation.

        Returns
        -------
        Variable iterable
        """
        for k, _ in self.mapping.items():
            yield value.Variable(k)

    @property
    def values(self):
        """Compute the codomain of the observation.

        Returns
        -------
        Iterable[Value]
        """
        for _, v in self.mapping.items():
            yield v

    def evaluate(self, store, functor, wrap_args={}):
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

    def __str__(self):
        return str(self.mapping)

    def equality(self, store, functor, prefix="", default=1.0):
        # build variables
        keys = value.Variable(f"{prefix}:keys")
        vals = value.Variable(f"{prefix}:vals")
        result = value.Variable(f"{prefix}:is_equal")

        # if we don't have any observations, default
        if self.is_empty:
            functor.run(result, "set", [default], store)
        else:
            # convert to tensors and evaluate
            functor.run(keys, "tensorize", self.variables, store)
            functor.run(vals, "tensorize", self.values, store)

            functor.run(result, "equal", [keys, vals], store)

        # return the variable storing the result
        return result