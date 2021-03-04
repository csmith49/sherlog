from ..engine import value, evaluate
from . import scg
import torch

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

    def evaluate(self, store, algebra):
        """Evaluate the observation.

        Parameters
        ----------
        store : Store

        algebra : Algebra

        Returns
        -------
        value
        """
        for _, v in self.mapping.items():
            yield evaluate(v, store, algebra)

    def __str__(self):
        return str(self.mapping)

    def similarity(self, store, epsilon=1.0, default=1.0):
        """Computes cosine similarity between a given observation and a store.
        
        Parameters
        ----------
        store : Store

        epsilon : float option

        default : float option
        
        Returns
        -------
        tensor
        """

        # undefined for empty vectors, so use the relevant provided value
        if self.size == 0: return torch.tensor(default)

        # evaluate the obs and store to get tensors
        obs_vec = self.evaluate(store, scg.algebra)
        str_vec = [store[v] for v in self.variables]

        # cant stack in storch (yet), so manually compute cosine similarity
        # epsilon ensures we don't divide by 0
        # equivalent to extending each vector with 1 extra index w/ value epsilon
        dot_prod = torch.tensor(epsilon)
        mag_a, mag_b = torch.tensor(epsilon ** 2), torch.tensor(epsilon ** 2)

        for a, b in zip(obs_vec, str_vec):
            dot_prod += a * b
            mag_a += torch.pow(a, 2)
            mag_b += torch.pow(b, 2)

        return dot_prod / torch.sqrt(mag_a * mag_b)
