from ..engine import value, evaluate
from ..logs import get
from . import scg
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

    def similarity(self, store, default=1.0, temperature=0.001):
        """Computes similarity between a given observation and a store.
        
        Parameters
        ----------
        store : Store

        default : float option
        
        Returns
        -------
        tensor
        """

        # undefined for empty vectors, so use the relevant provided value
        if self.size == 0: 
            logger.info(f"Similarity computation with empty observation. Returning default {default}.")
            return torch.tensor(default)

        # evaluate the obs and store to get tensors
        obs_vec = torch.tensor(list(self.evaluate(store, scg.algebra)))
        str_vec = torch.stack([store[v] for v in self.variables])

        result = torch.exp(
            -torch.pow(torch.dist(obs_vec, str_vec), 2) / torch.tensor(temperature)
        )
        
        logger.info(f"Similarity between {self} and {store}: {result}")

        return result
