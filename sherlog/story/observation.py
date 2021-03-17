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

    def evaluate(self, store, functor, wrap_args={}):
        """Evaluate the observation.

        Parameters
        ----------
        store : Store

        functor : Functor

        Returns
        -------
        value
        """
        for _, v in self.mapping.items():
            yield functor.evaluate(v, store, wrap_args=wrap_args)

    def __str__(self):
        return str(self.mapping)

    def equality(self, store, epsilon=0.0005, **kwargs):
        """Returns 1 if the store equals the observation, 0 otherwise.

        Parameters
        ----------
        store : Store

        epsilon : float option

        Returns
        -------
        tensor
        """

        # if there's no vector, we're equal by convention
        if self.size == 0:
            return torch.tensor(1.0)
        
        # otherwise build the vecs
        obs_vec = torch.tensor(list(self.evaluate(store, semantics.tensor)))
        str_vec = torch.tensor([store[v] for v in self.variables])

        # check if distance is sufficiently small
        if torch.dist(obs_vec, str_vec) < epsilon:
            return torch.tensor(1.0)
        else:
            return torch.tensor(0.0)

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

        obs_vec = torch.tensor(list(self.evaluate(store, semantics.tensor)))
        str_vec = torch.tensor([store[v] for v in self.variables])

        distance = torch.dist(obs_vec, str_vec)

        # pass through gaussian rbf
        result = torch.exp(- torch.pow(distance, 2) / (2 * torch.tensor(temperature)))
        
        logger.info(f"Similarity between {self} and {store}: {result}")

        return result
