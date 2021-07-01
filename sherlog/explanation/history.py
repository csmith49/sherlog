from sherlog.program import parameter
from torch import tensor, stack, Tensor
from typing import List, Iterable

import torch

class Record:
    """A record captures the features and context of an explanation during a single point of the stochastic search process."""

    def __init__(self, features : Tensor, context : List[Tensor]):
        """Construct a record directly from tensors.
        
        Parameters
        ----------
        features : Tensor

        context : List[Tensor]        
        """
        self.features = features
        self.context = context

    @classmethod
    def of_json(cls, json) -> 'Record':
        """Construct a record from a JSON-like record.
        
        Parameters
        ----------
        json : JSON-like object
        
        Returns
        -------
        Record
        """
        features = torch.tensor(json["features"])
        context = [torch.tensor(c) for c in json["context"]]
        return cls(features, context)

    def score(self, parameterization : Tensor) -> Tensor:
        """Score the record by combining the features.

        Parameters
        ----------
        parameterization : Tensor
            Linear parameterization of the features.

        Returns
        -------
        Tensor
        """
        return self.features.dot(parameterization)

    def normalization_constant(self, parameterization : Tensor) -> Tensor:
        """Compute the normalization constant by which the record's score is convertible into a likelihood.

        Parameters
        ----------
        parameterization : Tensor
            Linear parameterization of the features.

        Returns
        -------
        Tensor
        """
        return stack([context.dot(parameterization) for context in self.context]).sum()

    def log_prob(self, parameterization : Tensor) -> Tensor:
        """The log-likelihood the record was chosen from the context.

        Parameters
        ----------
        parameterization : Tensor
            Linear parameterization of the features.

        Returns
        -------
        Tensor
        """
        return self.score(parameterization).log() - self.normalization_constant(parameterization).log()

    # MAGIC METHODS

    def __str__(self) -> str:
        return str(self._context)

    def __repr__(self) -> str:
        return str(self)

class History:
    """Histories are sequences of records explaining the derivation of an explanation."""
    
    def __init__(self, records : Iterable[Record]):
        """Construct a history from a list of records.
        
        Parameters
        ----------
        records : Iterable[Record]

        """
        self._records = list(records)
    
    @classmethod
    def of_json(cls, json) -> 'History':
        """Construct a history from a JSON-like object.
        
        Parameters
        ----------
        json : JSON-like object
        
        Returns
        -------
        History
        """
        records = [Record.of_json(r) for r in json["records"]]
        return cls(records)

    def log_prob(self, parameterization : Tensor) -> Tensor:
        """Log-likelihood of the history being derived.
        
        Parameters
        ----------
        parameterization : Tensor
            Linear parameterization of the features.
            
        Returns
        -------
        Tensor
        """
        if self.records:
            return stack([record.log_prob(parameterization) for record in self.records]).sum()
        else:
            return tensor(0.0)

    def join(self, other : 'History') -> 'History':
        """Append a history to the end of this history.
        
        Parameters
        ----------
        other : History
        
        Returns
        -------
        History
        """
        records = self.records + other.records
        return History(records)

    # MAGIC METHODS

    def __add__(self, other):
        return self.join(other)
    
    def __str__(self):
        return str(self._records)
    
    def __repr__(self):
        return str(self)