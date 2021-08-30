from ..explanation import Explanation

from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from torch import ones, tensor, Tensor

# OPERATORS

class Operator:
    def __init__(self, defaults, context_clues):
        self.defaults = defaults
        self.context_clues = context_clues

    @classmethod
    def of_json(cls, json) -> "Operator":
        """Construct an operator from a JSON-like encoding."""

        assert json["type"] == "operator"

        defaults = json["defaults"]
        context_clues = json["context-clues"]
        
        return cls(defaults, context_clues)

    def to_json(self):
        """Construct a JSON-like encoding for an operator."""

        return {
            "type" : "operator",
            "defaults" : self.defaults,
            "context-clues" : self.context_clues
        }

# ENSEMBLES

class Ensemble(ABC): pass

class LinearEnsemble(Ensemble):
    def __init__(self, weights):
        self.weights = tensor(weights, requires_grad=True)

    @classmethod
    def of_json(cls, json) -> "LinearEnsemble":
        """Construct a liner ensemble from a JSON-like encoding."""

        assert json["type"] == "ensemble"

        weights = json["weights"]
        
        return cls(weights)

    def to_json(self):
        """Construct a JSON-like encoding for an operator."""
        
        return {
            "type" : "ensemble",
            "kind" : "linear",
            "weights" : self.weights.tolist()
        }

    def parameters(self) -> Iterable[Tensor]:
        yield self.weights

# MONKEY-PATCHIN ENSEMBLE CONSTRUCTORS

@staticmethod
def ensemble_of_json(json) -> Ensemble:
    """Construct an ensemble from a JSON-like encoding."""

    assert json["type"] == "ensemble"

    if json["kind"] == "linear":
        return LinearEnsemble.of_json(json)
    else:
        raise TypeError()

Ensemble.of_json = ensemble_of_json

# POSTERIORS

class Posterior:
    def __init__(self, operator, ensemble):
        self.operator = operator
        self.ensemble = ensemble

    @classmethod
    def of_json(cls, json) -> "Posterior":
        """Construct a posterior from a JSON-like encoding."""

        assert json["type"] == "posterior"

        operator = Operator.of_json(json["operator"])
        ensemble = Ensemble.of_json(json["ensemble"])

        return cls(operator, ensemble)

    def to_json(self):
        """Construct a JSON-like encoding of the posterior."""

        return {
            "type" : "posterior",
            "operator" : self.operator.to_json(),
            "ensemble" : self.ensemble.to_json()
        }

    def parameters(self) -> Iterable[Tensor]:
        yield from self.ensemble.parameters()

    def log_prob(self, explanation : Explanation) -> Tensor:
        """Compute the posterior log-likelihood of the explanation."""

        return explanation.history.log_prob(self.ensemble.weights)

class UniformPosterior(Posterior):
    def __init__(self):
        operator = Operator(False, [])
        ensemble = LinearEnsemble([])

        super().__init__(operator, ensemble) 