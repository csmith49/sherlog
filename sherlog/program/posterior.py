from ..explanation import Explanation

from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from torch import ones, tensor, Tensor

class Ensemble(ABC): pass

class LinearEnsemble(Ensemble):
    def __init__(self, weights):
        self.weights = tensor(weights, requires_grad=True)

    def parameters(self) -> Iterable[Tensor]:
        yield self.weights

    @classmethod
    def of_json(cls, json) -> "LinearEnsemble":
        """"Construct a Linear Ensemble from a JSON-like encoding."""

        assert json["kind"] == "linear"

        weights = json["weights"]
        return cls(weights)

    def dump(self):
        return {
            "type" : "ensemble",
            "kind" : "linear",
            "weights" : self.weights.tolist()
        }

class Feature:
    def __init__(self, json):
        self.json = json

class Posterior:
    def __init__(self, features : List[Feature], ensemble : Ensemble):
        """Construct a posterior."""

        self.features = features
        self.ensemble = ensemble

    @classmethod
    def of_json(cls, json):
        assert json["type"] == "posterior"

        features = json["features"]
        ensemble = LinearEnsemble.of_json(json["ensemble"])
        return cls(features, ensemble)

    def dump(self):
        return {
            "type" : "posterior",
            "features" : self.features,
            "ensemble" : self.ensemble.dump()
        }

    def parameters(self) -> Iterable[Tensor]:
        yield from self.ensemble.parameters()