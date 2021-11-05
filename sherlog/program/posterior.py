from ..explanation import Explanation

from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from torch import ones, tensor, Tensor

class Operation:
    def __init__(self, json):
        self.source = json

    @classmethod
    def of_json(cls, json) -> "Operation":
        assert json["type"] == "operation"

        return cls(json)

    def to_json(self):
        return self.source

class Feature:
    def __init__(self, weight, operation : Operation):
        self.weight = weight
        self.operation = operation

    @classmethod
    def of_json(cls, json) -> "Feature":
        assert json["type"] == "feature"

        weight = json["weight"]
        operation = Operation.of_json(json["operation"])

        return cls(weight, operation)

    def to_json(self):
        return {
            "type" : "feature",
            "weight" : self.weight,
            "operation" : self.operation.to_json()
        }

    def parameters(self) -> Iterable[Tensor]:
        yield self.weight

class Posterior:
    def __init__(self, features : List[Feature]):
        self.features = features

    @classmethod
    def of_json(cls, json) -> "Posterior":
        assert json["type"] == "posterior"

        features = [Feature.of_json(feature) for feature in json["features"]]

        return cls(features)

    def to_json(self):
        return {
            "type" : "posterior",
            "features" : [feature.to_json() for feature in self.features]
        }

    def parameters(self) -> Iterable[Tensor]:
        for feature in self.features:
            yield from feature.parameters()