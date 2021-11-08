from typing import List, Optional, Iterable
from torch import tensor, Tensor

class Operation:
    """Thin wrapper around JSON representation of operations, which map proof branches to real-valued scores."""

    def __init__(self, json):
        """Construct an operation from a JSON representation."""
        self.source = json

    @classmethod
    def of_json(cls, json) -> "Operation":
        assert json["type"] == "operation"

        return cls(json)

    def to_json(self):
        """Return the JSON representation for `self`."""

        return self.source

class Feature:
    """Features pair operations with a weight indicating the contribution of the operation."""

    def __init__(self, weight : float, operation : Operation):
        """Construct a feature from a weight and an operation."""
    
        self.weight = tensor(weight, requires_grad=True)
        self.operation = operation

    @classmethod
    def of_json(cls, json) -> "Feature":
        assert json["type"] == "feature"

        weight = json["weight"]
        operation = Operation.of_json(json["operation"])

        return cls(weight, operation)

    def to_json(self):
        """Return a JSON representation for `self`."""

        return {
            "type" : "feature",
            "weight" : self.weight.item(),
            "operation" : self.operation.to_json()
        }

    def parameters(self) -> Iterable[Tensor]:
        """Yield all optimizable parameters in the feature."""

        yield self.weight

class Posterior:
    """A posterior collects features."""

    def __init__(self, features : List[Feature]):
        """Construct a posterior from a list of features."""

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