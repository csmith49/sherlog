from torch import tensor, Tensor
from typing import List

class Embedding:
    def __init__(self, features : Tensor, weights : Tensor):
        self.features = features
        self.weights = weights

    @classmethod
    def of_json(cls, json) -> "Embedding":
        assert json["type"] == "embedding"

        features = tensor(json["features"])
        weights = tensor(json["weights"])

        return cls(features, weights)

class Choice:
    def __init__(self, features : Tensor, context : List[Tensor]):
        self.features, self.context = features, context

    @classmethod
    def of_json(cls, json) -> "Choice":
        assert json["type"] == "choice"

        features = Embedding.of_json(json["embedding"])
        context = [Embedding.of_json(ctx) for ctx in json["context"]]
        return cls(features, context)

class History:
    def __init__(self, choices : List[Choice]):
        self.choices = choices

    @classmethod
    def of_json(cls, json) -> "History":
        assert json["type"] == "history"

        choices = [Choice.of_json(c) for c in json["choices"]]
        return cls(choices)
