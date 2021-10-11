from torch import tensor, Tensor
from typing import List

class Choice:
    def __init__(self, features : Tensor, context : List[Tensor]):
        self.features, self.context = features, context

    @classmethod
    def of_json(cls, json) -> "Choice":
        assert json["type"] == "choice"

        features = tensor(json["embedding"])
        context = [tensor(ctx) for ctx in json["context"]]
        return cls(features, context)

class History:
    def __init__(self, choices : List[Choice]):
        self.choices = choices

    @classmethod
    def of_json(cls, json) -> "History":
        assert json["type"] == "history"

        choices = [Choice.of_json(c) for c in json["choices"]]
        return cls(choices)
