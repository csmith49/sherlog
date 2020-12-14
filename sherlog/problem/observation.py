from ..engine import value

class Observation:
    def __init__(self, mapping):
        self.mapping = mapping

    @classmethod
    def of_json(cls, json):
        mapping = {}
        for obs in json:
            mapping[obs["variable"]] = value.of_json(obs["value"])
        return cls(mapping)