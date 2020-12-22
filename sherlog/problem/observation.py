from ..engine import value, evaluate

class Observation:
    def __init__(self, mapping):
        self.mapping = mapping

    @classmethod
    def of_json(cls, json):
        mapping = {}
        for obs in json:
            mapping[obs["variable"]] = value.of_json(obs["value"])
        return cls(mapping)

    def variables(self):
        for k, _ in self.mapping.items():
            yield value.Variable(k)

    def evaluate(self, store, algebra):
        for _, v in self.mapping.items():
            yield evaluate(v, store, algebra)