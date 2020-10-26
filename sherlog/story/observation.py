from . import term

class Observation:
    def __init__(self, observations):
        self.observations = observations

    def __str__(self):
        args = (f"{k}/{v}" for k, v in self.observations.items())
        return f"[{', '.join(args)}]"

    def items(self):
        yield from self.observations.items()

    @classmethod
    def of_json(cls, json):
        observations = {entry["variable"] : term.of_json(entry["value"]) for entry in json}
        return cls(observations)

def of_json(json):
    return Observation.of_json(json)