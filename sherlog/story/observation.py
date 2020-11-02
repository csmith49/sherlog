from . import term
from .context import Value

class Observation:
    def __init__(self, observations):
        '''A set of desired symbolic observations of variables in a Story.

        Parameters
        ----------
        observations : (string, Term) dict
        '''
        self.observations = observations

    def __str__(self):
        args = (f"{k}/{v}" for k, v in self.observations.items())
        return f"[{', '.join(args)}]"

    def items(self):
        yield from self.observations.items()

    def variables(self):
        yield from self.observations.keys()

    @classmethod
    def of_json(cls, json):
        '''Builds an Observation from a JSON-like object.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Observation
        '''
        observations = {entry["variable"] : term.of_json(entry["value"]) for entry in json}
        return cls(observations)

    def evaluate(self, context):
        '''Evaluates an observation in a context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        (string, Value) dict
        '''
        for name, value in self.observations.items():
            # if the value is a string, look it up
            if isinstance(value, str):
                yield (name, context.lookup_value(value))
            # otherwise, just lift it
            else:
                yield (name, Value.lift(value))