from . import term
import torch

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
                if not torch.is_tensor(value):
                    yield (name, torch.tensor(value))
                else:
                    yield (name, value)

    def to_tensor(self, context):
        '''Converts an observation to a tensor by evaluating in a context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        torch.tensor
        '''
        return torch.tensor([value.value for _, value in self.evaluate(context)])

    def project_context_to_tensor(self, context):
        '''Converts a context to a tensor via projection of the keys of the observation.

        Parameters
        ----------
        context : Context

        Returns
        -------
        torch.tensor
        '''
        return torch.tensor([context[name].value for name, _ in self.items()])

    def distance(self, context, p=1):
        '''Computes the L-p distance between the observation and the context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        torch.tensor
        '''
        return torch.dist(
            self.to_tensor(context),
            self.project_context_to_tensor(context),
            p=p
        )

    def similarity(self, context):
        '''Computes the cosine similarity between the observation and the context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        torch.tensor
        '''
        return torch.cosine_similarity(
            self.to_tensor(context),
            self.project_context_to_tensor(context),
            dim=0
        )
