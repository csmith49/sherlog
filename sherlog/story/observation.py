from . import term
from . import semantics
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

    def evaluate(self, context, algebra):
        '''Evaluates an observation in a context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        (string, Value) dict
        '''
        for name, value in self.observations.items():
            yield (name, semantics.evaluate(value, context, algebra))

    def to_tensor(self, context, algebra):
        '''Converts an observation to a tensor by evaluating in a context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        torch.tensor
        '''
        return torch.tensor([value for _, value in self.evaluate(context, algebra)])

    def project_context_to_tensor(self, context):
        '''Converts a context to a tensor via projection of the keys of the observation.

        Parameters
        ----------
        context : Context

        Returns
        -------
        torch.tensor
        '''
        return torch.tensor([context[name] for name, _ in self.items()])

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
            self.to_tensor(context, semantics.torch.algebra),
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
            self.to_tensor(context, semantics.torch.algebra),
            self.project_context_to_tensor(context),
            dim=0
        )
