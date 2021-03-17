from .namespace import Namespace
from .parameter import Parameter
from ..inference import Objective
from .. import interface
from ..engine import Model, value, Store
from ..story import Story
from ..logs import get

from itertools import islice

import torch
import pickle
import random

logger = get("problem")

class Problem:
    def __init__(self, parameters, namespaces=None, evidence=None, program=None):
        """Object representing a SherLog problem file.

        Parameters
        ----------
        parameters : Parameter iterable

        namespaces : string iterable

        evidence : JSON-like object iterable

        program : JSON-like object
        """
        # convert params to name-param map
        self._parameters = {p.name : p for p in parameters}
        # build the namespace obj from the namespaces provided
        self._namespace = Namespace(modules=namespaces)
        # just record the rest
        self._evidence = evidence
        self.program = program

    @property
    def evidence(self):
        yield from self._evidence

    @classmethod
    def of_json(cls, json):
        """Build a problem from a JSON-like object.

        Parameters
        ----------
        json : JSON-like object
        
        Returns
        -------
        Problem
        """
        parameters = [Parameter.of_json(p) for p in json["parameters"]]
        # namespaces = json["namespaces"]
        evidence = json["evidence"]
        program = json
        return cls(parameters=parameters, namespaces=[], evidence=evidence, program=program)

    def stories(self, evidence, samples=1, attempts=100, width=None, depth=None):
        """Construct all stories encoded by the problem.

        Parameters
        ----------
        evidence : JSON-like object

        samples : int

        attempts : int
        
        width : int option
        
        depth : int option

        Returns
        -------
        Story list iterable
        """

        logger.info(f"Sampling stories for evidence {evidence}...")

        # build the external evaluation context with the namespace
        external = (self.parameter_map, self._namespace)

        # build the generator
        def gen():
            for attempt in range(attempts):
                logger.info(f"Starting story generation attempt {attempt}...")
                for json in interface.query(self.program, evidence, width=width, depth=depth):
                    logger.info("Story found.")
                    yield Story.of_json(json, external=external)
        
        # and grab only the number of samples desired
        yield from islice(gen(), samples)

    def all_stories(self, **kwargs):
        for evidence in self.evidence:
            yield from self.stories(evidence, **kwargs)

    def likelihood(self, evidence, **kwargs):
        lls = torch.tensor([story.dice() for story in self.stories(evidence, **kwargs)])
        return torch.mean(lls)

    def log_likelihood(self, **kwargs):
        lls = torch.tensor([self.likelihood(evidence, **kwargs) for evidence in self.evidence])
        total = torch.sum(lls)
        return Objective("log_likelihood", total)    

    def save_parameters(self, filepath):
        """Write all parameter values in scope to a file.

        Parameters
        ----------
        filepath : string
        """
        output = { "parameters" : {}, "models" : {} }
        for p, parameter in self._parameters.items():
            output["parameters"][p] = parameter.value
        for name, obj in self._namespace.items():
            if hasattr(obj, "state_dict"):
                output["models"][name] = obj.state_dict()
        with open(filepath, "wb") as f:
            pickle.dump(output, f)

    def load_parameters(self, filepath):
        """Update (in-place) all parameter values in scope with the values contained in the file.

        Paramters
        ---------
        filepath : string
        """
        with open(filepath, "rb") as f:
            params = pickle.load(f)
        for p, value in params["parameters"].items():
            self._parameters[p].value = value
        for name, state_dict in params["models"].items():
            model = self._namespace[name]
            model.load_state_dict(state_dict)

    def clamp_parameters(self):
        """Update the value of all parameters in-place to satisfy the constraints of their domain."""
        with torch.no_grad():
            for _, parameter in self._parameters.items():
                parameter.clamp()

    def parameters(self):
        """Return an iterable over all tuneable parameters in-scope.

        Returns
        -------
        tensor iterable
        """
        for _, parameter in self._parameters.items():
            yield parameter.value
        for _, obj in self._namespace.items():
            if hasattr(obj, "parameters"):
                yield from obj.parameters()

    @property
    def parameter_map(self):
        return {n : p.value for n, p in self._parameters.items()}

    # def log_likelihood(self, num_samples=1):
    #     total = torch.tensor(0.0)
    #     for story in self.stories():
    #         total += torch.log(story.likelihood(num_samples=num_samples))
    #     return total