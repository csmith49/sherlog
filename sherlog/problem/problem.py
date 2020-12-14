from .namespace import Namespace
from .parameter import Parameter
from .evidence import Evidence
from .observation import Observation
from .. import interface
from ..engine import Model, value, Store
from ..story import Story

import torch
import pickle

class Problem:
    def __init__(self, parameters, namespaces=None, evidence=None, program=None):
        '''Object representing a SherLog problem file.

        Parameters
        ----------
        parameters : Parameter iterable

        namespaces : string iterable

        evidence : Evidence iterable

        program : JSON-like object
        '''
        # convert params to name-param map
        self._parameters = {p.name : p for p in parameters}
        # build the namespace obj from the namespaces provided
        self._namespace = Namespace(modules=namespaces)
        # just record the rest
        self._evidence = evidence
        self.program = program

    @classmethod
    def of_json(cls, json):
        '''Builds a problem from a JSON-like object.

        Parameters
        ----------
        json : JSON-like object
        
        Returns
        -------
        Problem
        '''
        parameters = [Parameter.of_json(p) for p in json["parameters"]]
        namespaces = json["namespaces"]
        evidence = [Evidence.of_json(e) for e in json["evidence"]]
        program = json["program"]
        return cls(parameters=parameters, namespaces=namespaces, evidence=evidence, program=program)

    def stories(self):
        for evidence in self._evidence:
            result = interface.query(self.program, evidence.atoms)
            model = Model.of_json(result["model"])
            observations = [Observation.of_json(obs) for obs in result["observations"]]
            for context in evidence.concretize(self._namespace):
                external = (context, self.parameter_map, self._namespace)
                yield Story(model, observations, external=external)

    def save_parameters(self, filepath):
        '''Writes all parameter values in scope to a file.

        Parameters
        ----------
        filepath : string
        '''
        output = { "parameters" : {}, "models" : {} }
        for p, parameter in self._parameters.items():
            output["parameters"][p] = parameter.value
        for name, obj in self._namespace.items():
            if hasattr(obj, "state_dict"):
                output["models"][name] = obj.state_dict()
        with open(filepath, "wb") as f:
            pickle.dump(output, f)

    def load_parameters(self, filepath):
        '''Updates (in-place) all parameter values in scope with the values contained in the file.

        Paramters
        ---------
        filepath : string
        '''
        with open(filepath, "rb") as f:
            params = pickle.load(f)
        for p, value in params["parameters"].items():
            self._parameters[p].value = value
        for name, state_dict in params["models"].items():
            model = self._namespace[name]
            model.load_state_dict(state_dict)

    def clamp_parameters(self):
        '''Updates the value of all parameters in-place to satisfy the constraints of their domain.'''
        with torch.no_grad():
            for _, parameter in self._parameters.items():
                parameter.clamp()

    def parameters(self):
        '''Returns an iterable over all tuneable parameters in-scope.

        Returns
        -------
        torch.tensor iterable
        '''
        for _, parameter in self._parameters.items():
            yield parameter.value
        for _, obj in self._namespace.items():
            if hasattr(obj, "parameters"):
                yield from obj.parameters()

    @property
    def parameter_map(self):
        return {n : p.value for n, p in self._parameters.items()}

def load(filepath: str):
    """Load a problem from a filepath.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Problem
    """
    with open(filepath, "r") as f:
        contents = f.read()
    json = interface.parse(contents)
    return Problem.of_json(json)