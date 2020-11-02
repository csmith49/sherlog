from .namespace import Namespace
from .parameter import Parameter
from .evidence import Evidence
from ..story import Story, Context
from ..engine import register, query

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
        namespaces = {n.name : n for n in [Namespace.of_json(n) for n in json["namespaces"]]}
        evidence = [Evidence.of_json(e) for e in json["evidence"]]
        program = json["program"]
        return cls(parameters=parameters, namespaces=namespaces, evidence=evidence, program=program)

    def story(self, atoms):
        '''Constructs a story and set of observations sufficient to generate the atoms.

        Parameters
        ----------
        atoms : JSON-like object list

        Returns
        -------
        Story
        '''
        register(self.program)
        story, observations = query(atoms)
        return Story.of_json(story, observations)

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

    def evidence(self):
        '''Returns an iterable over all flattened evidence.

        Returns
        -------
        (JSON-like object, (string, torch.tensor) dict) iterable
        '''
        for evidence in self._evidence:
            yield from evidence.flatten(self._namespace)

    def stories(self):
        for atoms, map in self.evidence():
            story = self.story(atoms)
            param_map = {n : p.value for n, p in self._parameters.items()}
            context = Context(maps=(map, param_map, self._namespace))
            yield (story, context)

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

    def log_likelihood(self, num_samples=100):
        result = torch.tensor(0.0)
        for (story, context) in self.stories():
            context = story.run(context)
            result += torch.log(story.likelihood(context, num_samples=num_samples))
        return result

    def clamp_parameters(self):
        '''Updates the value of all parameters in-place to satisfy the constraints of their domain.'''
        with torch.no_grad():
            for _, parameter in self._parameters.items():
                parameter.clamp()
