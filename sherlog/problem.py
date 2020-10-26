from .engine import parse, register, query
from .story import Story
from importlib import import_module
import torch.distributions.constraints as constraints
import torch

class Parameter:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.value = torch.tensor(0.5, requires_grad=True)

    def constraint(self):
        if self.domain == "unit":
            return constraints.unit_interval
        elif self.domain == "positive":
            return constraints.positive
        elif self.domain == "real":
            return constraints.real
        else: raise NotImplementedError()

    @classmethod
    def of_json(cls, json):
        name = json["name"]
        domain = json["domain"]
        return cls(name, domain)

class Function:
    def __init__(self, method, module):
        self._method = method
        self._module = module
        self.module = import_module(self._module)
        self.method = self.module[self._method]

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    @classmethod
    def of_json(cls, json):
        method = json["method"]
        module = json["module"]
        return cls(method, module)

class Problem:
    def __init__(self, parameters=None, functions=None, evidence=None, program=None, queries=None):
        self.parameters = parameters
        self.functions = functions
        self.evidence = evidence
        self.program = program
        self.queries = query
        # make sure the associated program is actually registered
        register(self.program)

    @classmethod
    def of_json(cls, json):
        parameters = {p.name : p for p in [Parameter.of_json(p) for p in json["parameters"]]}
        functions = {f.name : f for f in [Function.of_json(f) for f in json["functions"]]}
        evidence = json["evidence"]
        program = json["program"]
        queries = json["queries"]
        return cls(parameters=parameters, functions=functions, evidence=evidence, program=program, queries=queries)

    def generative_story(self, evidence):
        story, observations = query(evidence)
        return Story.of_json(story, observations, self.parameters, self.functions)
    
    def trainable_parameters(self):
        for _, v in self.parameters.items():
            yield v.value

def load_problem_file(filepath):
    with open(filepath, "r") as f:
        contents = f.read()
    json = parse(contents)
    return Problem.of_json(json)