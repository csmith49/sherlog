from . import term
from .generation import Generation

class Statement:
    def __init__(self, variable, dependencies, generation):
        """A statement in a generative story.

        Parameters
        ----------
        variable : Variable

        dependencies : Variable list

        generation : Generation
        """
        self.variable = variable
        self.dependencies = dependencies
        self.generation = generation

    def __str__(self):
        return f"{self.variable} = {self.generation}"

    @classmethod
    def of_json(cls, json):
        """Build a statement from a JSON encoding.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Statement
        """
        variable = term.Variable(json["variable"])
        dependencies = [term.Variable(dep) for dep in json["dependencies"]]
        gen = Generation.of_json(json["generation"])
        return cls(variable, dependencies, gen)

    def run(self, context):
        """Evaluates the statement in the provided namespace.

        Parameters
        ----------
        context : context.Context

        Returns
        -------
        string

        Value
        """
        value = self.generation.evaluate(self.variable.name, context)
        return self.variable.name, value