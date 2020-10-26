from . import term
from . import generation

class Statement:
    def __init__(self, variable, dependencies, generation):
        """A statement in a generative story.

        Parameters
        ----------
        variable : term.Variable

        dependencies : term.Variable list

        generation : generation.Generation
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
        json : JSON object
            A JSON representation, as encoded by `json.loads(...)`

        Returns
        -------
        Statement
            A statement in the generative story

        """
        variable = term.Variable(json["variable"])
        dependencies = [term.Variable(dep) for dep in json["dependencies"]]
        gen = generation.of_json(json["generation"])
        return cls(variable, dependencies, gen)

    def run(self, namespace, functions):
        """Evaluates the statement in the provided namespace.

        Parameters
        ----------
        namespace : string-to-term dictionary

        functions : string-to-function dictionary

        Returns
        -------
        string

        torch.tensor
        """
        value, _ = self.generation.to_torch(namespace, functions)
        return self.variable.name, value

    def dice(self, namespace, functions):
        value, log_prob = self.generation.to_torch(namespace, functions)
        return self.variable.name, value, log_prob

def of_json(json):
    return Statement.of_json(json)