from . import term

class Statement:
    def __init__(self, variable, dependencies, function, arguments):
        """A statement in a generative story.

        Parameters
        ----------
        variable : Variable

        dependencies : Variable list

        function : string

        arguments : Term list
        """
        self.variable = variable
        self.dependencies = dependencies
        self.function = function
        self.arguments = arguments

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
        function = json["generation"]["function"]
        arguments = [term.of_json(v) for v in json["generation"]["arguments"]]
        return cls(variable, dependencies, function, arguments)

    def run(self, context, semantics, **kwargs):
        """Evaluates the statement in the provided namespace.

        Parameters
        ----------
        context : context.Context

        Returns
        -------
        string

        Value
        """
        value = semantics(self.variable.name, self.function, self.arguments, context, **kwargs)
        return self.variable.name, value