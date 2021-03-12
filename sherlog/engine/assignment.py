from .value import Variable
from .value import of_json as value_of_json

class Assignment:
    def __init__(self, target, guard, arguments):
        """An assignment statement of the form `target = function(arguments)`.

        Parameters
        ----------
        target : Variable
        guard : string
        arguments : list of values

        Returns
        -------
        Statement
        """
        self.target = target
        self.guard = guard
        self.arguments = arguments

    def dependencies(self):
        """Compute variable dependencies.

        A dependency is a variable whose value must be known to evaluate the statement.

        Returns
        -------
        Variable iterable
        """
        for arg in self.arguments:
            if isinstance(arg, Variable):
                yield arg

    def __str__(self):
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.target} <- {self.guard}({args})"

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

        target = Variable(json["target"])        
        guard = json["guard"]
        arguments = [value_of_json(p) for p in json["parameters"]]
        return cls(target, guard, arguments)