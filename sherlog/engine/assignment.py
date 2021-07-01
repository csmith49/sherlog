from .value import Value, Identifier
from typing import List, Iterable

class Assignment:
    """Exectuable statement of the form `target <- function(arguments)`."""

    def __init__(self, target : Identifier, guard : str, arguments : List[Value]):
        """Construct an assignment.
        
        Parameters
        ----------
        target : Identifier
        
        guard : str
        
        arguments : List[Value]
        """
        self.target, self.guard, self.arguments = target, guard, arguments

    def dependencies(self) -> Iterable[Identifier]:
        """Compute identifier dependencies. A dependency is any identifier whose value must be known in order to evaluate statement.
        
        Returns
        -------
        Iterable[Identifier]
        """
        for arg in self.arguments:
            if isinstance(arg, Identifier):
                yield arg

    # CONSTRUCTION
    @classmethod
    def of_json(cls, json) -> 'Assignment':
        """Build an assignment statement from a JSON representation.
        
        Parameters
        ----------
        json : JSON-like object
        
        Returns
        -------
        Assignment
        """
        target = Identifier(json["target"])
        guard = json["guard"]
        arguments = [Value.of_json(p) for p in json["parameters"]]
        return cls(target, guard, arguments)

    # MAGIC METHODS
    def __str__(self):
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.target} <- {self.guard}({args})"
