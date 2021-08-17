from .value import Value, Identifier
from typing import List
from dataclasses import dataclass

@dataclass
class Statement:
    """Statement of the form `target <- func(*args)`."""

    target : str
    function : str
    arguments : List[Value]

    def dependencies(self) -> List[str]:
        """Return a list of all dependencies in the right-hand side of the statement."""
        def gen():
            for arg in self.arguments:
                if isinstance(arg, Identifier):
                    yield arg
        return list(gen())

    # IO

    @classmethod
    def load(cls, json) -> "Statement":
        """Construct a statement from a JSON-like object."""

        if not json["type"] == "statement":
            raise TypeError(f"{json} does not represent a statement.")

        target = json["target"]
        function = json["function"]
        arguments = [Value.load(arg) for arg in json["arguments"]]

        return cls(target, function, arguments)

    def dump(self):
        """Construct a JSON-like representation of the statement."""

        return {
            "type" : "statement",
            "target" : self.target,
            "function" : self.function,
            "arguments" : [arg.dump() for arg in self.arguments]
        }