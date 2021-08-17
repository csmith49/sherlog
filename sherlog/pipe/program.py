from typing import List, Iterable
from .statement import Statement

from itertools import unique
from networkx import DiGraph
from networkx.algorithms.dag import topological_order

class Program:
    """Pipe programs are collections of statements."""

    def __init__(self, statements : List[Statement]):
        """Construct a program."""

        self.statements = statements

        self._source_map = {statement.target : statement for statement in self.statements}
        
        self._targets = set(self._source_map.keys())

        self._dependency_graph = DiGraph()
        for target in self.targets():
            self._dependency_graph.add_node(target)
        
        for target in self.targets():
            for dependency in self.source(target).dependencies():
                if self.is_target(dependency):
                    self._dependency_graph.add_node(target, dependency)

    def targets(self) -> Iterable[str]:
        """Iterate over all targets in the program."""

        yield from self._targets

    def is_target(self, target : str) -> bool:
        """Checks if the provided string is a target in the program."""

        return target in self._targets

    def parameters(self) -> Iterable[str]:
        """Iterate over all non-target identifiers in the program."""

        def gen():
            for target in self.targets():
                for dependency in self.source(target).dependencies():
                    if not self.is_target(dependency):
                        yield dependency
        yield from unique(gen())

    def source(self, target : str):
        """Return the source statement for a target."""

        return self._source_map[target]

    def evaluation_order(self) -> Iterable[Statement]:
        """Iterate over all statements in the program in an order suitable for evaluation."""

        for target in topological_order(self._dependency_graph).reverse():
            yield self.source(target)

    # IO

    @classmethod
    def load(cls, json) -> "Program":
        """Construct a program form a JSON-like object."""

        if json["type"] != "program":
            raise TypeError(f"{json} does not represent a program.")
        
        statements = [Statement.load(stmt) for stmt in json["statements"]]

        return cls(statements)

    def dump(self):
        """Construct a JSON-like object representing the program."""
        return {
            "type" : "program",
            "statements" : [stmt.dump() for stmt in self.statements]
        }