from typing import List, Iterable
from .statement import Statement

from itertools import filterfalse
from networkx import DiGraph
from networkx.algorithms.dag import topological_sort

# utility for ensuring uniqueness in enumeration
def unique(iterable, key=None):
    """List unique elements by `key`, if provided."""

    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

class Pipeline:
    """Pipelines are programs that represent sequences of statement evaluations in dependnecy order."""

    def __init__(self, statements : List[Statement]):
        """Construct a pipeline."""

        self.statements = statements

        self._source_map = {statement.target : statement for statement in self.statements}
        
        self._targets = set(self._source_map.keys())

        self._dependency_graph = DiGraph()
        for target in self.targets():
            self._dependency_graph.add_node(target)
        
        for target in self.targets():
            for dependency in self.source(target).dependencies():
                if self.is_target(dependency.value):
                    self._dependency_graph.add_edge(dependency.value, target)
        
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

        for target in topological_sort(self._dependency_graph):
            yield self.source(target)

    # IO

    @classmethod
    def of_json(cls, json) -> "Pipeline":
        """Construct a pipeline form a JSON-like object."""

        if json["type"] != "pipeline":
            raise TypeError(f"{json} does not represent a pipeline.")
        
        statements = [Statement.of_json(stmt) for stmt in json["statements"]]

        return cls(statements)

    def to_json(self):
        """Construct a JSON-like object representing the pipeline."""
        return {
            "type" : "pipeline",
            "statements" : [stmt.to_json() for stmt in self.statements]
        }

    # magic methods

    def __str__(self):
        return '\n'.join((str(stmt) for stmt in self.evaluation_order()))