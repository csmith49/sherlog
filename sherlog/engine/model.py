import networkx as nx
from typing import Iterable
from .assignment import Assignment

class Model:
    """A sequence of assignments encoding a set of generative semantics."""

    def __init__(self, assignments : Iterable[Assignment]):
        """Construct a model from a sequence of assignments.
        
        Parameters
        ----------
        assignments : Iterable[Assignment]
        """
        # store hidden, we'll reveal assignments in topological order later
        self._assignments = list(assignments)

        # build the target map
        self._targets = {}
        for assignment in self._assignments:
            self._targets[assignment.target] = assignment

        # and the dataflow graph
        self._dataflow = nx.DiGraph()
        for assignment in self._assignments:
            self._dataflow.add_node(assignment.target)
            for dependency in assignment.dependencies():
                self._dataflow.add_edge(dependency, assignment.target)
    
    @property
    def assignments(self) -> Iterable[Assignment]:
        for node in nx.algorithms.dag.topological_sort(self._dataflow):
            try:
                yield self._targets[node]
            except KeyError: pass

    # CONSTRUCTION
    @classmethod
    def of_json(cls, json) -> 'Model':
        """Construct a model from a JSON-like representation.
        
        Parameters
        ----------
        json : JSON-like object
        
        Returns
        -------
        Model
        """
        assignments = [Assignment.of_json(line) for line in json]
        return cls(assignments)

    # MAGIC METHODS
    def __str__(self):
        return str([str(assignment) for assignment in self.assignments])