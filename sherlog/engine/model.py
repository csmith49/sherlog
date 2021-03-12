import networkx as nx
from ..interface import console
from . import value
from .assignment import Assignment

class Model:
    def __init__(self, assignments):
        """A collection of assignments encoding a generative model.

        Parameters
        ----------
        statements : list of Statement objects

        Returns
        -------
        Model
        """
        self._assignments = assignments

        # build the target map
        self._target_map = {}
        for assignment in self._assignments:
            self._target_map[assignment.target] = assignment

        # build the dataflow graph
        self._dataflow_graph = nx.DiGraph()
        for assignment in self._assignments:
            self._dataflow_graph.add_node(assignment.target)
            for dependency in assignment.dependencies():
                self._dataflow_graph.add_edge(dependency, assignment.target)
 
    @property
    def assignments(self):
        """An iterable of assignments in the model in topological order.

        The order is determined by variable dependencies.

        Returns
        -------
        Assignment iterable
        """
        for node in nx.algorithms.dag.topological_sort(self._dataflow_graph):
            yield self._target_map[node]

    @classmethod
    def of_json(cls, json):
        """Builds a model from a JSON-like representation.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Model
        """
        assignments = [Assignment.of_json(line) for line in json]
        return cls(assignments)

    def __str__(self):
        return str([str(assignment) for assignment in self.assignments])