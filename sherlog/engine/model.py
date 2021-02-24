import networkx as nx
from . import value

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
            if isinstance(arg, value.Variable):
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
        target = value.Variable(json["target"])
        guard = json["guard"]
        arguments = [value.of_json(p) for p in json["parameters"]]
        return cls(target, guard, arguments)

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
