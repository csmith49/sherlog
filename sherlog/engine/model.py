import networkx as nx
from . import value

class Statement:
    def __init__(self, target, function, arguments):
        self.target = target
        self.function = function
        self.arguments = arguments

    def dependencies(self):
        for arg in self.arguments:
            if isinstance(arg, value.Variable):
                yield arg

    def __str__(self):
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.target} <- {self.function}({args})"

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
        target = value.Variable(json["variable"])
        function = json["generation"]["function"]
        arguments = [value.of_json(arg) for arg in json["generation"]["arguments"]]
        return cls(target, function, arguments)

class Model:
    def __init__(self, statements):
        self._statements = statements

        # build the target map
        self._target_map = {}
        for statement in self._statements:
            self._target_map[statement.target] = statement

        # build the dataflow graph
        dataflow_edges = []
        for statement in self._statements:
            for dependency in statement.dependencies():
                edge = (dependency, statement.target)
                dataflow_edges.append(edge)
        self._dataflow_graph = nx.DiGraph(dataflow_edges)

    @property
    def statements(self):
        for node in nx.algorithms.dag.topological_sort(self._dataflow_graph):
            yield self._target_map[node]

    @classmethod
    def of_json(cls, json):
        statements = [Statement.of_json(line) for line in json]
        return cls(statements)