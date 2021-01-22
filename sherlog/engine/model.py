import networkx as nx
from . import value

class Statement:
    def __init__(self, target, function, arguments):
        """An assignment statement of the form `target = function(arguments)`.

        Parameters
        ----------
        target : Variable
        function : string
        arguments : list of values

        Returns
        -------
        Statement
        """
        self.target = target
        self.function = function
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
        target = value.Variable(json["target"])
        semantics = json["semantics"]
        arguments = [value.of_json(p) for p in json["parameters"]]
        return cls(target, semantics, arguments)

class Model:
    def __init__(self, statements):
        """A collection of statements encoding a generative model.

        Parameters
        ----------
        statements : list of Statement objects

        Returns
        -------
        Model
        """
        self._statements = statements

        # build the target map
        self._target_map = {}
        for statement in self._statements:
            self._target_map[statement.target] = statement

        # build the dataflow graph
        self._dataflow_graph = nx.DiGraph()
        for statement in self._statements:
            self._dataflow_graph.add_node(statement.target)
            for dependency in statement.dependencies():
                self._dataflow_graph.add_edge(dependency, statement.target)
 
    @property
    def statements(self):
        """An iterable of statements in the model in topological order.

        The order is determined by variable dependencies.

        Returns
        -------
        Statement iterable
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
        statements = [Statement.of_json(line) for line in json]
        return cls(statements)
