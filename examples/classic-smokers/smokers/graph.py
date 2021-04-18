from networkx.generators.directed import scale_free_graph
from networkx.algorithms.traversal import dfs_preorder_nodes
from networkx import DiGraph
from random import random
from typing import Iterable, Tuple, Mapping, Set, Union, Optional

# UTILITY

def flip(weight):
    """Returns true with probability `weight`, and false with probability `1 - weight`.
    
    Parameters
    ----------
    weight : float

    Returns
    -------
    bool
    """
    return random() <= weight

# PARAMETERIZATION

from dataclasses import dataclass

@dataclass
class Parameterization:
    stress : float
    spontaneous : float
    comorbid : float
    influence : float

    def label(self, nodes : Iterable[int], edges : Iterable[Tuple[int, int]]):
        """Construct intensional labels for a provided set of nodes and edges.

        Parameters
        ----------
        nodes : Iterable[int]
        edges : Iterable[int, int]
        """
        mapping = {
            "stress" : set(),
            "spontaneous" : set(),
            "comorbid" : set(),
            "influence" : set()
        }

        for node in nodes:
            if flip(self.stress): mapping["stress"].add(node)
            if flip(self.spontaneous): mapping["spontaneous"].add(node)
            if flip(self.comorbid): mapping["comorbid"].add(node)

        for edge in edges:
            if flip(self.influence): mapping["influence"].add(edge)

        return mapping

# SOCIAL GRAPH

class Graph:
    def __init__(self, size : int, parameterization : Parameterization, classification_target : int = 0):
        """A directed scale-free graph of the indicated size (in terms of nodes).

        Parameters
        ----------
        size : int
        parameterization : Parameterization
        classification_target : int (default=0)
        """
        self._size = size
        self._parameterization = parameterization
        self._classification_target = classification_target
        self._graph = scale_free_graph(size)

        mapping = parameterization.label(self.nodes(), self.edges())
        self._smokes, self._asthma = reference_implementation(self, mapping)

    def nodes(self) -> Iterable[int]:
        return self._graph.nodes()

    def edges(self) -> Iterable[Tuple[int, int]]:
        return self._graph.edges()

    def _symbol(self, node : int, index : Optional[int] = None) -> str:
        """Converts a node to a symbol. If the index is provided, nodes are additionally subscripted.

        Parameters
        ----------
        node : int
        index : Optional[int]

        Returns
        -------
        str
        """
        if index is None:
            return f"p_{node}"
        else:
            return f"p_{index}_{node}"

    def people(self, index : Optional[int] = None) -> Iterable[str]:
        """Yields all symbols to be captured by the `person` relation.

        Parameters
        ----------
        index : Optional[int]

        Returns
        -------
        Iterable[str]
        """
        for node in self.nodes():
            yield self._symbol(node, index=index)

    def friends(self, index : Optional[int] = None) -> Iterable[Tuple[str, str]]:
        """Yields all pairs of symbols to be captured by the `friend` relation.

        Parameters
        ----------
        index : Optional[int]

        Returns
        -------
        Iterable[Tuple[str, str]]
        """
        for (s, d) in self.edges():
            yield (
                self._symbol(s, index=index),
                self._symbol(d, index=index)
            )

    def smokes(self, value : bool, index : Optional[int] = None, avoid_classification_target : bool = False) -> Iterable[str]:
        """Yields all symbols whose smoking truthiness matches the provided value.

        Parameters
        ----------
        value : bool
        index : Optional[int]
        avoid_classification_target : bool (default=False)
        
        Returns
        -------
        Iterable[str]
        """
        for node in self.nodes():
            if (node in self._smokes) == value:
                if not avoid_classification_target or node != self._classification_target:
                    yield self._symbol(node, index=index)

    def asthma(self, value : bool, index : Optional[int] = None, avoid_classification_target : bool = False) -> Iterable[str]:
        """Yields all symbols whose asthma truthiness matches the provided value.

        Parameters
        ----------
        value : bool
        index : Optional[int]
        avoid_classification_target : bool (default=False)

        Returns
        -------
        Iterable[str]
        """
        for node in self.nodes():
            if (node in self._asthma) == value:
                if not avoid_classification_target or node != self._classification_target:
                    yield self._symbol(node, index=index)

    def classification_target_symbol(self, index : Optional[int] = None) -> str:
        """The symbol for the classification target (with optional index).

        Parameters
        ----------
        index : Optional[int]

        Returns
        -------
        str
        """
        return self._symbol(self._classification_target, index=index)

    def classification_target_smoke(self) -> bool:
        return self._classification_target in self._smokes
    
    def classification_target_asthma(self) -> bool:
        return self._classification_target in self._asthma

def reference_implementation(graph, mapping):
    """Reference implementation for deriving extensional labels for a social graph.

    Parameters
    ----------
    graph : Graph
    mapping : Mapping[str, Union[Set[int], Set[Tuple[int, int]]]]

    Returns
    -------
    Tuple[Set[int], Set[int]]
    """
    smokes, asthma = set(), set()

    # stress -> smokes
    smokes.update(mapping["stress"])

    # inf, smokes -> smokes
    inf_graph = DiGraph()
    inf_graph.add_nodes_from(graph.nodes())
    inf_graph.add_edges_from(mapping["influence"])

    for node in graph.nodes():
        if node in smokes:
            reachable = dfs_preorder_nodes(inf_graph, node)
            smokes.update(reachable)
    
    # spontaneous -> asthma
    asthma.update(mapping["spontaneous"])

    # smokes, comorbid -> asthma
    asthma.update(smokes.intersection(mapping["comorbid"]))

    return smokes, asthma

def to_sherlog(task):
    """Convert a social graph with concrete observations to a Sherlog problem.

    Parameters
    ----------
    task : Dict[str, Any]

    Returns
    -------
    str
    """
    with open("./programs/smokers.sl", "r") as f:
        source = f.read()
    
    graph_atoms, evidence = [], []

    # convert nodes
    for node in task["graph"]["nodes"]:
        graph_atoms.append(f"person(p_{node}).")
    
    # convert edges
    for (s, d) in task["graph"]["edges"]:
        graph_atoms.append(f"friend(p_{s}, p_{d}).")

    graph = '\n'.join(graph_atoms)

    # convert observations to evidence
    for observation in task["observations"]:
        atoms = []
        
        for node in observation["smokes"]["true"]:
            atoms.append(f"smokes(p_{node})")
        for node in observation["smokes"]["false"]:
            atoms.append(f"not_smokes(p_{node})")
        for node in observation["asthma"]["true"]:
            atoms.append(f"asthma(p_{node})")
        for node in observation["asthma"]["false"]:
            atoms.append(f"not_asthma(p_{node})")
        
        evidence.append(f"!evidence {', '.join(atoms)}.")
    
    return source + "\n\n" + graph + "\n\n" + '\n'.join(evidence)
