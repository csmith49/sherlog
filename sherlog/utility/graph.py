import networkx as nx
from typing import NewType
from ..engine import parse

# need to be able to dump and load graphs
def dump(graph, path):
    """Write graph to file.

    Parameters
    ----------
    graph
        NetworkX graph to write to file.
    path : str
        Filepath to write to.
    """
    nx.readwrite.write_gml(graph, path)

def load(path):
    """Load graph from file.

    Parameters
    ----------
    path : str
        Filepath to load from.
    
    Returns
    -------
    NetworkX graph encoded by the file at `path`.
    """
    return nx.readwrite.read_gml(path)

# convert ids to value
def convert_id(kind, id):
    """Utility function for converting nodes to Sherlog values.

    Parameters
    ----------
    kind : str
    id : int

    Returns
    -------
    String representation of the form `{kind}_{id}`.
    """
    return f"{kind}_{id}"

# generate graphs
def random_power_law_graph(size: int, degree: int):
    """Build a random scale-free graph.

    Utilizes the Barabasi-Albert attachment algorithm.

    Parameters
    ----------
    size : int
        Desired number of nodes in the graph.
    degree : int
        Minimal degree of each node in the graph.

    Returns
    -------
    NetworkX graph with `size` nodes.
    """
    return nx.generators.random_graphs.barabasi_albert_graph(size, degree)

# convert graphs
def convert_graph(graph, node_kind, node_relation, edge_relation):
    """Convert a graph to a Sherlog representation.

    Parameters
    ----------
    graph
        NetworkX graph to convert.
    node_kind : str
        The "kind" each node represents.
    node_relation : str
        The name of the relation that defines the set of nodes.
    edge_relation : str
        The name of the relation that defines the edges of the graph.

    Returns
    -------
    Sherlog representation of the input graph.
    """
    output = []

    # build node relations
    for id in graph.nodes:
        node = convert_id(node_kind, id)
        rel = f"{node_relation}({node})."
        output.append(rel)

    # build edge relations
    for s_id, d_id in graph.edges:
        s_node, d_node = convert_id(node_kind, s_id), convert_id(node_kind, d_id)
        rel = f"{edge_relation}({s_node}, {d_node})."
        output.append(rel)

    # use parse utility to generate JSON rep
    return parse('\n'.join(output))