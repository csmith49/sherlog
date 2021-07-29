from ...engine import Functor
from ...logs import get
import torch
import networkx as nx

logger = get("story.semantics.graph")

class Node:
    def __init__(self, value, parents=None, attributes=None):
        """A value wrapped by the Graph functor.

        Parameters
        ----------
        value : Value

        parents : Optional[Iterable[Node]]
        
        attributes : Optional[Dict[str, Any]]
        """
        self.value = value
        self._parents = parents
        self._attributes = attributes

    @property
    def parents(self):
        if self._parents:
            return self._parents
        else:
            return []

    @property
    def attributes(self):
        if self._attributes:
            return self._attributes
        else:
            return {}

    @property
    def label(self):
        return f"{self.value}"

def to_graph(*args):
    """Build a graph from the dependencies of all provided nodes.

    Parameters
    ----------
    *args : Iterable[Node]

    Returns
    -------
    networkx.DiGraph
    """

    g = nx.DiGraph()
    worklist = list(args)
    while worklist:
        node = worklist.pop()
        g.add_node(node, **node.attributes)
        for parent in node.parents:
            g.add_edge(parent, node)
            worklist.extend(node.parents)
    return g

def wrap(obj, **kwargs):
    attributes = {}

    # if obj is a tensor with a gradient, its a parameter to the program
    if torch.is_tensor(obj) and obj.requires_grad:
        attributes["type"] = "parameter"

    # if obj is anything else, it's a constant
    else:
        attributes["type"] = "constant"
    
    return Node(obj, attributes=attributes)

def fmap(callable, args, kwargs, **fmap_args):
    attributes = {"type" : "external"}
    return Node(callable, parents=list(args), attributes=attributes)

class BuiltinFactory:
    def __init__(self):
        """Factory for constructing builtins in the Graph functor."""
        logger.info(f"Factory {self} built.")

    def build(self, name):
        # build an attribute based on the name of the function
        attributes = {}
        if name in ["bernoulli", "normal", "beta"]:
            attributes["type"] = "distribution"
        else:
            attributes["type"] = "function"

        # the builtin will always use this attribute
        def builtin(*args, **kwargs):
            return Node(name, parents=list(args), attributes=attributes)
        return builtin
    
    def __getitem__(self, key):
        logger.info(f"Builtin requested for {key}.")
        return self.build(key)

functor = Functor(wrap, fmap, BuiltinFactory())

# DRAWING FUNCTIONALITY
def to_dot(graph, output):
    """Write Graphviz representation to output path.

    Parameters
    ----------
    graph : networkx.Graph

    output : str
    """
    nx.drawing.nx_agraph.write_dot(graph, output)

def draw_matplotlib(graph, output=None):
    """Uses NetworkX's Matplotlib interface to render graphs.

    Parameters
    ----------
    graph : networkx.Graph

    output : Optional[str]
    """
    # make sure we have the proper inputs and displays
    import matplotlib.pyplot as plt

    # step 1 - get the node layout
    layout = nx.planar_layout(graph)

    # step 2 - draw the node labels (diff colors for diff types)
    
    # parameters
    nx.draw_networkx_labels(
        graph,
        labels={k : k.label for k, v in graph.nodes(data="type") if v == "parameter"},
        pos=layout,
        font_weight="bold",
        font_color="tab:red"
    )

    # constants
    nx.draw_networkx_labels(
        graph,
        labels={k : k.label for k, v in graph.nodes(data="type") if v == "constant"},
        pos=layout,
        font_weight="bold",
        font_color="tab:green"
    )

    # distributions
    nx.draw_networkx_labels(
        graph,
        labels={k : k.label for k, v in graph.nodes(data="type") if v == "distribution"},
        pos=layout,
        font_weight="bold",
        font_color="tab:blue"
    )

    # function
    nx.draw_networkx_labels(
        graph,
        labels={k : k.label for k, v in graph.nodes(data="type") if v == "function"},
        pos=layout,
        font_weight="bold"
    )

    # external
    nx.draw_networkx_labels(
        graph,
        labels={k : k.label for k, v in graph.nodes(data="type") if v == "external"},
        pos=layout,
        font_weight="bold",
        font_color="tab:orange"
    )

    # step 3 - add edges and labels
    nx.draw_networkx_edges(
        graph, 
        pos=layout,
        arrowstyle="->",
        min_source_margin=15,
        min_target_margin=15)

    # save if told, otherwise just show
    if output:
        plt.savefig(output)
    else:
        plt.show()