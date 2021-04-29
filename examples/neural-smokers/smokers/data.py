from networkx.generators.directed import scale_free_graph
from networkx.algorithms.traversal import dfs_preorder_nodes
from sherlog.logs import get_external
from networkx import DiGraph
from random import random
from typing import Iterable, Tuple, Mapping, Set, Union, Optional
import torch.nn as nn
import torch.distributions as dist
import torch

logger = get_external("neural-smokers.data")

# utility
def flip(weight : float) -> bool: return random() <= weight

# parameters and models
HEALTH_DIMENSIONS = 5
HEALTH_DISTRIBUTION = dist.Dirichlet(torch.ones(HEALTH_DIMENSIONS))
ASTHMA_RISK_PARAMETERS = torch.logspace(-1, 0, HEALTH_DIMENSIONS)
ASTHMA_RISK = lambda h: torch.dot(ASTHMA_RISK_PARAMETERS, h)
INFLUENCE = lambda x, y: nn.functional.cosine_similarity(x, y, dim=0)
STRESS = 0.1
COMORBID = 0.3

# social graph
class Graph:
    def __init__(self, size : int, classification_target : Tuple[int, int] = (0, 1)):
        logger.info(f"Generating a graph with size {size}.")
        self._size = size
        self._graph = scale_free_graph(size) # somehow this cannot be less than 3?
        self._class_target = classification_target

        self._initialize()

    def nodes(self) -> Iterable[int]:
        return self._graph.nodes()
    
    def edges(self, avoid_classification_target : bool = False) -> Iterable[Tuple[int, int]]:
        for edge in self._graph.edges():
            if not avoid_classification_target or edge != self._class_target:
                yield edge

    def classification_target(self):
        l, r = self._class_target
        return (self._symbol(l), self._symbol(r))

    def people(self) -> Iterable[str]:
        for node in self.nodes():
            yield self._symbol(node)

    def friends(self) -> Iterable[Tuple[str, str]]:
        for (s, d) in self.edges():
            yield (self._symbol(s), self._symbol(d))

    def smokes(self, value : bool) -> Iterable[str]:
        for node in self.nodes():
            if (node in self._smokes) == value:
                yield self._symbol(node)
    
    def asthma(self, value : bool) -> Iterable[str]:
        for node in self.nodes():
            if (node in self._asthma) == value:
                yield self._symbol(node)

    def observed_health(self) -> Iterable[Tuple[str, int]]:
        for node, tensor in self._observed_health.items():
            yield (self._symbol(node), tensor.item())

    def _symbol(self, node : int) -> str:
        """Convert a node to a consistent logical symbol."""
        return f"p_{node}"

    def _initialize(self):
        """Construct the labels for each node and edge in the graph."""
        logger.info(f"Initializing the graph labels.")

        # health(X; flat_dirichlet[10]) <- person(X)
        self._health = {node : HEALTH_DISTRIBUTION.rsample() for node in self.nodes()}
        
        # obs_health(X; categorical[H]) <- health(X, H)
        self._observed_health = {node : dist.Categorical(health).sample() for node, health in self._health.items()}
        
        # asthma_risk(X; dot[H, risk_parameters]) <- health(X, H)
        self._asthma_risk = {node : ASTHMA_RISK(health) for node, health in self._health.items()}
        
        # influence(X, Y; cosine[H, I]) <- health(X, H), health(Y, I), friend(X, Y)
        self._influence = {(s, d) : INFLUENCE(self._health[s], self._health[d]) for (s, d) in self.edges()}
        
        # now compute the smoking and asthma observations accordingly
        self._smokes, self._asthma = set(), set()

        # stress :: smokes(X)
        self._smokes.update([n for n in self.nodes() if flip(STRESS)])
        
        # I :: smokes(X) <- influence(X, Y, I), smokes(Y)
        influence_graph = DiGraph()
        influence_graph.add_nodes_from(self.nodes())
        for edge, influence in self._influence.items():
            if flip(influence): influence_graph.add_edge(*edge)

        for node in self.nodes():
            if node in self._smokes:
                reachable = dfs_preorder_nodes(influence_graph, node)
                self._smokes.update(reachable)

        # comorbid :: asthma(X) <- smokes(X)
        self._asthma.update([n for n in self.nodes() if n in self._smokes and flip(COMORBID)])
        
        # R :: asthma(X) <- asthma_risk(X)
        self._asthma.update([n for n in self.nodes() if flip(self._asthma_risk[n])])