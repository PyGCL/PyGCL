from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, Callable

import torch


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unapply(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class GraphAug(ABC):
    """Base class for graph augmentations."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, edge_index, edge_weight = self.augment(Graph(x, edge_index, edge_weight)).unapply()
        return x, edge_index, edge_weight

    @staticmethod
    def make_simple_graph(x: torch.FloatTensor, edge_index: torch.LongTensor) -> Graph:
        """Make a simple graph where edges are not weighted."""
        return Graph(x, edge_index, None)

    def __rshift__(self, other: 'GraphAug'):
        return ComposedGraphAug(self, other)


class ComposedGraphAug(GraphAug):
    def __init__(self, g1: GraphAug, g2: GraphAug):
        super(ComposedGraphAug, self).__init__()
        self.g1 = g1
        self.g2 = g2

    def augment(self, g: Graph) -> Graph:
        return self.g2.augment(self.g1.augment(g))
