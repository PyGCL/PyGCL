from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, Callable

import torch


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()

    def __rshift__(self, other: Augmentor):
        return CompositionalAugmentor(self, other)


class CompositionalAugmentor(Augmentor):
    def __init__(self, g1: Augmentor, g2: Augmentor):
        super(CompositionalAugmentor, self).__init__()
        self.g1 = g1
        self.g2 = g2

    def augment(self, g: Graph) -> Graph:
        return self.g2.augment(self.g1.augment(g))
