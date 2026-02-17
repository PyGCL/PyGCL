from __future__ import annotations

from GCL.augmentors.augmentor import PyGGraph, Augmentor
from GCL.augmentors.functional import add_edge


class EdgeAdding(Augmentor):
    """Augment the graph by adding edges."""

    def __init__(self, pe: float):
        """
        Args:
            pe: Probability of adding an edge.
        """
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        g.edge_index = add_edge(g.edge_index, ratio=self.pe)
        return g

    def dgl_augment(self, g):
        import GCL.augmentors.functional_dgl as F_dgl
        return F_dgl.add_edge(g, self.pe)
