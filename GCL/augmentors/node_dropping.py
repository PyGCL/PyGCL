from __future__ import annotations

from GCL.augmentors.augmentor import PyGGraph, Augmentor
from GCL.augmentors.functional import drop_node


class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        """
        This augmentor drops nodes from the graph with probability pn.
        Args:
            pn: Probability of dropping a node.
        """
        super(NodeDropping, self).__init__()
        self.pn = pn

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        edge_index, edge_weights = drop_node(g.edge_index, g.edge_attr, keep_prob=1. - self.pn)
        g.edge_index = edge_index
        g.edge_attr = edge_weights
        return g

    def dgl_augment(self, g):
        import GCL.augmentors.functional_dgl as F_dgl
        return F_dgl.drop_node(g, self.pn)
