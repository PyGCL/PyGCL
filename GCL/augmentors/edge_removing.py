from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentors.functional import dropout_adj
import GCL.augmentors.functional_dgl as F_dgl


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        """This augmentor removes edges from the graph.

        Args:
            pe (float): Probability of edge removal.
        """
        self.pe = pe
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        edge_index, edge_weights = dropout_adj(g.edge_index, edge_attr=g.edge_attr, p=self.pe)
        g.edge_index = edge_index
        g.edge_attr = edge_weights
        return g

    def dgl_augment(self, g: DGLGraph):
        return F_dgl.drop_edge(g, drop_prob=self.pe)
