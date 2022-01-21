from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import dropout_adj
import GCL.augmentor.functional_dgl as F_dgl


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
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
