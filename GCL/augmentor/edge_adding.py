from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import add_edge
import GCL.augmentor.functional_dgl as F_dgl


class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        g.edge_index = add_edge(g.edge_index, ratio=self.pe)
        return g

    def dgl_augment(self, g: DGLGraph):
        return F_dgl.add_edge(g, self.pe)
