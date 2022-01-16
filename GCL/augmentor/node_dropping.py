from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import drop_node


class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        edge_index, edge_weights = drop_node(g.edge_index, g.edge_attr, keep_prob=1. - self.pn)
        g.edge_index = edge_index
        g.edge_attr = edge_weights
        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError
