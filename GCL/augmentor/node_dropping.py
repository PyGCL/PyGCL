from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import drop_node
import GCL.augmentor.functional_dgl as F_dgl


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

    def dgl_augment(self, g: DGLGraph):
        return F_dgl.drop_node(g, self.pn)
