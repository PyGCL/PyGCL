from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentors.functional import add_edge
import GCL.augmentors.functional_dgl as F_dgl


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

    def dgl_augment(self, g: DGLGraph):
        return F_dgl.add_edge(g, self.pe)
