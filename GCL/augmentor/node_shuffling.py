from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import permute


class NodeShuffling(Augmentor):
    def __init__(self):
        """
        Shuffle the nodes of the graph.
        """
        super(NodeShuffling, self).__init__()

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        g.x = permute(g.x)
        return g

    def dgl_augment(self, g: DGLGraph):
        g = g.clone()
        g.ndata['x'] = permute(g.ndata['x'])
        return g
