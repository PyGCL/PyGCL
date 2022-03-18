from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentors.functional import dropout_feature


class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        """
        This augmentor drops out feature dimensions of the graph with probability pf.
        Args:
            pf: probability of dropping out a feature dimension.
        """
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        g.x = dropout_feature(g.x, self.pf)
        return g

    def dgl_augment(self, g: DGLGraph):
        g = g.clone()
        new_x = dropout_feature(g.ndata['x'], self.pf)
        g.ndata['x'] = new_x

        return g
