from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import dropout_feature


class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        g.x = dropout_feature(g.x, self.pf)
        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError
