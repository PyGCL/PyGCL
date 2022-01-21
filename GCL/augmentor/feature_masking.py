from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import drop_feature


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        g.x = drop_feature(g.x, self.pf)
        return g

    def dgl_augment(self, g: DGLGraph):
        g = g.clone()
        new_x = drop_feature(g.ndata['x'], self.pf)
        g.ndata['x'] = new_x

        return g
