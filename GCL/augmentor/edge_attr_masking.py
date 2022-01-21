from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import drop_feature


class EdgeAttrMasking(Augmentor):
    def __init__(self, pf: float):
        super(EdgeAttrMasking, self).__init__()
        self.pf = pf

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        if g.edge_attr is not None:
            g.edge_attr = drop_feature(g.edge_attr, self.pf)
        return g

    def dgl_augment(self, g: DGLGraph):
        g = g.clone()

        edata_keys = list(g.edata.keys())

        for edata_key in edata_keys:
            g.edata[edata_key] = drop_feature(g.edata[edata_key], self.pf)

        return g
