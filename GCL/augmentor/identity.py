from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor


class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def pyg_augment(self, g: PyGGraph):
        return g

    def dgl_augment(self, g: DGLGraph):
        return g
