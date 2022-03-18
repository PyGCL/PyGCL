from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor


class Identity(Augmentor):
    def __init__(self):
        """
        Identity augmentor.
        """
        super(Identity, self).__init__()

    def pyg_augment(self, g: PyGGraph):
        return g

    def dgl_augment(self, g: DGLGraph):
        return g
