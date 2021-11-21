from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import random_walk_subgraph


class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()

        g.edge_index, g.edge_attr = random_walk_subgraph(g.edge_index, g.edge_attr, batch_size=self.num_seeds, length=self.walk_length)

        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError
