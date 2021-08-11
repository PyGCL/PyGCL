from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import random_walk_subgraph


class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = random_walk_subgraph(edge_index, edge_weights, batch_size=self.num_seeds, length=self.walk_length)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
