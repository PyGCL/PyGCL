from GCL.augmentations.GraphAug import Graph, GraphAug
from GCL.augmentations.functional import permute


class NodeShuffling(GraphAug):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unapply()

        x = permute(x)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
