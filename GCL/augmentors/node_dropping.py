from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_node


class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
