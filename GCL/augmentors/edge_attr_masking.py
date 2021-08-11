from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature


class EdgeAttrMasking(Augmentor):
    def __init__(self, pf: float):
        super(EdgeAttrMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        if edge_weights is not None:
            edge_weights = drop_feature(edge_weights, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
