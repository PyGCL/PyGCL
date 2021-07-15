from GCL.augmentations.GraphAug import Graph, GraphAug
from GCL.augmentations.functional import dropout_feature


class EdgeAttrDropout(GraphAug):
    def __init__(self, pf: float):
        super(EdgeAttrDropout, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unapply()
        if edge_weights is not None:
            edge_weights = dropout_feature(edge_weights, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
