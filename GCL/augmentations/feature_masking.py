from GCL.augmentations.GraphAug import Graph, GraphAug
from GCL.augmentations.functional import drop_feature


class FeatureMasking(GraphAug):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unapply()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
