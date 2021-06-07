from GCL.augmentations.GraphAug import Graph, GraphAug
from GCL.augmentations.functional import add_edge


class EdgeAdding(GraphAug):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unapply()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
