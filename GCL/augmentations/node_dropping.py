import torch
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.utils import subgraph

from GCL.augmentations.GraphAug import Graph, GraphAug
from GCL.augmentations.functional import drop_node


class NodeDropping(GraphAug):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unapply()

        edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
