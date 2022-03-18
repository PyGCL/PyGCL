import torch
import dgl

from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index

import GCL.augmentors.functional as F


def random_pyg_graph(num_nodes: int, num_edges: int, feature_dim: int = 256) -> Data:
    edge_index = torch.randint(0, num_nodes - 1, size=(2, num_edges))
    edge_index = sort_edge_index(edge_index)
    edge_index = F.coalesce_edge_index(edge_index)[0]

    x = torch.randn((num_nodes, feature_dim), dtype=torch.float32)

    return Data(edge_index=edge_index, num_nodes=num_nodes, x=x)


def random_dgl_graph(num_nodes: int, num_edges: int, feature_dim: int = 256) -> dgl.DGLGraph:
    edge_index = torch.randint(0, num_nodes - 1, size=(2, num_edges))
    u, v = edge_index
    g = dgl.graph((u, v))
    g = dgl.to_simple(g)

    x = torch.randn((g.num_nodes(), feature_dim), dtype=torch.float32)
    g.ndata['x'] = x

    return g

