import torch
import dgl

from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index

import GCL.augmentor.functional as F


def random_pyg_graph(num_nodes: int, num_edges: int) -> Data:
    edge_index = torch.randint(0, num_nodes - 1, size=(2, num_edges))
    edge_index = sort_edge_index(edge_index)[0]
    edge_index = F.coalesce_edge_index(edge_index)[0]

    return Data(edge_index=edge_index, num_nodes=num_nodes)


def random_dgl_graph(num_nodes: int, num_edges: int) -> dgl.DGLGraph:
    edge_index = torch.randint(0, num_nodes - 1, size=(2, num_edges))
    u, v = edge_index
    g = dgl.graph((u, v))
    g = dgl.to_simple(g)

    return g

