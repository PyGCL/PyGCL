import math
import torch
import dgl


def add_edge(g: dgl.DGLGraph, ratio: float) -> dgl.DGLGraph:
    g = g.clone()
    num_edges = g.num_edges()
    num_nodes = g.num_nodes()

    num_add = math.floor(num_edges * ratio)
    added_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(g.device)
    u_new, v_new = added_edge_index

    g.add_edges(u_new, v_new)
    return dgl.to_simple(g)
