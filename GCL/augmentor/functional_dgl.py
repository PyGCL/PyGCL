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


def drop_edge(g: dgl.DGLGraph, drop_prob: float) -> dgl.DGLGraph:
    assert 0 <= drop_prob <= 1, 'Dropping probability should be between 0 and 1.'
    g = g.clone()

    mask = torch.zeros((g.num_edges(),), dtype=torch.float32).to(g.device)
    torch.fill_(mask, drop_prob)
    mask = torch.bernoulli(mask).to(torch.bool)

    remove_eids = g.edges(form='eid')[mask]
    g.remove_edges(eids=remove_eids)

    return g


def drop_node(g: dgl.DGLGraph, drop_prob: float) -> dgl.DGLGraph:
    g = g.clone()
    mask = torch.zeros((g.num_nodes(),), dtype=torch.float32).to(g.device)
    torch.fill_(mask, drop_prob)
    mask = torch.bernoulli(mask)

    remove_nids = torch.nonzero(mask).view(-1)
    g.remove_edges(remove_nids)

    return g
