import dgl
import torch
from dgl import DGLGraph
from torch_geometric.data import Data as PyGGraph


def from_pyg_to_dgl(pyg_graph: PyGGraph) -> DGLGraph:
    """Convert PyG graph to DGL graph.

    Args:
        pyg_graph (PyGGraph): PyG graph to be converted.

    Returns:
        DGLGraph: DGL graph.
    """
    g = dgl.DGLGraph()
    g.add_nodes(pyg_graph.num_nodes)
    g.ndata['feat'] = pyg_graph.x
    g.edata['feat'] = pyg_graph.edge_attr
    g.add_edges(pyg_graph.edge_index[0], pyg_graph.edge_index[1])
    return g


def from_dgl_to_pyg(dgl_graph: DGLGraph) -> PyGGraph:
    """Convert DGL graph to PyG graph.

    Args:
        dgl_graph (DGLGraph): DGL graph to be converted.

    Returns:
        PyGGraph: PyG graph.
    """
    u, v = dgl_graph.edges()
    edge_index = torch.stack([u, v], dim=0)

    def guess_feature_field(dgl_graph):
        possible_names = ["x", "feat"]
        for name in possible_names:
            if name in dgl_graph.ndata:
                return name
        return None

    def guess_label_field(dgl_graph):
        possible_names = ["y", "label"]
        for name in possible_names:
            if name in dgl_graph.ndata:
                return name
        return None

    def guess_edge_feature_field(dgl_graph):
        possible_names = ["edge_attr", "feat"]
        for name in possible_names:
            if name in dgl_graph.edata:
                return name
        return None

    x_field = guess_feature_field(dgl_graph)
    y_field = guess_label_field(dgl_graph)
    edge_attr_field = guess_edge_feature_field(dgl_graph)

    pyg_graph = PyGGraph(
        x=dgl_graph.ndata[x_field] if x_field is not None else None,
        edge_index=edge_index,
        edge_attr=dgl_graph.edata[edge_attr_field] if edge_attr_field is not None else None,
        y=dgl_graph.ndata[y_field] if y_field is not None else None,
    )

    return pyg_graph
