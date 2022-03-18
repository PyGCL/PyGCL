import dgl
import torch
import torch_geometric.transforms as T

from functools import partial
from torch_geometric.data import Data
from GCL.augmentors import PyGAugmentor, DGLAugmentor, Compose
from GCL.augmentors import \
    EdgeAdding, EdgeRemoving, EdgeAttrMasking, \
    FeatureDropout, FeatureMasking, \
    NodeDropping, NodeShuffling

import testing_utils


def test_pygaugmentor():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, num_nodes=3)
    aug = PyGAugmentor(T.Constant(1))
    data = aug(data)
    assert data.x.tolist() == [[1.0], [1.0], [1.0]]

    data = aug(data)
    data.x.tolist()
    assert data.x.tolist() == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]


def test_dglaugmentor():
    g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    f = partial(dgl.add_edges, u=torch.tensor([1, 3]), v=torch.tensor([0, 1]))
    aug = DGLAugmentor(f)
    g = aug(g)
    assert g.num_nodes() == 4
    assert g.num_edges() == 4
    assert g.edges()[0].tolist() == [0, 1, 1, 3]
    assert g.edges()[1].tolist() == [1, 2, 0, 1]


def test_composing_augmentor():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, num_nodes=3)
    aug1 = PyGAugmentor(T.Constant(1))
    aug2 = PyGAugmentor(T.Constant(2))
    comp_aug = Compose([aug1, aug2])
    data = comp_aug(data)
    assert data.x.tolist() == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]


def test_edge_adding():
    aug = EdgeAdding(pe=0.8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, num_nodes=3)
    aug_data = aug(data)
    assert aug_data.edge_index.shape[0] == 2
    assert aug_data.edge_index.shape[1] >= data.edge_index.shape[1]


def test_edge_adding_dgl():
    aug = EdgeAdding(pe=0.8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    u, v = edge_index
    g: dgl.DGLGraph = dgl.graph((u, v), num_nodes=3)
    aug_g = aug(g)

    assert aug_g.num_edges() >= g.num_edges()


def test_edge_removing():
    aug = EdgeRemoving(pe=0.5)
    data = testing_utils.random_pyg_graph(num_nodes=3, num_edges=4)
    aug_data = aug(data)
    assert aug_data.edge_index.shape[0] == 2
    assert aug_data.edge_index.shape[1] <= data.edge_index.shape[1]


def test_edge_removing_dgl():
    aug = EdgeRemoving(pe=0.5)
    g = testing_utils.random_dgl_graph(num_nodes=3, num_edges=4)
    aug_g = aug(g)
    assert aug_g.num_edges() <= g.num_edges()


def test_feature_dropout():
    aug = FeatureDropout(pf=0.5)
    g = testing_utils.random_pyg_graph(num_nodes=3, num_edges=5, feature_dim=10)
    aug_g = aug(g)

    assert aug_g.x.shape == g.x.shape
    # assert aug_g.x.abs().mean().item() < g.x.abs().mean().item()


def test_feature_dropout_dgl():
    aug = FeatureDropout(pf=0.5)
    g = testing_utils.random_dgl_graph(num_nodes=3, num_edges=5, feature_dim=10)
    aug_g = aug(g)

    assert aug_g.ndata['x'].shape == g.ndata['x'].shape
    # assert aug_g.x.abs().mean().item() < g.x.abs().mean().item()


def test_feature_masking():
    aug = FeatureMasking(pf=0.5)
    g = testing_utils.random_pyg_graph(num_nodes=3, num_edges=5, feature_dim=10)
    aug_g = aug(g)

    assert aug_g.x.shape == g.x.shape
    # assert aug_g.x.abs().mean().item() < g.x.abs().mean().item()


def test_feature_masking_dgl():
    aug = FeatureMasking(pf=0.5)
    g = testing_utils.random_dgl_graph(num_nodes=3, num_edges=5, feature_dim=10)
    aug_g = aug(g)

    assert aug_g.ndata['x'].shape == g.ndata['x'].shape
    # assert aug_g.x.abs().mean().item() < g.x.abs().mean().item()


def test_node_dropping():
    aug = NodeDropping(pn=0.5)
    g = testing_utils.random_pyg_graph(num_nodes=100, num_edges=500, feature_dim=32)
    aug_g = aug(g)

    assert aug_g.edge_index.max().item() + 1 <= g.edge_index.max().item() + 1


def test_node_dropping_dgl():
    aug = NodeDropping(pn=0.5)
    g = testing_utils.random_dgl_graph(num_nodes=100, num_edges=500, feature_dim=32)
    aug_g = aug(g)

    assert aug_g.num_nodes() <= g.num_nodes()


def test_node_shuffling():
    aug = NodeShuffling()
    g = testing_utils.random_pyg_graph(num_nodes=100, num_edges=1000, feature_dim=128)
    aug_g = aug(g)

    assert (g.x.mean().item() - aug_g.x.mean().item()) < 1e-8


def test_node_shuffling_dgl():
    aug = NodeShuffling()
    g = testing_utils.random_dgl_graph(num_nodes=100, num_edges=1000, feature_dim=128)
    aug_g = aug(g)

    assert (g.ndata['x'].mean().item() - aug_g.ndata['x'].mean().item()) < 1e-8
