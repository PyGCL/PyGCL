import dgl
import torch
import torch_geometric.transforms as T

from functools import partial
from torch_geometric.data import Data
from GCL.augmentor import PyGAugmentor, DGLAugmentor, Compose


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
