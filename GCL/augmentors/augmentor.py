from __future__ import annotations

import torch
from dgl import DGLGraph
from abc import ABC, abstractmethod
from typing import Optional, List, Callable
from torch_geometric.data import Data as PyGGraph


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    def augment(self, g: Optional[DGLGraph, PyGGraph]):
        if isinstance(g, DGLGraph):
            return self.dgl_augment(g)
        elif isinstance(g, PyGGraph):
            return self.pyg_augment(g)

    @abstractmethod
    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError

    @abstractmethod
    def pyg_augment(self, g: PyGGraph):
        raise NotImplementedError

    def __call__(self, g: Optional[DGLGraph, PyGGraph]):
        assert isinstance(g, DGLGraph) or isinstance(g, PyGGraph)
        return self.augment(g)


class PyGAugmentor(Augmentor):
    def __init__(self, augmentor: Callable):
        super(PyGAugmentor, self).__init__()
        self.augmentor = augmentor

    def pyg_augment(self, g: PyGGraph):
        return self.augmentor(g)

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError


class DGLAugmentor(Augmentor):
    def __init__(self, augmentor: Callable):
        super(DGLAugmentor, self).__init__()
        self.augmentor = augmentor

    def pyg_augment(self, g: PyGGraph):
        raise NotImplementedError

    def dgl_augment(self, g: DGLGraph):
        return self.augmentor(g)


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g):
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g):
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g


if __name__ == '__main__':
    import dgl
    import os.path as osp
    import torch_geometric.transforms as T
    from functools import partial
    from torch_geometric.datasets import Planetoid

    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]

    aug1 = PyGAugmentor(T.Constant(1))
    print(aug1(data))

    g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    f = partial(dgl.add_edges, u=torch.tensor([1, 3]), v=torch.tensor([0, 1]))
    aug2 = DGLAugmentor(f)
    print(aug2(g))
