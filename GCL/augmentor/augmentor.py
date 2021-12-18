from __future__ import annotations

import torch
from dgl import DGLGraph
from abc import ABC, abstractmethod
from typing import Union, List, Callable
from torch_geometric.data import Data as PyGGraph


class Augmentor(ABC):
    """Base class for graph augmentor."""
    def __init__(self):
        pass

    def augment(self, g: Union[DGLGraph, PyGGraph]):
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

    def __call__(self, g: Union[DGLGraph, PyGGraph]):
        assert isinstance(g, DGLGraph) or isinstance(g, PyGGraph)
        return self.augment(g)


class PyGAugmentor(Augmentor):
    def __init__(self, augmentor: Callable):
        super(PyGAugmentor, self).__init__()
        self.augmentor = augmentor

    def pyg_augment(self, g: PyGGraph):
        g_new = g.clone()
        return self.augmentor(g_new)

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

    def pyg_augment(self, g: PyGGraph):
        raise NotImplementedError

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError


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

    def pyg_augment(self, g: PyGGraph):
        raise NotImplementedError

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError
