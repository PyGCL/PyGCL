"""Augmentors."""
from __future__ import annotations

import torch
from dgl import DGLGraph
from abc import ABC, abstractmethod
from typing import Union, List, Callable
from torch_geometric.data import Data as PyGGraph


class Augmentor(ABC):
    """Augmentor."""
    def __init__(self):
        """Initialize the augmentor."""
        pass

    def augment(self, g: Union[DGLGraph, PyGGraph]):
        """Augment the graph. It dynamically calls the appropriate augmentor.

        Args:
            g: Graph to be augmented.

        Returns:
            Augmented graph.
        """
        if isinstance(g, DGLGraph):
            return self.dgl_augment(g)
        elif isinstance(g, PyGGraph):
            return self.pyg_augment(g)

    @abstractmethod
    def dgl_augment(self, g: DGLGraph):
        """Augment the DGL graph.

        Args:
            g: DGLGraph to be augmented.

        Returns:
            Augmented graph.
        """
        raise NotImplementedError

    @abstractmethod
    def pyg_augment(self, g: PyGGraph):
        """Augment the PyG graph.

        Args:
            g: PyGGraph to be augmented.

        Returns:
            Augmented graph.
        """
        raise NotImplementedError

    def __call__(self, g: Union[DGLGraph, PyGGraph]):
        """
        Call the augmentor.
        Args:
            g: Graph to be augmented.

        Returns:
            Augmented graph.
        """
        assert isinstance(g, DGLGraph) or isinstance(g, PyGGraph)
        return self.augment(g)


class PyGAugmentor(Augmentor):
    """Augmentor wrapper for PyG graphs."""
    def __init__(self, augmentor: Callable):
        """Initialize the augmentor wrapper.

        Args:
            augmentor: Augmentor function.
        """
        super(PyGAugmentor, self).__init__()
        self.augmentor = augmentor

    def pyg_augment(self, g: PyGGraph):
        """Augment the PyG graph.

        Args:
            g: PyGGraph to be augmented.

        Returns:
            Augmented graph.
        """
        g_new = g.clone()
        return self.augmentor(g_new)

    def dgl_augment(self, g: DGLGraph):
        """
        Augment the DGL graph. It will raise an NotImplementedError.
        Args:
            g: DGLGraph to be augmented.
        """
        raise NotImplementedError


class DGLAugmentor(Augmentor):
    """Augmentor wrapper for DGL graphs."""
    def __init__(self, augmentor: Callable):
        """Initialize the augmentor wrapper.

        Args:
            augmentor: Augmentor function.
        """
        super(DGLAugmentor, self).__init__()
        self.augmentor = augmentor

    def pyg_augment(self, g: PyGGraph):
        """Augment the PyG graph. It will raise an NotImplementedError.

        Args:
            g: PyGGraph to be augmented.
        """
        raise NotImplementedError

    def dgl_augment(self, g: DGLGraph):
        """Augment the DGL graph.

        Args:
            g: DGLGraph to be augmented.

        Returns:
            Augmented graph.
        """
        return self.augmentor(g)


class Compose(Augmentor):
    """Compose multiple augmentors."""
    def __init__(self, augmentors: List[Augmentor]):
        """Initialize the augmentor.

        Args:
            augmentors: List of augmentors to be composed.
        """
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g):
        """Augment the graph.

        Args:
            g: Graph to be augmented.

        Returns:
            Augmented graph.
        """
        for aug in self.augmentors:
            g = aug.augment(g)
        return g

    def pyg_augment(self, g: PyGGraph):
        """Augment the PyG graph.

        Args:
            g: PyGGraph to be augmented.

        Returns:
            Augmented graph.
        """
        raise NotImplementedError

    def dgl_augment(self, g: DGLGraph):
        """Augment the DGL graph.

        Args:
            g: DGLGraph to be augmented.

        Returns:
            Augmented graph.
        """
        raise NotImplementedError


class RandomChoice(Augmentor):
    """Argument the graph with a random choice of augmentors."""
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        """Initialize the augmentor.

        Args:
            augmentors: List of augmentors to be composed.
            num_choices: Number of augmentors to be chosen.
        """
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g):
        """Augment the graph.
        It will choose `num_choices` augmentors randomly, and call them to augment the graph.

        Args:
            g: Graph to be augmented.

        Returns:
            Augmented graph.
        """
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
