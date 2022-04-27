from typing import Callable, Union
import torch
from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentors.functional import get_feature_weights, drop_edge_by_weight, drop_feature_by_weight
from functools import lru_cache


class AdaptiveFeatureMasking(Augmentor):
    def __init__(self, pe: float, sparse_feat: bool, weights_op: Callable[[Union[PyGGraph, DGLGraph]], torch.Tensor]):
        """This augmentor adaptively removes edges from the graph.

        Args:
            pe (float): Probability of edge removal.
            sparse_feat (bool): Whether to use sparse features.
            weights_op (Callable[[Union[PyGGraph, DGLGraph]], torch.Tensor]): Function that computes edge weights.
        """
        super(AdaptiveFeatureMasking, self).__init__()
        self.pe = pe
        self.sparse_feat = sparse_feat
        self.op = weights_op

    def compute_weights(self, g: PyGGraph):
        edge_weights = self.op(g)[1]
        return get_feature_weights(g.x, edge_weights, self.sparse_feat)

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        feat_weights = self.compute_weights(g)
        g.x = drop_feature_by_weight(g.x, feat_weights, self.pe)

        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError(f"DGLGraph is not supported by {self.__class__.__name__}")
