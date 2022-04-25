from typing import Callable, Union
import torch
from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentors.functional import get_feature_weights, drop_edge_by_weight, drop_feature_by_weight
import GCL.augmentors.functional as F_pyg
from functools import lru_cache


class WeightsOp:
    def __init__(self):
        super(WeightsOp, self).__init__()
        self._pinned = None
        self._pinned_weights = None

    def __call__(self, g: PyGGraph):
        if self._pinned is not None:
            return self._pinned_weights
        else:
            return self.compute(g)

    def compute(self, g: PyGGraph):
        raise NotImplementedError

    def pin_graph(self, g: PyGGraph):
        self._pinned = g
        self._pinned_weights = self.compute(g)



class DegreeWeights(WeightsOp):
    def __init__(self):
        super(DegreeWeights, self).__init__()

    @lru_cache(4)
    def compute(self, g: PyGGraph):
        return F_pyg.get_degree_weights(g)


class PageRankWeights(WeightsOp):
    def __init__(self, aggr: str = 'sink', k: int = 200):
        super(PageRankWeights, self).__init__()
        self.aggr = aggr
        self.k = k

    @lru_cache(4)
    def compute(self, g: PyGGraph):
        return F_pyg.get_pagerank_weights(g, self.aggr, self.k)


class EigenVectorWeights(WeightsOp):
    def __init__(self):
        super(EigenVectorWeights, self).__init__()

    @lru_cache(4)
    def compute(self, g: PyGGraph):
        return F_pyg.get_eigenvector_weights(g)


class AdaptiveEdgeRemoving(Augmentor):
    def __init__(self, pe: float, weights_op: Callable[[Union[PyGGraph, DGLGraph]], torch.Tensor]):
        """This augmentor adaptively removes edges from the graph.

        Args:
            pe (float): Probability of edge removal.
            sparse_feat (bool): Whether to use sparse features.
            weights_op (Callable[[Union[PyGGraph, DGLGraph]], torch.Tensor]): Function that computes edge weights.
        """
        super(AdaptiveEdgeRemoving, self).__init__()
        self.pe = pe
        self.op = weights_op

    def compute_weights(self, g: PyGGraph):
        return self.op(g)[0]

    def pyg_augment(self, g: PyGGraph):
        g = g.clone()
        edge_weights = self.compute_weights(g)
        g.edge_index = drop_edge_by_weight(g.edge_index, edge_weights, self.pe)

        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError(f"DGLGraph is not supported by {self.__class__.__name__}")
