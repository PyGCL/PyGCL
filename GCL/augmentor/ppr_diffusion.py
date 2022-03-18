from GCL.augmentor.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentor.functional import compute_ppr


class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        """
        Run PPR diffusion on the graph.
        Args:
            alpha: The probability of returning to the original node.
            eps: Epsilon for sparsification.
            use_cache: Whether to use cache.
            add_self_loop: Whether to add self loop.
        """
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def pyg_augment(self, g: PyGGraph):
        if self._cache is not None and self.use_cache:
            return self._cache
        g = g.clone()
        edge_index, edge_weights = compute_ppr(
            g.edge_index, g.edge_attr,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        g.edge_index = edge_index
        g.edge_attr = edge_weights
        self._cache = g
        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError
