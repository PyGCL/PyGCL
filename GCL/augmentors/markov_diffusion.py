from GCL.augmentors.augmentor import PyGGraph, DGLGraph, Augmentor
from GCL.augmentors.functional import compute_markov_diffusion


class MarkovDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.05, order: int = 16, sp_eps: float = 1e-4, use_cache: bool = True,
                 add_self_loop: bool = True):
        """
        The augmentor performs Markov diffusion on the graph.
        Args:
            alpha: The probability of self-loop.
            order: The order of Markov diffusion.
            sp_eps: The epsilon sparsifying the diffusion matrix.
            use_cache: Whether to use cache.
            add_self_loop: Whether to add self-loop.
        """
        super(MarkovDiffusion, self).__init__()
        self.alpha = alpha
        self.order = order
        self.sp_eps = sp_eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def pyg_augment(self, g: PyGGraph) -> PyGGraph:
        if self._cache is not None and self.use_cache:
            return self._cache
        g = g.clone()
        edge_index, edge_weights = compute_markov_diffusion(
            g.edge_index, g.edge_attr,
            alpha=self.alpha, degree=self.order,
            sp_eps=self.sp_eps, add_self_loop=self.add_self_loop
        )
        g.edge_index = edge_index
        g.edge_attr = edge_weights
        self._cache = g
        return g

    def dgl_augment(self, g: DGLGraph):
        raise NotImplementedError
