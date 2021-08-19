from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import compute_markov_diffusion


class MarkovDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.05, order: int = 16, sp_eps: float = 1e-4, use_cache: bool = True,
                 add_self_loop: bool = True):
        super(MarkovDiffusion, self).__init__()
        self.alpha = alpha
        self.order = order
        self.sp_eps = sp_eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_markov_diffusion(
            edge_index, edge_weights,
            alpha=self.alpha, degree=self.order,
            sp_eps=self.sp_eps, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res
