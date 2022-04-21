import torch

from abc import ABC, abstractmethod
from torch_scatter import scatter
from GCL.losses import ContrastInstance


def compute_supervised_masks(data):
    # compute extra pos and neg masks for semi-supervised learning
    extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    # construct extra supervision signals for only training samples
    extra_pos_mask[~data.train_mask][:, ~data.train_mask] = False
    extra_pos_mask.fill_diagonal_(False)
    # pos_mask: [N, 2N] for both inter-view and intra-view samples
    extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to('cuda')
    # fill inter-view positives only
    # pos_mask for intra-view samples should have zeros in diagonal
    extra_pos_mask.fill_diagonal_(True)

    extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    extra_neg_mask[~data.train_mask][:, ~data.train_mask] = True
    extra_neg_mask.fill_diagonal_(False)
    extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to('cuda')
    return extra_pos_mask, extra_neg_mask


class DefaultSampler:
    def __init__(self):
        pass

    def __call__(self, anchor: torch.Tensor, sample: torch.Tensor, *args, **kwargs) -> ContrastInstance:
        return self.sample(anchor, sample)

    @staticmethod
    def sample(anchor: torch.Tensor, sample: torch.Tensor) -> ContrastInstance:
        assert anchor.size(0) == sample.size(0) and anchor.size(1) == sample.size(1)
        return ContrastInstance(anchor=anchor, sample=sample, pos_mask=None, neg_mask=None)


class DenseSampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(
            self, anchor: torch.Tensor, sample: torch.Tensor,
            extra_pos_mask: torch.Tensor=None, extra_neg_mask: torch.Tensor=None,
            *args, **kwargs) -> ContrastInstance:
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(ret)
        self.combine_extra_masks(ret, extra_pos_mask, extra_neg_mask)
        return ret

    @abstractmethod
    def sample(self, anchor: torch.Tensor, sample: torch.Tensor, *args, **kwargs) -> ContrastInstance:
        raise NotImplementedError

    @staticmethod
    def add_intraview_negs(contrast_instance: ContrastInstance) -> ContrastInstance:
        anchor, sample, pos_mask, neg_mask = contrast_instance.unpack()
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)                     # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)     # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)     # M * (M+N)
        return ContrastInstance(anchor=anchor, sample=new_sample, pos_mask=new_pos_mask, neg_mask=new_neg_mask)

    @staticmethod
    def combine_extra_masks(
            contrast_instance: ContrastInstance,
            extra_pos_mask: torch.Tensor, extra_neg_mask: torch.Tensor):
        if extra_pos_mask is not None:
            contrast_instance.pos_mask = torch.bitwise_or(
                contrast_instance.pos_mask.bool(), extra_pos_mask.bool()).float()
        if extra_neg_mask is not None:
            contrast_instance.neg_mask = torch.bitwise_and(
                contrast_instance.neg_mask.bool(), extra_neg_mask.bool()).float()


class SameScaleDenseSampler(DenseSampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleDenseSampler, self).__init__(*args, **kwargs)

    def sample(
            self, anchor: torch.Tensor, sample: torch.Tensor,
            pos_mask: torch.Tensor=None, neg_mask: torch.Tensor=None,
            *args, **kwargs) -> ContrastInstance:
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        if pos_mask is None:
            pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        if neg_mask is None:
            neg_mask = 1. - pos_mask
        return ContrastInstance(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask)


class CrossScaleDenseSampler(DenseSampler):
    def __init__(self, *args, **kwargs):
        super(CrossScaleDenseSampler, self).__init__(*args, **kwargs)

    def sample(
            self, anchor: torch.Tensor, sample: torch.Tensor,
            batch: torch.Tensor = None, neg_sample: torch.Tensor = None,
            use_gpu: bool = True, *args, **kwargs) -> ContrastInstance:
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]   # N
        device = sample.device

        if neg_sample is not None:
            assert num_graphs == 1  # only one graph, explicit negative samples are needed
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)     # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)         # 2N * K
        else:
            assert batch is not None
            if use_gpu:
                ones = torch.eye(num_nodes, dtype=torch.float32, device=device)     # N * N
                pos_mask = scatter(ones, batch, dim=0, reduce='sum')                # M * N
            else:
                pos_mask = torch.zeros((num_graphs, num_nodes), dtype=torch.float32).to(device)
                for node_idx, graph_idx in enumerate(batch):
                    pos_mask[graph_idx][node_idx] = 1.                              # M * N

        neg_mask = 1. - pos_mask
        return ContrastInstance(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask)


def get_dense_sampler(mode: str, intraview_negs: bool) -> DenseSampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleDenseSampler(intraview_negs=intraview_negs)
    elif mode == 'G2L':
        return CrossScaleDenseSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'Unsupported mode: {mode}')
