import torch

from GCL.losses import Loss
from GCL.samplers import Sampler, CrossScaleSampler, SameScaleSampler


def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    elif mode == 'G2L':
        return CrossScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')


class SingleBranchContrastModel(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, *args, **kwargs):
        super(SingleBranchContrastModel, self).__init__()
        assert mode == 'G2L'  # only global-local pairs allowed in single-branch contrastive learning
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, hn=None):
        if batch is None:  # for single-graph datasets
            assert hn is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, neg_sample=hn)
        else:  # for multi-graph datasets
            assert batch is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, batch=batch)

        loss = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
        return loss


class DualBranchContrastModel(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, *args, **kwargs):
        super(DualBranchContrastModel, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
