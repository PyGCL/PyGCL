import torch

from GCL.samplers import Sampler, CrossScaleSampler, SameScaleSampler
from GCL.losses import Loss

__all__ = ['ContrastModel']


def get_sampler(mode: str) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler()
    elif mode == 'G2L':
        return CrossScaleSampler()
    else:
        raise RuntimeError(f'unsupported mode: {mode}')


class ContrastModel(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, *args, **kwargs):
        super(ContrastModel, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode)
        self.kwargs = kwargs

    def forward(self, h1, h2, g1=None, g2=None, batch=None, h3=None, h4=None):
        if self.mode == 'L2L':
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
