import torch

from GCL.losses import Loss
from GCL.models import get_sampler


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask


class SingleBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(SingleBranchContrast, self).__init__()
        assert mode == 'G2L'  # only global-local pairs allowed in single-branch contrastive learning
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, hn=None, extra_pos_mask=None, extra_neg_mask=None):
        if batch is None:  # for single-graph datasets
            assert hn is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, neg_sample=hn)
        else:  # for multi-graph datasets
            assert batch is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, batch=batch)

        pos_mask, neg_mask = add_extra_mask(pos_mask, neg_mask, extra_pos_mask, extra_neg_mask)
        loss = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
        return loss


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
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

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5


class BootstrapContrast(torch.nn.Module):
    def __init__(self, loss, mode='L2L'):
        super(BootstrapContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=False)

    def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
                g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
                batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=h1_target, sample=h2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=h2_target, sample=h1_pred)
        elif self.mode == 'G2G':
            assert all(v is not None for v in [g1_pred, g2_pred, g1_target, g2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=g2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=g1_pred)
        else:
            assert all(v is not None for v in [h1_pred, h2_pred, g1_target, g2_target])
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                pos_mask1 = pos_mask2 = torch.ones([1, h1_pred.shape[0]], device=h1_pred.device)
                anchor1, sample1 = g1_target, h2_pred
                anchor2, sample2 = g2_target, h1_pred
            else:
                anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=h2_pred, batch=batch)
                anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=h1_pred, batch=batch)

        pos_mask1, _ = add_extra_mask(pos_mask1, extra_pos_mask=extra_pos_mask)
        pos_mask2, _ = add_extra_mask(pos_mask2, extra_pos_mask=extra_pos_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2)

        return (l1 + l2) * 0.5


class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        return (l1 + l2) * 0.5
