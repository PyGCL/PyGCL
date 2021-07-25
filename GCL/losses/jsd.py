import torch

import numpy as np
import torch.nn.functional as F


def jsd_loss(anchor, sample, discriminator, pos_mask, neg_mask=None, *args, **kwargs):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(anchor, sample)

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    E_pos /= num_pos

    neg_similarity = similarity * neg_mask
    E_neg = (F.softplus(- neg_similarity) + neg_similarity - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos


class JSDLoss(torch.nn.Module):
    def __init__(self, discriminator, sampler, include_intraview_negs=True):
        super(JSDLoss, self).__init__()
        self.discriminator = discriminator
        self.sampler = sampler
        self.include_intraview_negs = include_intraview_negs

    def forward(self, h1, h2, g1=None, g2=None, batch=None, h3=None, h4=None, *args, **kwargs):
        if batch is None:
            if h3 is None and h4 is None:  # same-scale contrasting
                anchor, sample, pos_mask, neg_mask = self.sampler(anchor=h1, sample=h2)
                return jsd_loss(anchor, sample, discriminator=self.discriminator,
                                pos_mask=pos_mask, neg_mask=neg_mask, *args, **kwargs)
            else:  # global to local, only one graph
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
        else:  # global to local, multiple graphs
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        l1 = jsd_loss(anchor1, sample1, self.discriminator, pos_mask=pos_mask1, neg_mask=neg_mask1, *args, **kwargs)
        l2 = jsd_loss(anchor2, sample2, self.discriminator, pos_mask=pos_mask2, neg_mask=neg_mask2, *args, **kwargs)
        return l1 + l2
