import numpy as np
import torch.nn.functional as F


def jsd_loss(anchor, sample, pos_mask, neg_mask=None,
             discriminator=lambda x, y: x @ y.t(), *args, **kwargs):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(anchor, sample)

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    E_pos /= num_pos

    neg_sim = similarity * neg_mask
    E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos


def debiased_jsd_loss(anchor, sample, pos_mask, neg_mask=None,
                      discriminator=lambda x, y: x @ y.t(), tau_plus=0.1, *args, **kwargs):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(anchor, sample)

    pos_sim = similarity * pos_mask
    E_pos = np.log(2) - F.softplus(- pos_sim) - (tau_plus / (1 - tau_plus)) * (F.softplus(-pos_sim) + pos_sim)
    E_pos = E_pos.sum() / num_pos

    neg_sim = similarity * neg_mask
    E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)) / (1 - tau_plus)
    E_neg = E_neg.sum() / num_neg

    return E_neg - E_pos


def hardness_jsd_loss(anchor, sample, pos_mask, neg_mask=None,
                      discriminator=lambda x, y: x @ y.t(), tau_plus=0.1, beta=0.05, *args, **kwargs):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(anchor, sample)

    pos_sim = similarity * pos_mask
    E_pos = np.log(2) - F.softplus(- pos_sim) - (tau_plus / (1 - tau_plus)) * (F.softplus(-pos_sim) + pos_sim)
    E_pos = E_pos.sum() / num_pos

    neg_sim = similarity * neg_mask
    E_neg = F.softplus(- neg_sim) + neg_sim

    reweight = -2 * neg_sim / max(neg_sim.max(), neg_sim.min().abs())
    reweight = (beta * reweight).exp()
    reweight = reweight / reweight.mean(dim=1, keepdim=True)

    E_neg = (reweight * E_neg) / (1 - tau_plus) - np.log(2)
    E_neg = E_neg.sum() / num_neg

    return E_neg - E_pos
