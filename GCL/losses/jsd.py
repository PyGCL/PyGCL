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

    neg_similarity = similarity * neg_mask
    E_neg = (F.softplus(- neg_similarity) + neg_similarity - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos


def debiased_jsd_loss(anchor, sample, pos_mask, neg_mask=None,
                      discriminator=lambda x, y: x @ y.t(), tau_plus=0.1, *args, **kwargs):
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


def hardness_jsd_loss(anchor, sample, pos_mask, neg_mask=None,
                      discriminator=lambda x, y: x @ y.t(), tau_plus=0.1, beta=0.05, *args, **kwargs):
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
