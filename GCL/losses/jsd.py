import numpy as np
import torch.nn.functional as F


def jsd_loss(z1, z2, discriminator, pos_mask, neg_mask=None):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(z1, z2)

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    E_pos /= num_pos
    neg_similarity = similarity * neg_mask
    E_neg = (F.softplus(- neg_similarity) + neg_similarity - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos
