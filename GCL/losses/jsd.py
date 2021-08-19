import numpy as np
import torch.nn.functional as F

from .losses import Loss


class JSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t()):
        super(JSD, self).__init__()
        self.discriminator = discriminator

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
        E_pos /= num_pos

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
        E_neg /= num_neg

        return E_neg - E_pos


class DebiasedJSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t(), tau_plus=0.1):
        super(DebiasedJSD, self).__init__()
        self.discriminator = discriminator
        self.tau_plus = tau_plus

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        pos_sim = similarity * pos_mask
        E_pos = np.log(2) - F.softplus(- pos_sim)
        E_pos -= (self.tau_plus / (1 - self.tau_plus)) * (F.softplus(-pos_sim) + pos_sim)
        E_pos = E_pos.sum() / num_pos

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)) / (1 - self.tau_plus)
        E_neg = E_neg.sum() / num_neg

        return E_neg - E_pos


class HardnessJSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t(), tau_plus=0.1, beta=0.05):
        super(HardnessJSD, self).__init__()
        self.discriminator = discriminator
        self.tau_plus = tau_plus
        self.beta = beta

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        pos_sim = similarity * pos_mask
        E_pos = np.log(2) - F.softplus(- pos_sim)
        E_pos -= (self.tau_plus / (1 - self.tau_plus)) * (F.softplus(-pos_sim) + pos_sim)
        E_pos = E_pos.sum() / num_pos

        neg_sim = similarity * neg_mask
        E_neg = F.softplus(- neg_sim) + neg_sim

        reweight = -2 * neg_sim / max(neg_sim.max(), neg_sim.min().abs())
        reweight = (self.beta * reweight).exp()
        reweight /= reweight.mean(dim=1, keepdim=True)

        E_neg = (reweight * E_neg) / (1 - self.tau_plus) - np.log(2)
        E_neg = E_neg.sum() / num_neg

        return E_neg - E_pos
