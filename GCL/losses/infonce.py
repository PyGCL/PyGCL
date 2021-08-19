import torch
import numpy as np
import torch.nn.functional as F

from .losses import Loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCESP(Loss):
    """
    InfoNCE loss for single positive.
    """
    def __init__(self, tau):
        super(InfoNCESP, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        sim = f(_similarity(anchor, sample))  # anchor x sample
        assert sim.size() == pos_mask.size()  # sanity check

        neg_mask = 1 - pos_mask
        pos = (sim * pos_mask).sum(dim=1)
        neg = (sim * neg_mask).sum(dim=1)

        loss = pos / (pos + neg)
        loss = -torch.log(loss)

        return loss.mean()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


class DebiasedInfoNCE(Loss):
    def __init__(self, tau, tau_plus=0.1):
        super(DebiasedInfoNCE, self).__init__()
        self.tau = tau
        self.tau_plus = tau_plus

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)

        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        pos = pos_sum / pos_mask.int().sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        ng = (-num_neg * self.tau_plus * pos + neg_sum) / (1 - self.tau_plus)
        ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))

        log_prob = sim - torch.log((pos + ng).sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return loss.mean()


class HardnessInfoNCE(Loss):
    def __init__(self, tau, tau_plus=0.1, beta=1.0):
        super(HardnessInfoNCE, self).__init__()
        self.tau = tau
        self.tau_plus = tau_plus
        self.beta = beta

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)

        pos = (exp_sim * pos_mask).sum(dim=1) / pos_mask.int().sum(dim=1)
        imp = torch.exp(self.beta * (sim * neg_mask))
        reweight_neg = (imp * (exp_sim * neg_mask)).sum(dim=1) / imp.mean(dim=1)
        ng = (-num_neg * self.tau_plus * pos + reweight_neg) / (1 - self.tau_plus)
        ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))

        log_prob = sim - torch.log((pos + ng).sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return loss.mean()


class HardMixingLoss(torch.nn.Module):
    def __init__(self, projection):
        super(HardMixingLoss, self).__init__()
        self.projection = projection

    @staticmethod
    def tensor_similarity(z1, z2):
        z1 = F.normalize(z1, dim=-1)  # [N, d]
        z2 = F.normalize(z2, dim=-1)  # [N, s, d]
        return torch.bmm(z2, z1.unsqueeze(dim=-1)).squeeze()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, threshold=0.1, s=150, mixup=0.2, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        num_samples = z1.shape[0]
        device = z1.device

        threshold = int(num_samples * threshold)

        refl1 = _similarity(z1, z1).diag()
        refl2 = _similarity(z2, z2).diag()
        pos_similarity = f(_similarity(z1, z2))
        neg_similarity1 = torch.cat([_similarity(z1, z1), _similarity(z1, z2)], dim=1)  # [n, 2n]
        neg_similarity2 = torch.cat([_similarity(z2, z1), _similarity(z2, z2)], dim=1)
        neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
        neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
        neg_similarity1 = f(neg_similarity1)
        neg_similarity2 = f(neg_similarity2)
        z_pool = torch.cat([z1, z2], dim=0)
        hard_samples1 = z_pool[indices1[:, :threshold]]  # [N, k, d]
        hard_samples2 = z_pool[indices2[:, :threshold]]
        hard_sample_idx1 = torch.randint(hard_samples1.shape[1], size=[num_samples, 2 * s]).to(device)  # [N, 2 * s]
        hard_sample_idx2 = torch.randint(hard_samples2.shape[1], size=[num_samples, 2 * s]).to(device)
        hard_sample_draw1 = hard_samples1[
            torch.arange(num_samples).unsqueeze(-1), hard_sample_idx1]  # [N, 2 * s, d]
        hard_sample_draw2 = hard_samples2[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx2]
        hard_sample_mixing1 = mixup * hard_sample_draw1[:, :s, :] + (1 - mixup) * hard_sample_draw1[:, s:, :]
        hard_sample_mixing2 = mixup * hard_sample_draw2[:, :s, :] + (1 - mixup) * hard_sample_draw2[:, s:, :]

        h_m1 = self.projection(hard_sample_mixing1)
        h_m2 = self.projection(hard_sample_mixing2)

        neg_m1 = f(self.tensor_similarity(z1, h_m1)).sum(dim=1)
        neg_m2 = f(self.tensor_similarity(z2, h_m2)).sum(dim=1)
        pos = pos_similarity.diag()
        neg1 = neg_similarity1.sum(dim=1)
        neg2 = neg_similarity2.sum(dim=1)
        loss1 = -torch.log(pos / (neg1 + neg_m1 - refl1))
        loss2 = -torch.log(pos / (neg2 + neg_m2 - refl2))
        loss = (loss1 + loss2) * 0.5
        loss = loss.mean()
        return loss


class RingLoss(torch.nn.Module):
    def __init__(self):
        super(RingLoss, self).__init__()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, y: torch.Tensor, tau, threshold=0.1, *args, **kwargs):
        f = lambda x: torch.exp(x / tau)
        num_samples = h1.shape[0]
        device = h1.device
        threshold = int(num_samples * threshold)

        false_neg_mask = torch.zeros((num_samples, 2 * num_samples), dtype=torch.int).to(device)
        for i in range(num_samples):
            false_neg_mask[i] = (y == y[i]).repeat(2)

        pos_sim = f(_similarity(h1, h2))
        neg_sim1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
        neg_sim2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
        neg_sim1, indices1 = torch.sort(neg_sim1, descending=True)
        neg_sim2, indices2 = torch.sort(neg_sim2, descending=True)

        y_repeated = y.repeat(2)
        false_neg_cnt = torch.zeros((num_samples)).to(device)
        for i in range(num_samples):
            false_neg_cnt[i] = (y_repeated[indices1[i, threshold:-threshold]] == y[i]).sum()

        neg_sim1 = f(neg_sim1[:, threshold:-threshold])
        neg_sim2 = f(neg_sim2[:, threshold:-threshold])

        pos = pos_sim.diag()
        neg1 = neg_sim1.sum(dim=1)
        neg2 = neg_sim2.sum(dim=1)

        loss1 = -torch.log(pos / neg1)
        loss2 = -torch.log(pos / neg2)

        loss = (loss1 + loss2) * 0.5
        loss = loss.mean()

        return loss
