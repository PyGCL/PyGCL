import torch
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def nt_xent_loss(h1: torch.FloatTensor, h2: torch.FloatTensor,
                 tau: float, *args, **kwargs):
    f = lambda x: torch.exp(x / tau)
    inter_sim = f(_similarity(h1, h1))
    intra_sim = f(_similarity(h1, h2))
    pos = intra_sim.diag()
    neg = inter_sim.sum(dim=1) + intra_sim.sum(dim=1) - inter_sim.diag()

    loss = pos / neg
    loss = -torch.log(loss)
    return loss


def debiased_nt_xent_loss(h1: torch.Tensor, h2: torch.Tensor,
                          tau: float, tau_plus: float, *args, **kwargs):
    f = lambda x: torch.exp(x / tau)
    intra_sim = f(_similarity(h1, h1))
    inter_sim = f(_similarity(h1, h2))

    pos = inter_sim.diag()
    neg = intra_sim.sum(dim=1) - intra_sim.diag() \
          + inter_sim.sum(dim=1) - inter_sim.diag()

    num_neg = h1.size()[0] * 2 - 2
    ng = (-num_neg * tau_plus * pos + neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / tau))

    return -torch.log(pos / (pos + ng))


def hardness_nt_xent_loss(h1: torch.Tensor, h2: torch.Tensor,
                          tau: float, tau_plus: float, beta: float, *args, **kwargs):
    f = lambda x: torch.exp(x / tau)
    intra_sim = f(_similarity(h1, h1))
    inter_sim = f(_similarity(h1, h2))

    pos = inter_sim.diag()
    neg = intra_sim.sum(dim=1) - intra_sim.diag() \
          + inter_sim.sum(dim=1) - inter_sim.diag()

    num_neg = h1.size()[0] * 2 - 2
    # imp = (beta * neg.log()).exp()
    imp = beta * neg
    reweight_neg = (imp * neg) / neg.mean()
    neg = (-num_neg * tau_plus * pos + reweight_neg) / (1 - tau_plus)
    neg = torch.clamp(neg, min=num_neg * np.e ** (-1. / tau))

    return -torch.log(pos / (pos + neg))


def subsampling_nt_xent_loss(h1: torch.Tensor, h2: torch.Tensor,
                             tau: float, sample_size: int, *args, **kwargs):
    f = lambda x: torch.exp(x / tau)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    device = h1.device
    num_nodes = h1.size(0)
    neg_indices = torch.randint(low=0, high=num_nodes * 2, size=(sample_size,), device=device)

    z_pool = torch.cat([h1, h2], dim=0)
    negatives = z_pool[neg_indices]

    pos = f(cos(h1, h2))
    neg = f(_similarity(h1, negatives)).sum(dim=1)

    loss = -torch.log(pos / (pos + neg))

    return loss


def nt_xent_loss_en(anchor: torch.FloatTensor,
                    samples: torch.FloatTensor,
                    pos_mask: torch.FloatTensor,
                    tau: float, *args, **kwargs):
    f = lambda x: torch.exp(x / tau)
    sim = f(_similarity(anchor, samples))  # anchor x sample
    assert sim.size() == pos_mask.size()  # sanity check

    neg_mask = 1 - pos_mask
    pos = (sim * pos_mask).sum(dim=1)
    neg = (sim * neg_mask).sum(dim=1)

    loss = pos / (pos + neg)
    loss = -torch.log(loss)

    return loss.mean()


class InfoNCELoss(torch.nn.Module):
    def __init__(self, loss_fn=nt_xent_loss):
        super(InfoNCELoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, h1: torch.FloatTensor, h2: torch.FloatTensor, *args, **kwargs):
        l1 = self.loss_fn(h1, h2, *args, **kwargs)
        l2 = self.loss_fn(h2, h1, *args, **kwargs)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret


class InfoNCELossG2L(torch.nn.Module):
    def __init__(self):
        super(InfoNCELossG2L, self).__init__()

    def forward(self,
                h1: torch.FloatTensor, g1: torch.FloatTensor,
                h2: torch.FloatTensor, g2: torch.FloatTensor,
                batch: torch.LongTensor, tau: float, *args, **kwargs):
        num_nodes = h1.size()[0]  # M := num_nodes
        ones = torch.eye(num_nodes, dtype=torch.float32, device=h1.device)  # [M, M]
        pos_mask = scatter(ones, batch, dim=0, reduce='sum')  # [M, N]
        l1 = nt_xent_loss_en(g1, h2, pos_mask=pos_mask, tau=tau)
        l2 = nt_xent_loss_en(g2, h1, pos_mask=pos_mask, tau=tau)
        return l1 + l2


class InfoNCELossG2LEN(torch.nn.Module):
    def __init__(self):
        super(InfoNCELossG2L, self).__init__()

    def forward(self,
                h1: torch.FloatTensor, g1: torch.FloatTensor,
                h2: torch.FloatTensor, g2: torch.FloatTensor,
                h3: torch.FloatTensor, h4: torch.FloatTensor,
                *args, **kwargs):
        num_nodes = h1.size()[0]
        device = h1.device
        pos_mask1 = torch.ones((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask0 = torch.zeros((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)

        samples1 = torch.cat([h2, h4], dim=0)
        samples2 = torch.cat([h1, h3], dim=0)

        l1 = nt_xent_loss_en(g1, samples1, pos_mask=pos_mask, *args, **kwargs)
        l2 = nt_xent_loss_en(g2, samples2, pos_mask=pos_mask, *args, **kwargs)

        return l1 + l2


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
