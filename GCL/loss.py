from typing import Optional
import torch

import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def _similarity(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return z1 @ z2.t()


def subsampling_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                             sample_size: int, temperature: float):
    f = lambda x: torch.exp(x / temperature)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    device = z1.device
    num_nodes = z1.size(0)
    neg_indices = torch.randint(low=0, high=num_nodes * 2, size=(sample_size,), device=device)

    z_pool = torch.cat([z1, z2], dim=0)
    negatives = z_pool[neg_indices]

    pos = f(cos(z1, z2))
    neg = f(_similarity(z1, negatives)).sum(dim=1)

    loss = -torch.log(pos / (pos + neg))

    return loss


def nt_xent_loss_with_mask(anchor: torch.FloatTensor, samples: torch.FloatTensor, pos_mask: torch.FloatTensor, temperature: float):
    f = lambda x: torch.exp(x / temperature)
    sim = f(_similarity(anchor, samples))  # anchor x sample
    assert sim.size() == pos_mask.size()  # sanity check

    pos = sim * pos_mask
    pos = pos.sum(dim=1)
    neg = sim.sum(dim=1) - pos

    loss = pos / (pos + neg)
    loss = -torch.log(loss)
    # loss = loss / pos_mask.sum(dim=1)

    return loss.mean()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 batch_size: int, temperature: float):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        batch_mask = indices[i * batch_size: (i + 1) * batch_size]
        intra_similarity = f(_similarity(z1[batch_mask], z1))  # [B, N]
        inter_similarity = f(_similarity(z1[batch_mask], z2))  # [B, N]

        positives = inter_similarity[:, batch_mask].diag()
        negatives = intra_similarity.sum(dim=1) + inter_similarity.sum(dim=1)\
                    - intra_similarity[:, batch_mask].diag()

        losses.append(-torch.log(positives / negatives))

    return torch.cat(losses)


def debiased_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                          tau: float, tau_plus: float):
    f = lambda x: torch.exp(x / tau)
    intra_similarity = f(_similarity(z1, z1))
    inter_similarity = f(_similarity(z1, z2))

    pos = inter_similarity.diag()
    neg = intra_similarity.sum(dim=1) - intra_similarity.diag() \
          + inter_similarity.sum(dim=1) - inter_similarity.diag()

    num_neg = z1.size()[0] * 2 - 2
    ng = (-num_neg * tau_plus * pos + neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / tau))

    return -torch.log(pos / (pos + ng))


def hardness_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                          tau: float, tau_plus: float, beta: float):
    f = lambda x: torch.exp(x / tau)
    intra_similarity = f(_similarity(z1, z1))
    inter_similarity = f(_similarity(z1, z2))

    pos = inter_similarity.diag()
    neg = intra_similarity.sum(dim=1) - intra_similarity.diag() \
          + inter_similarity.sum(dim=1) - inter_similarity.diag()

    num_neg = z1.size()[0] * 2 - 2
    imp = (beta * neg.log()).exp()
    reweight_neg = (imp * neg) / neg.mean()
    ng = (-num_neg * tau_plus * pos + reweight_neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / tau))

    return -torch.log(pos / (pos + ng))


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


def multiple_triplet_loss(anchor: torch.FloatTensor,  # [N, D]
                        pos_samples: torch.FloatTensor,  # [N, P, D]
                        neg_samples: torch.FloatTensor,  # [N, Q, D]
                        eps: float):
    anchor = torch.unsqueeze(anchor, dim=1)  # [N, 1, D]
    pos_dist = torch.cdist(anchor, pos_samples, p=2)  # [N, 1, P]
    neg_dist = torch.cdist(anchor, neg_samples, p=2)  # [N, 1, Q]
    pos_dist, neg_dist = torch.squeeze(pos_dist, dim=1), torch.squeeze(neg_dist, dim=1)

    loss = pos_dist - neg_dist.sum(dim=1, keepdim=True) / neg_samples.size()[1] + eps  # [N, P]
    loss = torch.where(loss > 0, loss, torch.zeros_like(loss))

    return loss.mean(dim=1).sum()


def triplet_loss(anchor: torch.FloatTensor, samples: torch.FloatTensor, pos_mask: torch.FloatTensor, eps: float):
    neg_mask = 1. - pos_mask

    num_pos = pos_mask.to(torch.long).sum(dim=1)
    num_neg = neg_mask.to(torch.long).sum(dim=1)

    dist = torch.cdist(anchor, samples, p=2)  # [num_anchors, num_samples]

    pos_dist = pos_mask * dist
    neg_dist = neg_mask * dist

    pos_dist, neg_dist = pos_dist.sum(dim=1), neg_dist.sum(dim=1)

    loss = pos_dist / num_pos - neg_dist / num_neg + eps
    loss = torch.where(loss > 0, loss, torch.zeros_like(loss))

    return loss.mean()


def candidate_mask_mixing_loss(self, z1: torch.Tensor, z2: torch.Tensor, candidate_mask: torch.Tensor, s: int, mixup: Optional[float] = None):
    """
    Args:
        self: Model instance.
        z1: Node embeddings of view 1.
        z2: Node embeddings of view 2.
        candidate_mask: Mask for mixing candidates.
        s: Number of negative samples to synthesize.
    Returns:
    """
    candidate_mask = candidate_mask.to(torch.float32)
    f = lambda x: torch.exp(x / self.tau)

    h1 = self.projection(z1)
    h2 = self.projection(z2)

    # detach gradient
    z1_ = z1.detach()
    z2_ = z2.detach()

    # sample mixing candidates
    if s > 0:
        distribution = Categorical(probs=candidate_mask + 1e-8)
        candidates1 = distribution.sample(sample_shape=[s]).t().to(z1.device)
        candidates2 = distribution.sample(sample_shape=[s]).t().to(z1.device)
    else:
        candidates1 = torch.tensor([]).view(z1.size()[0], s).to(torch.long).to(z1.device)
        candidates2 = torch.tensor([]).view(z1.size()[0], s).to(torch.long).to(z1.device)

    # get embeddings of candidates
    candidates1 = z1_[candidates1]  # [num_nodes, s, embed_dim]
    candidates2 = z2_[candidates2]

    # sample mixup coefficients
    if mixup is None:
        mixup = torch.rand((z1.size()[0], s)).expand((z1.size()[0], s, z1.size()[1])).to(z1.device)  # [num_nodes, s]

    # mixing
    z_mix = candidates1 * mixup + candidates2 * (1.0 - mixup)
    h_mix = self.projection(z_mix)

    # calculate the loss
    def tensor_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)  # [N, d]
        b = F.normalize(b, dim=-1)  # [N, s, d]
        return torch.bmm(b, a.unsqueeze(dim=-1)).squeeze()

    intra_similarity1 = f(_similarity(h1, h1))
    intra_similarity2 = f(_similarity(h2, h2))
    inter_similarity1 = f(_similarity(h1, h2))
    inter_similarity2 = inter_similarity1.t()
    mix_similarity1 = f(tensor_similarity(h1, h_mix))  # [num_nodes, s]
    mix_similarity2 = f(tensor_similarity(h2, h_mix))
    pos1 = inter_similarity1.diag()
    pos2 = inter_similarity2.diag()
    refl1 = intra_similarity1.diag()
    refl2 = intra_similarity2.diag()
    neg1 = intra_similarity1.sum(dim=1) + inter_similarity1.sum(dim=1) + mix_similarity1.sum(dim=1) - pos1 - refl1
    neg2 = intra_similarity2.sum(dim=1) + inter_similarity2.sum(dim=1) + mix_similarity2.sum(dim=1) - pos2 - refl2

    loss1 = -torch.log(pos1 / (pos1 + neg1))
    loss2 = -torch.log(pos2 / (pos2 + neg2))

    loss = (loss1 + loss2) * 0.5
    loss = loss.mean()
    return loss


def hard_mixing_loss(self, z1: torch.Tensor, z2: torch.Tensor, threshold=0.1, s=150, mixup=0.2):
    f = lambda x: torch.exp(x / self.tau)
    num_samples = z1.shape[0]
    device = z1.device
    threshold = int(num_samples * threshold)
    h1 = self.projection(z1)
    h2 = self.projection(z2)
    refl1 = _similarity(h1, h1).diag()
    refl2 = _similarity(h2, h2).diag()
    pos_similarity = f(_similarity(h1, h2))
    neg_similarity1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
    neg_similarity2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
    neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
    neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
    neg_similarity1 = f(neg_similarity1)
    neg_similarity2 = f(neg_similarity2)
    z_pool = torch.cat([z1, z2], dim=0)
    hard_samples1 = z_pool[indices1[:, :threshold]]  # [N, k, d]
    hard_samples2 = z_pool[indices2[:, :threshold]]
    hard_sample_idx1 = torch.randint(hard_samples1.shape[1], size=[num_samples, 2 * s]).to(device)  # [N, 2 * s]
    hard_sample_idx2 = torch.randint(hard_samples2.shape[1], size=[num_samples, 2 * s]).to(device)
    hard_sample_draw1 = hard_samples1[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx1]  # [N, 2 * s, d]
    hard_sample_draw2 = hard_samples2[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx2]
    hard_sample_mixing1 = mixup * hard_sample_draw1[:, :s, :] + (1 - mixup) * hard_sample_draw1[:, s:, :]
    hard_sample_mixing2 = mixup * hard_sample_draw2[:, :s, :] + (1 - mixup) * hard_sample_draw2[:, s:, :]
    h_m1 = self.projection(hard_sample_mixing1)
    h_m2 = self.projection(hard_sample_mixing2)
    def tensor_similarity(z1, z2):
        z1 = F.normalize(z1, dim=-1)  # [N, d]
        z2 = F.normalize(z2, dim=-1)  # [N, s, d]
        return torch.bmm(z2, z1.unsqueeze(dim=-1)).squeeze()
    neg_m1 = f(tensor_similarity(h1, h_m1)).sum(dim=1)
    neg_m2 = f(tensor_similarity(h2, h_m2)).sum(dim=1)
    pos = pos_similarity.diag()
    neg1 = neg_similarity1.sum(dim=1)
    neg2 = neg_similarity2.sum(dim=1)
    loss1 = -torch.log(pos / (neg1 + neg_m1 - refl1))
    loss2 = -torch.log(pos / (neg2 + neg_m2 - refl2))
    loss = (loss1 + loss2) * 0.5
    loss = loss.mean()
    return loss
