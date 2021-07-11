from typing import Optional

import torch
import torch.nn.functional as F

from typing import Callable, Optional, Tuple, List
from GCL.loss import nt_xent_loss, _similarity, jsd_loss, triplet_loss, candidate_mask_mixing_loss, subsampling_nt_xent_loss
from GCL.losses import bt_loss, vicreg_loss
import GCL.augmentations as A

from utils import set_differ


def ring_loss(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor, threshold=0.1):
    f = lambda x: torch.exp(x / self.tau)
    num_samples = z1.shape[0]
    device = z1.device
    threshold = int(num_samples * threshold)

    h1 = self.projection(z1)
    h2 = self.projection(z2)

    false_negative_mask = torch.zeros((num_samples, 2 * num_samples), dtype=torch.int).to(device)
    for i in range(num_samples):
        false_negative_mask[i] = (y == y[i]).repeat(2)

    pos_similarity = f(_similarity(h1, h2))
    neg_similarity1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
    neg_similarity2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
    neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
    neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)

    y_repeated = y.repeat(2)
    false_negative_count = torch.zeros((num_samples)).to(device)
    for i in range(num_samples):
        false_negative_count[i] = (y_repeated[indices1[i, threshold:-threshold]] == y[i]).sum()
    within_threshold = (false_negative_count / threshold / 2).mean().item()
    within_dataset = (false_negative_count / num_samples / 2).mean().item()
    print(f'False negatives: {within_threshold * 100:.2f}%, {within_dataset * 100:.2f}% overall')

    neg_similarity1 = f(neg_similarity1[:, threshold:-threshold])
    neg_similarity2 = f(neg_similarity2[:, threshold:-threshold])

    neg_similarity1 = f(neg_similarity1 * (1 - false_negative_mask))
    neg_similarity2 = f(neg_similarity2 * (1 - false_negative_mask))

    pos = pos_similarity.diag()
    neg1 = neg_similarity1.sum(dim=1)
    neg2 = neg_similarity2.sum(dim=1)

    loss1 = -torch.log(pos / neg1)
    loss2 = -torch.log(pos / neg2)

    loss = (loss1 + loss2) * 0.5
    loss = loss.mean()

    return loss


def hard_mixing_loss(self, z1: torch.Tensor, z2: torch.Tensor, threshold=0.1, s=80, mixup=0.2):
    f = lambda x: torch.exp(x / self.tau)
    num_samples = z1.shape[0]
    device = z1.device
    threshold = int(num_samples * threshold)

    h1 = self.projection(z1)
    h2 = self.projection(z2)

    pos_similarity = f(_similarity(h1, h2))
    neg_similarity1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
    neg_similarity2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
    # neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
    # neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
    indices1 = torch.topk(neg_similarity1, threshold).indices
    indices2 = torch.topk(neg_similarity2, threshold).indices

    neg_similarity1 = f(neg_similarity1)
    neg_similarity2 = f(neg_similarity2)

    z_pool = torch.cat([z1, z2], dim=0)
    hard_samples1 = z_pool[indices1]  # [N, k, d]
    hard_samples2 = z_pool[indices2]
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

    loss1 = -torch.log(pos / (neg1 + neg_m1))
    loss2 = -torch.log(pos / (neg2 + neg_m2))

    loss = (loss1 + loss2) * 0.5
    loss = loss.mean()

    return loss


class GRACE(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module,
                 augmentation: Tuple[A.GraphAug, A.GraphAug],
                 hidden_dim: int, proj_dim: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.augmentation = augmentation

        self.tau = tau

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

        self.num_hidden = hidden_dim

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aug1, aug2 = self.augmentation
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        # edge_index2 = set_differ(edge_index.t(), edge_index1.t()).t()

        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)

        return z, z1, z2

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def subsampling_loss(self, z1: torch.Tensor, z2: torch.Tensor, sample_size: int, mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = subsampling_nt_xent_loss(h1, h2, sample_size=sample_size, temperature=self.tau)
        l2 = subsampling_nt_xent_loss(h2, h1, sample_size=sample_size, temperature=self.tau)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True,
             batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            batch_size = z1.size(0)

        l1 = nt_xent_loss(h1, h2, batch_size, self.tau)
        l2 = nt_xent_loss(h2, h1, batch_size, self.tau)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret



    def jsd_loss(self, z1: torch.FloatTensor, z2: torch.FloatTensor):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        num_nodes = h1.size(0)
        device = h1.device

        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)

        return jsd_loss(h1, h2, discriminator=_similarity, pos_mask=pos_mask)

    def triplet_loss(self, z1: torch.FloatTensor, z2: torch.FloatTensor, eps: float = 1):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        num_nodes = h1.size(0)
        device = h1.device

        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)

        l1 = triplet_loss(h1, h2, pos_mask=pos_mask, eps=eps)
        l2 = triplet_loss(h2, h1, pos_mask=pos_mask, eps=eps)

        return ((l1 + l2) * 0.5).mean()

