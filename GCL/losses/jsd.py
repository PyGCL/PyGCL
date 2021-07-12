import torch

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


class JSDLossG2L(torch.nn.Module):
    def __init__(self, discriminator):
        super(JSDLossG2L, self).__init__()
        self.discriminator = discriminator

    def forward(self, h1, g1, h2, g2, batch):
        num_graphs = g1.shape[0]
        num_nodes = h1.shape[0]
        device = h1.device

        pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
        for node_idx, graph_idx in enumerate(batch):
            pos_mask[node_idx][graph_idx] = 1.

        l1 = jsd_loss(g2, h1, self.discriminator, pos_mask=pos_mask.t())
        l2 = jsd_loss(g1, h2, self.discriminator, pos_mask=pos_mask.t())

        return l1 + l2


class JSDLossL2L(torch.nn.Module):
    def __init__(self, discriminator):
        super(JSDLossL2L, self).__init__()
        self.discriminator = discriminator

    def forward(self, h1: torch.FloatTensor, h2: torch.FloatTensor):
        num_nodes = h1.size(0)
        device = h1.device

        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)

        return jsd_loss(h1, h2, discriminator=self.discriminator, pos_mask=pos_mask)


class JSDLossEN(torch.nn.Module):
    def __init__(self, discriminator):
        super(JSDLossEN, self).__init__()
        self.discriminator = discriminator

    def forward(self,
                h1: torch.FloatTensor, g1: torch.FloatTensor,
                h2: torch.FloatTensor, g2: torch.FloatTensor,
                h3: torch.FloatTensor, h4: torch.FloatTensor):
        num_nodes = h1.size(0)
        device = h1.device

        pos_mask1 = torch.ones((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask0 = torch.zeros((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)

        samples1 = torch.cat([h2, h4], dim=0)
        samples2 = torch.cat([h1, h3], dim=0)

        l1 = jsd_loss(g1, samples1, self.discriminator, pos_mask=pos_mask)
        l2 = jsd_loss(g2, samples2, self.discriminator, pos_mask=pos_mask)

        return l1 + l2
