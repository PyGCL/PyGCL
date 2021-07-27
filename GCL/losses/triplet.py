import torch
from torch_scatter import scatter

from .losses import Loss


class TripletLoss(Loss):
    def __init__(self, margin: float = 1.0, p: float = 2):
        super(TripletLoss, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=p, reduction='none')

    def __compute(self, anchor, sample, pos_mask, neg_mask=None, *args, **kwargs):
        num_anchors = anchor.size()[0]
        num_samples = sample.size()[0]

        # Key idea here:
        #  (1) Use all possible triples (will be num_anchors * num_positives * num_negatives triples in total)
        #  (2) Use PyTorch's TripletMarginLoss to compute the marginal loss for each triple
        #  (3) Since TripletMarginLoss accepts input tensors of shape (B, D), where B is the batch size,
        #        we have to manually construct all triples and flatten them as an input tensor in the
        #        shape of (num_triples, D).
        #  (4) We first compute loss for all triples (including those that are not anchor - positive - negative), which
        #        will be num_anchors * num_samples * num_samples triples, and then filter them with masks.

        # compute negative mask
        neg_mask = 1. - pos_mask if neg_mask is None else neg_mask

        anchor = torch.unsqueeze(anchor, dim=1)  # [N, 1, D]
        anchor = torch.unsqueeze(anchor, dim=1)  # [N, 1, 1, D]
        anchor = anchor.expand(-1, num_samples, num_samples, -1)  # [N, M, M, D]
        anchor = torch.flatten(anchor, end_dim=1)  # [N * M * M, D]

        pos_sample = torch.unsqueeze(sample, dim=0)  # [1, M, D]
        pos_sample = torch.unsqueeze(pos_sample, dim=2)  # [1, M, 1, D]
        pos_sample = pos_sample.expand(num_anchors, -1, num_samples, -1)  # [N, M, M, D]
        pos_sample = torch.flatten(pos_sample, end_dim=1)  # [N * M * M, D]

        neg_sample = torch.unsqueeze(sample, dim=0)  # [1, M, D]
        neg_sample = torch.unsqueeze(neg_sample, dim=0)  # [1, 1, M, D]
        neg_sample = neg_sample.expand(num_anchors, -1, num_samples, -1)  # [N, M, M, D]
        neg_sample = torch.flatten(neg_sample, end_dim=1)  # [N * M * M, D]

        loss = self.loss_fn(anchor, pos_sample, neg_sample)  # [N, M, M]
        loss = loss.view(num_anchors, num_samples, num_samples)

        pos_mask1 = torch.unsqueeze(pos_mask, dim=2)  # [N, M, 1]
        pos_mask1 = pos_mask1.expand(-1, -1, num_samples)  # [N, M, M]
        neg_mask1 = torch.unsqueeze(neg_mask, dim=1)  # [N, 1, M]
        neg_mask1 = neg_mask1.expand(-1, num_samples, -1)  # [N, M, M]

        pair_mask = pos_mask1 * neg_mask1  # [N, M, M]
        num_pairs = pair_mask.sum()

        loss = loss * pair_mask
        loss = loss.sum()

        return loss / num_pairs


def triplet_loss(anchor: torch.FloatTensor, samples: torch.FloatTensor,
                 pos_mask: torch.FloatTensor, eps: float, *args, **kwargs):
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


def triplet_loss_en(anchor: torch.FloatTensor,  # [N, D]
                    pos_samples: torch.FloatTensor,  # [N, P, D]
                    neg_samples: torch.FloatTensor,  # [N, Q, D]
                    eps: float, *args, **kwargs):
    anchor = torch.unsqueeze(anchor, dim=1)  # [N, 1, D]
    pos_dist = torch.cdist(anchor, pos_samples, p=2)  # [N, 1, P]
    neg_dist = torch.cdist(anchor, neg_samples, p=2)  # [N, 1, Q]
    pos_dist, neg_dist = torch.squeeze(pos_dist, dim=1), torch.squeeze(neg_dist, dim=1)

    loss = pos_dist - neg_dist.sum(dim=1, keepdim=True) / neg_samples.size()[1] + eps  # [N, P]
    loss = torch.where(loss > 0, loss, torch.zeros_like(loss))

    return loss.mean(dim=1).sum()


class TripletLoss(torch.nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, h1: torch.FloatTensor, h2: torch.FloatTensor, eps, *args, **kwargs):
        num_nodes = h1.size(0)
        device = h1.device

        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)

        l1 = triplet_loss(h1, h2, pos_mask=pos_mask, eps=eps, *args, **kwargs)
        l2 = triplet_loss(h2, h1, pos_mask=pos_mask, eps=eps, *args, **kwargs)

        return ((l1 + l2) * 0.5).mean()


class TripletLossG2L(torch.nn.Module):
    def __init__(self):
        super(TripletLossG2L, self).__init__()

    def forward(self, h1: torch.FloatTensor, g1: torch.FloatTensor,
                      h2: torch.FloatTensor, g2: torch.FloatTensor,
                      batch: torch.LongTensor, eps: float, *args, **kwargs):
        num_nodes = h1.size()[0]  # M := num_nodes
        ones = torch.eye(num_nodes, dtype=torch.float32, device=h1.device)  # [M, M]
        pos_mask = scatter(ones, batch, dim=0, reduce='sum')  # [M, N]
        l1 = triplet_loss(g1, h2, pos_mask=pos_mask, eps=eps, *args, **kwargs)
        l2 = triplet_loss(g2, h1, pos_mask=pos_mask, eps=eps, *args, **kwargs)

        return ((l1 + l2) * 0.5).mean()


class TripletLossG2LEN(torch.nn.Module):
    def __init__(self):
        super(TripletLossG2LEN, self).__init__()

    def forward(self,
                h1: torch.FloatTensor, g1: torch.FloatTensor,
                h2: torch.FloatTensor, g2: torch.FloatTensor,
                h3: torch.FloatTensor, h4: torch.FloatTensor,
                eps, *args, **kwargs):
        anchor = torch.cat([g1, g2], dim=0)
        pos_samples = torch.stack([h2, h1], dim=0)
        neg_samples = torch.stack([h4, h3], dim=0)

        return triplet_loss_en(anchor, pos_samples, neg_samples, eps=eps, *args, **kwargs)
