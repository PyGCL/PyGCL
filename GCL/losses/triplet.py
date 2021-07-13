import torch


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
    def __init__(self, *args, **kwargs):
        super(TripletLoss, self).__init__()

    def forward(self, h1: torch.FloatTensor, h2: torch.FloatTensor, eps, *args, **kwargs):
        num_nodes = h1.size(0)
        device = h1.device

        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)

        l1 = triplet_loss(h1, h2, pos_mask=pos_mask, eps=eps, *args, **kwargs)
        l2 = triplet_loss(h2, h1, pos_mask=pos_mask, eps=eps, *args, **kwargs)

        return ((l1 + l2) * 0.5).mean()


class TripletLossG2LEN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
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
