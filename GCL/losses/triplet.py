import torch
from .losses import Loss


class TripletMarginSP(Loss):
    def __init__(self, margin: float = 1.0, p: float = 2, *args, **kwargs):
        super(TripletMarginSP, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=p, reduction='none')
        self.margin = margin

    def compute(self, anchor, sample, pos_mask, neg_mask=None, *args, **kwargs):
        neg_mask = 1. - pos_mask

        num_pos = pos_mask.to(torch.long).sum(dim=1)
        num_neg = neg_mask.to(torch.long).sum(dim=1)

        dist = torch.cdist(anchor, sample, p=2)  # [num_anchors, num_samples]

        pos_dist = pos_mask * dist
        neg_dist = neg_mask * dist

        pos_dist, neg_dist = pos_dist.sum(dim=1), neg_dist.sum(dim=1)

        loss = pos_dist / num_pos - neg_dist / num_neg + self.margin
        loss = torch.where(loss > 0, loss, torch.zeros_like(loss))

        return loss.mean()


class TripletMargin(Loss):
    def __init__(self, margin: float = 1.0, p: float = 2, *args, **kwargs):
        super(TripletMargin, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=p, reduction='none')
        self.margin = margin

    def compute(self, anchor, sample, pos_mask, neg_mask=None, *args, **kwargs):
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
