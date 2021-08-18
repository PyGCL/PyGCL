import torch
from .losses import Loss


class BarlowTwinsLoss(Loss):
    def __init__(self, lambda_=None, batch_norm=True, eps=1e-15):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def compute(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs):
        batch_size = anchor.size(0)
        feature_dim = anchor.size(1)
        if self.lambda_ is None:
            self.lambda_ = 1 / feature_dim

        if self.batch_norm:
            z1_norm = (anchor - anchor.mean(dim=0)) / (anchor.std(dim=0) + self.eps)
            z2_norm = (sample - sample.mean(dim=0)) / (sample.std(dim=0) + self.eps)
            c = (z1_norm.T @ z2_norm) / batch_size
        else:
            c = anchor.T @ sample / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (1 - c.diagonal()).pow(2).sum()
        loss += self.lambda_ * c[off_diagonal_mask].pow(2).sum()

        return loss.mean()
