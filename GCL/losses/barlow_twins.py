import torch

from .loss import Loss


class BarlowTwins(Loss):
    def __init__(self, lambda_: float = None, batch_norm: bool = True, eps: float = 1e-5):
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def compute(self, contrast_instance, *args, **kwargs) -> torch.FloatTensor:
        anchor, sample, pos_mask, neg_mask = contrast_instance.unpack()

        batch_size = anchor.size(0)
        feature_dim = anchor.size(1)

        if self.lambda_ is None:
            lambda_ = 1. / feature_dim
        else:
            lambda_ = self.lambda_

        if self.batch_norm:
            z1_norm = (anchor - anchor.mean(dim=0)) / (anchor.std(dim=0) + self.eps)
            z2_norm = (sample - sample.mean(dim=0)) / (sample.std(dim=0) + self.eps)
            c = (z1_norm.T @ z2_norm) / batch_size
        else:
            c = anchor.T @ sample / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (1 - c.diagonal()).pow(2).sum()
        loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

        return loss.mean()
