import torch


def bt_loss(h1: torch.Tensor, h2: torch.Tensor, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)

    if batch_norm:
        z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
        z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        c = h1.T @ h2 / batch_size

    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

    return loss


class BTLoss(torch.nn.Module):
    def __init__(self, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
        super(BTLoss, self).__init__()
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, mean: bool = True, *args, **kwargs):
        l1 = bt_loss(h1, h2, self.lambda_, batch_norm=self.batch_norm, eps=self.eps, *args, **kwargs)
        l2 = bt_loss(h2, h1, self.lambda_, batch_norm=self.batch_norm, eps=self.eps, *args, **kwargs)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
