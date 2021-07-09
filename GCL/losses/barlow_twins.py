import torch


def bt_loss(z1: torch.Tensor, z2: torch.Tensor, batch_norm=True, eps=1e-15):
    batch_size = z1.size(0)
    feature_dim = z1.size(1)
    lambda_ = 1 / feature_dim

    if batch_norm:
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + eps)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        c = z1.T @ z2 / batch_size

    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

    return loss
