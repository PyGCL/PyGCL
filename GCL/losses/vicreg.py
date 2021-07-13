import torch
import torch.nn.functional as F


def invariance_loss(h1, h2):
    return F.mse_loss(h1, h2)


def variance_loss(h1, h2, eps=1e-4):
    std_z1 = torch.sqrt(h1.var(dim=0) + eps)
    std_z2 = torch.sqrt(h2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(h1, h2):
    num_nodes, hidden_dim = h1.size()

    h1 = h1 - h1.mean(dim=0)
    h2 = h2 - h2.mean(dim=0)
    cov_z1 = (h1.T @ h1) / (num_nodes - 1)
    cov_z2 = (h2.T @ h2) / (num_nodes - 1)

    diag = torch.eye(hidden_dim, device=h1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / hidden_dim + cov_z2[~diag.bool()].pow_(2).sum() / hidden_dim
    return cov_loss


def vicreg_loss(h1, h2, sim_weight, var_weight, cov_weight, *args, **kwargs):
    sim_loss = invariance_loss(h1, h2)
    var_loss = variance_loss(h1, h2)
    cov_loss = covariance_loss(h1, h2)

    loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss


class VICRegLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(VICRegLoss, self).__init__()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor,
                sim_weight=25.0, var_weight=25.0, cov_weight=1.0,
                mean: bool = True,
                *args, **kwargs):
        l1 = vicreg_loss(h1, h2, sim_weight=sim_weight, var_weight=var_weight, cov_weight=cov_weight, *args, **kwargs)
        l2 = vicreg_loss(h2, h1, sim_weight=sim_weight, var_weight=var_weight, cov_weight=cov_weight, *args, **kwargs)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
