import torch
import torch.nn.functional as F


def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)


def variance_loss(z1, z2):
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1, z2):
    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor,
                sim_weight, var_weight, cov_weight):
    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss


class VICRegLoss(torch.nn.Module):
    def __init__(self, projection,
                 sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super(VICRegLoss, self).__init__()
        self.projection = projection
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = vicreg_loss(h1, h2, self.sim_weight, self.var_weight, self.cov_weight)
        l2 = vicreg_loss(h2, h1, self.sim_weight, self.var_weight, self.cov_weight)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
