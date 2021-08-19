import torch
import torch.nn.functional as F
from .losses import Loss


class VICReg(Loss):
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
        super(VICReg, self).__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    @staticmethod
    def invariance_loss(h1, h2):
        return F.mse_loss(h1, h2)

    def variance_loss(self, h1, h2):
        std_z1 = torch.sqrt(h1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(h2.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        return std_loss

    @staticmethod
    def covariance_loss(h1, h2):
        num_nodes, hidden_dim = h1.size()

        h1 = h1 - h1.mean(dim=0)
        h2 = h2 - h2.mean(dim=0)
        cov_z1 = (h1.T @ h1) / (num_nodes - 1)
        cov_z2 = (h2.T @ h2) / (num_nodes - 1)

        diag = torch.eye(hidden_dim, device=h1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / hidden_dim + cov_z2[~diag.bool()].pow_(2).sum() / hidden_dim
        return cov_loss

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        sim_loss = self.invariance_loss(anchor, sample)
        var_loss = self.variance_loss(anchor, sample)
        cov_loss = self.covariance_loss(anchor, sample)

        loss = self.sim_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return loss.mean()
