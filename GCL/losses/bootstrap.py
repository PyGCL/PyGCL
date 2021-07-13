import torch
import torch.nn.functional as F


class BootstrapLoss(torch.nn.Module):
    def __init__(self):
        super(BootstrapLoss, self).__init__()

    def forward(self, h1_pred, h2_pred, h1_target, h2_target):
        cos = torch.nn.CosineSimilarity()
        loss = 2 - 2 * cos(h1_pred, h2_target)
        loss += 2 - 2 * cos(h2_pred, h1_target)
        loss = loss.mean()
        return loss


class BootstrapLossG2L(torch.nn.Module):
    def __init__(self):
        super(BootstrapLossG2L, self).__init__()

    @staticmethod
    def bootstrap_loss(g: torch.FloatTensor, h: torch.FloatTensor):
        g = F.normalize(g, dim=-1, p=2)
        h = F.normalize(h, dim=-1, p=2)

        similarity = torch.unsqueeze(g, dim=0) @ h.t()
        return similarity.sum(dim=-1)

    def forward(self, h1_pred, h2_pred, g1_target, g2_target):
        loss = self.bootstrap_loss(g2_target, h1_pred)
        loss += self.bootstrap_loss(g1_target, h2_pred)
        loss = loss.mean()
        return loss
