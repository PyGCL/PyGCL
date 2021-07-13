import torch
import torch.nn.functional as F
from torch_scatter import scatter


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
    def bootstrap_loss(g: torch.FloatTensor, h: torch.FloatTensor, batch=None):
        if batch is None:
            g = F.normalize(g, dim=-1, p=2)
            h = F.normalize(h, dim=-1, p=2)

            similarity = torch.unsqueeze(g, dim=0) @ h.t()
            return similarity.sum(dim=-1)

        num_nodes = h.size()[0]  # M := num_nodes
        device = h.device

        values = torch.eye(num_nodes, dtype=torch.float32, device=device)  # [M, M]
        pos_mask = scatter(values, batch, dim=0, reduce='sum')  # [M, N]

        g = F.normalize(g, dim=-1, p=2)
        h = F.normalize(h, dim=-1, p=2)

        similarity = g @ h.t()
        return (similarity * pos_mask).sum(dim=-1)

    def forward(self, h1_pred, h2_pred, g1_target, g2_target, batch=None):
        loss = self.bootstrap_loss(g2_target, h1_pred, batch=batch)
        loss += self.bootstrap_loss(g1_target, h2_pred, batch=batch)
        loss = loss.mean()
        return loss
