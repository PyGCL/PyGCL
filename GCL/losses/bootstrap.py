import torch


def bootstrap_loss(x: torch.FloatTensor, y: torch.FloatTensor):
    cos = torch.nn.CosineSimilarity()
    return 2 - 2 * cos(x, y)


class BootstrapLoss(torch.nn.Module):
    def __init__(self):
        super(BootstrapLoss, self).__init__()

    def forward(self, h1_pred, h2_pred, h1_target, h2_target):
        loss = bootstrap_loss(h1_pred, h2_target.detach())
        loss += bootstrap_loss(h2_pred, h1_target.detach())
        loss = loss.mean()
        return loss
