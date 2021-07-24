import torch
from torch_scatter import scatter


class SameScaleSampler(torch.nn.Module):
    def __init__(self):
        super(SameScaleSampler, self).__init__()

    def forward(self, anchor, sample):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


class CrossScaleSampler(torch.nn.Module):
    def __init__(self):
        super(CrossScaleSampler, self).__init__()

    def forward(self, anchor, sample, batch=None, neg_sample=None, use_gpu=True):
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]   # N
        device = sample.device

        if neg_sample is not None:
            assert num_graphs == 1  # for node-level tasks
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)     # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)         # 2N * K
        else:
            if use_gpu:
                ones = torch.eye(num_nodes, dtype=torch.float32, device=device)     # N * N
                pos_mask = scatter(ones, batch, dim=0, reduce='sum')                # M * N
            else:
                pos_mask = torch.zeros((num_graphs, num_nodes), dtype=torch.float32).to(device)
                for node_idx, graph_idx in enumerate(batch):
                    pos_mask[graph_idx][node_idx] = 1.                              # M * N

        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask
