from typing import Callable, Tuple, Optional
import torch

from GCL.loss import jsd_loss, triplet_loss, nt_xent_loss_with_mask, multiple_triplet_loss
from GCL.augmentations import GraphAug
from torch_geometric.utils import subgraph
from torch_scatter import scatter


class MVGRL(torch.nn.Module):
    def __init__(self, gnn1: torch.nn.Module, gnn2: torch.nn.Module,
                 mlp1: torch.nn.Module, mlp2: torch.nn.Module,
                 augmentations: Tuple[GraphAug, GraphAug],
                 sample_size: int,
                 discriminator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=lambda x, y: x @ y.t()):
        super(MVGRL, self).__init__()
        self.gnn1 = gnn1
        self.gnn2 = gnn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.sample_size = sample_size
        self.discriminator = discriminator
        self.augmentations = augmentations

    def forward(
            self,
            batch,
            x: Optional[torch.Tensor],
            edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
            sample_size: int = None
    ):

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(batch.device)

        aug1, aug2 = self.augmentations
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        num_nodes = x.size(0)
        if sample_size is not None:
            subset_idx = torch.randperm(num_nodes, dtype=torch.long, device=x.device)[:sample_size]
            subset_idx = torch.sort(subset_idx).values
            x = x[subset_idx]
            x1 = x1[subset_idx]
            x2 = x2[subset_idx]
            batch = batch[subset_idx]
            edge_index1, edge_weight1 = subgraph(subset_idx, edge_index1, edge_weight1, relabel_nodes=True, num_nodes=num_nodes)
            edge_index2, edge_weight2 = subgraph(subset_idx, edge_index2, edge_weight2, relabel_nodes=True, num_nodes=num_nodes)
            num_nodes = sample_size

        shuffled_x1 = x1[torch.randperm(num_nodes)]
        shuffled_x2 = x2[torch.randperm(num_nodes)]

        z1, g1 = self.gnn1(x1, edge_index1, edge_weight1, batch)
        z2, g2 = self.gnn2(x2, edge_index2, edge_weight2, batch)

        z3, _ = self.gnn1(shuffled_x1, edge_index1, edge_weight1, batch)
        z4, _ = self.gnn2(shuffled_x2, edge_index2, edge_weight2, batch)

        z1 = self.mlp1(z1)
        z2 = self.mlp1(z2)

        z3 = self.mlp1(z3)
        z4 = self.mlp1(z4)

        g1 = self.mlp2(g1)
        g2 = self.mlp2(g2)

        return z1, g1, z2, g2, z3, z4

    def nt_xent_loss(self,
                     z1: torch.FloatTensor, g1: torch.FloatTensor,
                     z2: torch.FloatTensor, g2: torch.FloatTensor,
                     batch: torch.LongTensor, temperature: float,
                     fast_mode: bool = False
                     ):
        num_graphs = batch.max().item() + 1  # N := num_graphs
        num_nodes = z1.size()[0]  # M := num_nodes
        device = z1.device

        if fast_mode:
            values = torch.eye(num_nodes, dtype=torch.float32, device=device)  # [M, M]
            pos_mask = scatter(values, batch, dim=0, reduce='sum')  # [M, N]
        else:
            pos_mask = []
            for i in range(num_graphs):
                mask = batch == i
                pos_mask.append(mask.to(torch.long))
            pos_mask = torch.stack(pos_mask, dim=0).to(torch.float32)

        l1 = nt_xent_loss_with_mask(g1, z2, pos_mask=pos_mask, temperature=temperature)
        l2 = nt_xent_loss_with_mask(g2, z1, pos_mask=pos_mask, temperature=temperature)

        return l1 + l2

    def triplet_loss(self,
                     z1: torch.FloatTensor, g1: torch.FloatTensor,
                     z2: torch.FloatTensor, g2: torch.FloatTensor,
                     batch: torch.LongTensor, eps: float,
                     fast_mode: bool = False
                     ):
        num_graphs = batch.max().item() + 1  # N := num_graphs
        num_nodes = z1.size()[0]  # M := num_nodes
        device = z1.device

        if fast_mode:
            values = torch.eye(num_nodes, dtype=torch.float32, device=device)  # [M, M]
            pos_mask = scatter(values, batch, dim=0, reduce='sum')  # [M, N]
        else:
            pos_mask = []
            for i in range(num_graphs):
                mask = batch == i
                pos_mask.append(mask.to(torch.long))
            pos_mask = torch.stack(pos_mask, dim=0).to(torch.float32)

        l1 = triplet_loss(g1, z2, pos_mask=pos_mask, eps=eps)
        l2 = triplet_loss(g2, z1, pos_mask=pos_mask, eps=eps)

        return l1 + l2

    def loss(self, z1, g1, z2, g2, batch):
        num_graphs = g1.shape[0]
        num_nodes = z1.shape[0]
        device = z1.device

        pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
        for node_idx, graph_idx in enumerate(batch):
            pos_mask[node_idx][graph_idx] = 1.

        # l1 = jsd_loss(z1, g2, self.discriminator, pos_mask)
        # l2 = jsd_loss(z2, g1, self.discriminator, pos_mask)

        l1 = jsd_loss(g2, z1, self.discriminator, pos_mask.t())
        l2 = jsd_loss(g1, z2, self.discriminator, pos_mask.t())

        return l1 + l2

    def single_graph_triplet_loss(self,
                                  z1: torch.FloatTensor, g1: torch.FloatTensor,
                                  z2: torch.FloatTensor, g2: torch.FloatTensor,
                                  z3: torch.FloatTensor, z4: torch.FloatTensor,
                                  eps: float = 0.1
                                  ):
        anchor = torch.cat([g1, g2], dim=0)
        pos_samples = torch.stack([z2, z1], dim=0)
        neg_samples = torch.stack([z4, z3], dim=0)

        return multiple_triplet_loss(anchor, pos_samples, neg_samples, eps=eps)

    def single_graph_nt_xent_loss(self,
                                  z1: torch.FloatTensor, g1: torch.FloatTensor,
                                  z2: torch.FloatTensor, g2: torch.FloatTensor,
                                  z3: torch.FloatTensor, z4: torch.FloatTensor,
                                  temperature: float
                                  ):
        num_nodes = z1.size()[0]
        device = z1.device
        pos_mask_1 = torch.ones((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask_0 = torch.zeros((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask = torch.cat([pos_mask_1, pos_mask_0], dim=1)

        samples1 = torch.cat([z2, z4], dim=0)
        samples2 = torch.cat([z1, z3], dim=0)

        l1 = nt_xent_loss_with_mask(g1, samples1, pos_mask, temperature=temperature)
        l2 = nt_xent_loss_with_mask(g2, samples2, pos_mask, temperature=temperature)

        return l1 + l2

    def single_graph_loss(self,
                          z1: torch.FloatTensor, g1: torch.FloatTensor,
                          z2: torch.FloatTensor, g2: torch.FloatTensor,
                          z3: torch.FloatTensor, z4: torch.FloatTensor
                          ):
        num_nodes = z1.size(0)
        device = z1.device

        pos_mask_1 = torch.ones((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask_0 = torch.zeros((1, num_nodes), dtype=torch.float32, device=device)
        pos_mask = torch.cat([pos_mask_1, pos_mask_0], dim=1)
        # pos_mask = torch.repeat_interleave(pos_mask, repeats=2, dim=0)

        samples1 = torch.cat([z2, z4], dim=0)
        samples2 = torch.cat([z1, z3], dim=0)

        l1 = jsd_loss(g1, samples1, self.discriminator, pos_mask)
        l2 = jsd_loss(g2, samples2, self.discriminator, pos_mask)

        return l1 + l2
