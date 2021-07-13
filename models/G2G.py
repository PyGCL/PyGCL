import torch
import GCL.augmentors as A

from torch import nn
from typing import Optional, Tuple
from torch_geometric.nn import global_add_pool


class G2G(nn.Module):
    def __init__(self, encoder: torch.nn.Module,
                 augmentor: Tuple[A.Augmentor, A.Augmentor],
                 loss,
                 hidden_dim: int, proj_dim: int):
        super(G2G, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.loss = loss
        self.num_hidden = hidden_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)

        g1 = global_add_pool(z1, batch)
        g2 = global_add_pool(z2, batch)

        return z, z1, z2, g1, g2
