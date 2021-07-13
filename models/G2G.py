import torch
import torch.nn.functional as F
import GCL.augmentors as A

from torch import nn
from typing import Optional, Tuple
from torch_geometric.nn import GINConv, global_add_pool


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


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation, num_layers: int, batch_norm: bool = False):
        super(GraphEncoder, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if batch_norm else None
        self.layers.append(make_gin_conv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            # add batch norm layer if batch norm is used
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        num_layers = len(self.layers)
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            if self.batch_norms is not None and i != num_layers - 1:
                z = self.batch_norms[i](z)
        return z
