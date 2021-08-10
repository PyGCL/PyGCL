import torch
import GCL.augmentors as A

from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import global_add_pool


class EncoderModel(nn.Module):
    def __init__(self, encoder: torch.nn.Module,
                 augmentor: Tuple[A.Augmentor, A.Augmentor],
                 hidden_dim: int, proj_dim: int):
        super(EncoderModel, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.num_hidden = hidden_dim

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x: torch.FloatTensor, batch: torch.LongTensor,
                edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None):
        num_nodes = x.size()[0]

        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        shuffled_x1 = x1[torch.randperm(num_nodes)]
        shuffled_x2 = x2[torch.randperm(num_nodes)]

        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        z3 = self.encoder(shuffled_x1, edge_index1, edge_weight1)
        z4 = self.encoder(shuffled_x2, edge_index2, edge_weight2)

        g = global_add_pool(z, batch)
        g1 = global_add_pool(z1, batch)
        g2 = global_add_pool(z2, batch)

        return z, g, z1, z2, g1, g2, z3, z4

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
