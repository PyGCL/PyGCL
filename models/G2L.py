import torch
from typing import Callable, Tuple, Optional
from torch_geometric import nn

from GCL.augmentors import Augmentor

from models.GConv import make_gin_conv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, readout, base_conv='GINConv'):
        super(GCN, self).__init__()
        self.activation = activation()
        self.readout = readout
        self.layers = torch.nn.ModuleList()
        if base_conv == 'GINConv':
            self.layers.append(make_gin_conv(input_dim, hidden_dim))
        else:
            self.layers.append(nn.GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            if base_conv == 'GINConv':
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            else:
                self.layers.append(nn.GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weights, batch):
        z = x
        g = []
        for conv in self.layers:
            z = conv(z, edge_index, edge_weights)
            z = self.activation(z)
            if self.readout == 'mean':
                g.append(nn.global_mean_pool(z, batch))
            elif self.readout == 'max':
                g.append(nn.global_max_pool(z, batch))
            elif self.readout == 'sum':
                g.append(nn.global_add_pool(z, batch))
        g = torch.cat(g, dim=1)
        return z, g


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class G2L(torch.nn.Module):
    def __init__(self, gnn1: torch.nn.Module, gnn2: torch.nn.Module,
                 mlp1: torch.nn.Module, mlp2: torch.nn.Module,
                 augmentor: Tuple[Augmentor, Augmentor], loss,
                 discriminator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: x @ y.t()):
        super(G2L, self).__init__()
        self.gnn1 = gnn1
        self.gnn2 = gnn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.loss = loss
        self.augmentor = augmentor
        self.discriminator = discriminator

    def forward(self, batch, x: Optional[torch.Tensor],
                edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(batch.device)

        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        num_nodes = x.size(0)
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
