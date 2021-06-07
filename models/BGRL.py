import copy
from typing import Tuple
import torch
from torch_geometric.nn import global_add_pool

import GCL.augmentations as A


class BGRL(torch.nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 augmentation: Tuple[A.GraphAug, A.GraphAug],
                 hidden_dim: int,
                 dropout: float = 0.2):
        super(BGRL, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentation = augmentation

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None):
        aug1, aug2 = self.augmentation
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h1_pred = self.predictor(h1_online)

        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


class BGRL_G2G(torch.nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 augmentation: Tuple[A.GraphAug, A.GraphAug],
                 hidden_dim: int,
                 dropout: float = 0.2):
        super(BGRL_G2G, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentation = augmentation

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x: torch.FloatTensor, batch, edge_index: torch.LongTensor, edge_weight=None):
        aug1, aug2 = self.augmentation
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        g1 = global_add_pool(h1, batch)
        g1_online = global_add_pool(h1_online, batch)
        g1_pred = self.predictor(g1_online)

        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)
        g2 = global_add_pool(h2, batch)
        g2_online = global_add_pool(h2_online, batch)
        g2_pred = self.predictor(g2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, g1_pred, g2_pred, g1_target, g2_target


class BGRL_G2L(torch.nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 augmentation: Tuple[A.GraphAug, A.GraphAug],
                 hidden_dim: int,
                 dropout: float = 0.2):
        super(BGRL_G2L, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentation = augmentation

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x: torch.FloatTensor, batch, edge_index: torch.LongTensor, edge_weight=None):
        aug1, aug2 = self.augmentation
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_online)

        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)
        g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target
