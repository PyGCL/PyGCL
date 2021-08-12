from hypothesis import given
import torch
import torch.nn as nn
import numpy as np

import GCL.losses as L
import GCL.augmentors as A
from GCL import EncoderModel
from models.L2L import L2L
from models.GConv import Encoder

from testutils import tensor, edge_index


@given(x=tensor([10, 10]),
       edge_index=edge_index(10, 20))
def test_encoder_golden(x, edge_index):
    aug1 = A.Identity()
    aug2 = A.Identity()

    encoder = Encoder(
        10, 256,
        activation=nn.ReLU,
        batch_norm=False,
        num_layers=2,
        base_conv='GCNConv'
    )

    model1 = L2L(
        encoder=encoder,
        augmentor=(aug1, aug2),
        hidden_dim=256,
        proj_dim=256,
        loss=None
    )

    model2 = EncoderModel(
        encoder=encoder,
        augmentor=(aug1, aug2),
        hidden_dim=256,
        proj_dim=256
    )

    batch = torch.zeros((x.size()[0],), dtype=torch.long, device=x.device)

    a, a1, a2 = model1(x, edge_index)
    b, _, b1, b2, _, _, _, _ = model2(x, batch, edge_index, None)

    assert (a == b).all()
    assert (a1 == b2).all()
    assert (a1 == b2).all()
