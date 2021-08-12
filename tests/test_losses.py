from hypothesis import given
import torch

import GCL.losses as L

from testutils import tensor


@given(anchor=tensor((3, 10)),
       sample=tensor((3, 10)))
def test_infonce_positive(anchor: torch.FloatTensor, sample: torch.FloatTensor):
    num_samples = anchor.size()[0]
    pos_mask = torch.eye(num_samples, dtype=torch.float32)
    neg_mask = 1. - pos_mask

    loss_fn = L.InfoNCELoss(tau=0.1)
    loss = loss_fn(anchor, sample, pos_mask, neg_mask)

    assert loss.item() > 0.0


@given(anchor=tensor((3, 10)),
       sample=tensor((3, 10)))
def test_infonce_fast_golden(anchor: torch.FloatTensor, sample: torch.FloatTensor):
    import torch.nn.functional as F
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    def nt_xent_loss_en(anchor: torch.FloatTensor,
                        samples: torch.FloatTensor,
                        pos_mask: torch.FloatTensor,
                        tau: float):
        f = lambda x: torch.exp(x / tau)
        sim = f(_similarity(anchor, samples))  # anchor x sample
        assert sim.size() == pos_mask.size()  # sanity check

        neg_mask = 1 - pos_mask
        pos = (sim * pos_mask).sum(dim=1)
        neg = (sim * neg_mask).sum(dim=1)

        loss = pos / (pos + neg)
        loss = -torch.log(loss)

        return loss.mean()

    num_samples = anchor.size()[0]
    pos_mask = torch.eye(num_samples, dtype=torch.float32)
    neg_mask = 1. - pos_mask

    loss_fn = L.InfoNCELoss(tau=0.1)

    loss1 = loss_fn(anchor, sample, pos_mask, neg_mask)
    loss2 = nt_xent_loss_en(anchor, sample, pos_mask, 0.1)

    assert loss1.item() == loss2.item()
