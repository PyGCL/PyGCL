import torch
import torch.nn.functional as F

from .loss import Loss, ContrastInstance


class BootstrapLatent(Loss):
    def __init__(self):
        super(BootstrapLatent, self).__init__()

    def compute(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        anchor, sample, pos_mask = contrast_instance.anchor, contrast_instance.sample, contrast_instance.pos_mask
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample = F.normalize(sample, dim=-1, p=2)

        similarity = anchor @ sample.t()
        loss = (similarity * pos_mask).sum(dim=-1)
        return loss.mean()

    def compute_default_positive(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        anchor, sample = contrast_instance.anchor, contrast_instance.sample
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample = F.normalize(sample, dim=-1, p=2)

        similarity = anchor @ sample.t()
        loss = similarity.diag()
        return loss.mean()
