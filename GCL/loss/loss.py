import torch

from abc import ABC, abstractmethod
from GCL.model.sampler import ContrastInstance


class Loss(ABC):
    @abstractmethod
    def compute_default_positive(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def compute(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    def __call__(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        anchor, sample, pos_mask, neg_mask = contrast_instance
        if pos_mask is None and neg_mask is None:
            loss = self.compute_default_positive(contrast_instance, *args, **kwargs)
        else:
            loss = self.compute(contrast_instance, *args, **kwargs)
        return loss
