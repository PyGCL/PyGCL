import torch

from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute_single_positive(self, contrast_instance, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def compute(self, contrast_instance, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss
