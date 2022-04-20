import torch

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class ContrastInstance:
    anchor: torch.Tensor
    sample: torch.Tensor
    pos_mask: Optional[torch.Tensor] = None
    neg_mask: Optional[torch.Tensor] = None

    def unpack(self):
        return self.anchor, self.sample, self.pos_mask, self.neg_mask

    def masks(self):
        return self.pos_mask, self.neg_mask


class Loss(ABC):
    @abstractmethod
    def compute_default_positive(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def compute(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    def __call__(self, contrast_instance: ContrastInstance, *args, **kwargs) -> torch.FloatTensor:
        pos_mask, neg_mask = contrast_instance.masks()
        if pos_mask is None and neg_mask is None:
            loss = self.compute_default_positive(contrast_instance, *args, **kwargs)
        else:
            loss = self.compute(contrast_instance, *args, **kwargs)
        return loss
