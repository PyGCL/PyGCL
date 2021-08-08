import torch
from abc import ABC, abstractmethod


__all__ = ['Loss']


class Loss(ABC):
    @abstractmethod
    def __compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        loss = self.__compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss
