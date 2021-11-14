import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict, Optional
from sklearn.model_selection import GridSearchCV, BaseCrossValidator


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


class BaseEvaluator(ABC):
    def __init__(
            self, split: Union[Dict, List[Dict]],
            metric: Union[Callable, List[Callable]], stop_metric: Union[None, Callable, int] = None,
            cv: Optional[BaseCrossValidator] = None):
        self.cv = cv
        self.split = split
        self.metric = metric
        if cv is None and stop_metric is None:
            stop_metric = 0
        if isinstance(stop_metric, int):
            if isinstance(metric, list):
                self.stop_metric = metric[stop_metric]
            else:
                raise ValueError
        else:
            self.stop_metric = stop_metric

    @abstractmethod
    def evaluate(self, x: Union[torch.FloatTensor, np.ndarray], y: Union[torch.LongTensor, np.ndarray]) -> dict:
        raise NotImplementedError

    def __call__(self, x: Union[torch.FloatTensor, np.ndarray], y: Union[torch.LongTensor, np.ndarray]) -> dict:
        result = self.evaluate(x, y)
        return result


class BaseSKLearnEvaluator:
    def __init__(self, evaluator, param_grid, cv, scoring, refit):
        self.evaluator = evaluator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    def evaluate(self, x, y):
        classifier = GridSearchCV(
            self.evaluator, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring, refit=self.refit,
            verbose=0, return_train_score=True)
        classifier.fit(x, y)
        return classifier.cv_results_
