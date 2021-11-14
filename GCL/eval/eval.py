import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict, Optional
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, BaseCrossValidator


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


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


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
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor) -> dict:
        raise NotImplementedError

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor) -> dict:
        result = self.evaluate(x, y)
        return result


class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

        return {
            'micro_f1': test_micro,
            'macro_f1': test_macro,
        }
