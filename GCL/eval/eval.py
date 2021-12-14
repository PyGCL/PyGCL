import torch
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict, Optional
from sklearn.base import BaseEstimator
from torch_geometric.data import Data
from sklearn.model_selection import GridSearchCV, BaseCrossValidator


def get_split(
        num_samples: int, num_splits: int = 1,
        train_ratio: float = 0.1, test_ratio: float = 0.8) -> Union[Dict, List[Dict]]:
    """
    Generate split indices for train, test, and validation sets.

    Args:
        num_samples (int): The size of the dataset.
        num_splits (int, optional): The number of splits to generate. (default: :obj:`1`)
        train_ratio (float, optional): The ratio of the train set. (default: :obj:`0.1`)
        test_ratio (float, optional): The ratio of the test set. (default: :obj:`0.8`)

    Returns:
        Union(Dict, List[Dict]): A dictionary of split indices or a list of dictionaries of split indices.

    Examples:
        >>> get_split(10, num_splits=1, train_ratio=0.5, test_ratio=0.4)
        [{'train': [3, 4, 0, 1, 2], 'test': [5, 7, 6, 8], 'valid': [9]}]
    """
    assert train_ratio + test_ratio < 1

    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    out = []
    for i in range(num_splits):
        indices = torch.randperm(num_samples)
        out.append({
            'train': indices[:train_size],
            'valid': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        })
    return out if num_splits > 1 else out[0]


def from_PyG_split(data: Data) -> Union[Dict, List[Dict]]:
    """
    Convert PyG split indices for train, test, and validation sets.

    Args:
        data (`Data`): The PyG data object.

    Returns:
        Union[Dict, List[Dict]]: A dictionary of split indices.

    Raises:
        ValueError: If the data object does not have the split indices.
    """
    if any([mask is None for mask in [data.train_mask, data.test_mask, data.val_mask]]):
        raise ValueError('The data object does not have the split indices.')
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)

    if data.train_mask.dim() == 1:
        return {
            'train': indices[data.train_mask],
            'valid': indices[data.val_mask],
            'test': indices[data.test_mask]
        }
    else:
        out = []
        for i in range(data.train_mask.size(1)):
            out_dict = {}
            for mask in ['train_mask', 'val_mask', 'test_mask']:
                if data[mask].dim() == 1:
                    # Datasets like WikiCS have only one split for the test set.
                    out_dict[mask[:-5]] = indices[data[mask]]
                else:
                    out_dict[mask[:-5]] = indices[data[mask][:, i]]
            out.append(out_dict)
        return out


class BaseEvaluator(ABC):
    """
    Base class for trainable (e.g., logistic regression) evaluation.

    Args:
        split (Union[Dict, List[Dict]]): The split indices.
        metrics (Dict[str, Callable]): The metrics to evaluate.
        stop_metric (Union[None, Callable, str], optional): The metric(s) to stop training.
            It could be a callable function, or a string specifying the key in :obj:`metric`.
            If set to :obj:`None`, the stopping metric will be set to the first in :obj:`metric`.
             (default: :obj:`None`)
        cv (BaseCrossValidator, optional): The sklearn cross-validator. (default: :obj:`None`)
    """

    def __init__(
            self, split: Union[Dict, List[Dict]],
            metrics: Dict[str, Callable], stop_metric: Union[None, Callable, int] = None,
            cv: Optional[BaseCrossValidator] = None):
        self.cv = cv
        self.split = split
        self.metrics = metrics
        if cv is None and stop_metric is None:
            stop_metric = 0
        if isinstance(stop_metric, str):
            self.stop_metric = metrics[stop_metric]
        else:
            self.stop_metric = stop_metric

    @abstractmethod
    def evaluate(self, x: Union[torch.FloatTensor, np.ndarray], y: Union[torch.LongTensor, np.ndarray]) -> Dict:
        raise NotImplementedError

    def __call__(self, x: Union[torch.FloatTensor, np.ndarray], y: Union[torch.LongTensor, np.ndarray]) -> Dict:
        result = self.evaluate(x, y)
        return result


class BaseSKLearnEvaluator:
    """
    Base class for sklearn-based evaluation.

    Args:
        evaluator (BaseEstimator): The sklearn evaluator.
        metrics (Dict[str, Callable]): The metric(s) to evaluate.
        split (BaseCrossValidator): The sklearn cross-validator to split the data.
        param_grid (List[Dict], optional): The parameter grid for the grid search. (default: :obj:`None`)
        grid_search_scoring (Dict[str, Callable], optional):
         If :obj:`param_grid` is given, provide metric(s) in grid search. (default: :obj:`None`)
        cv_params (Dict, optional): If :obj:`param_grid` is given, further pass the parameters
         for the sklearn cross-validator. See sklearn :obj:`GridSearchCV<https://scikit-learn.org/stable/modules/
         generated/sklearn.model_selection.GridSearchCV.html>`_ for details. (default: :obj:`None`)
    """

    def __init__(
            self, evaluator: BaseEstimator, metrics: Dict[str, Callable],
            split: BaseCrossValidator, param_grid: Optional[Dict] = None,
            grid_search_scoring: Optional[Dict[str, Callable]] = None,
            cv_params: Optional[Dict] = None):
        self.evaluator = evaluator
        self.split = split
        self.cv_params = cv_params
        self.grid_search_scoring = grid_search_scoring
        self.metrics = metrics
        self.param_grid = param_grid

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the model on the given data using sklearn evaluator.

        Args:
            x (np.ndarray): The data.
            y (np.ndarray): The targets (labels).

        Returns:
            Dict: The evaluation results with mean and standard deviation.
        """
        results = []
        for train_idx, test_idx in self.split.split(x, y):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            if self.param_grid is not None:
                predictor = GridSearchCV(
                    self.evaluator, self.param_grid, scoring=self.grid_search_scoring,
                    verbose=0, refit=next(iter(self.grid_search_scoring)))
                if self.cv_params is not None:
                    predictor.set_params(**self.cv_params)
            else:
                predictor = self.evaluator
            predictor.fit(x_train, y_train)
            y_pred = predictor.predict(x_test)
            results.append({name: metric(y_test, y_pred) for name, metric in self.metrics.items()})

        results = pd.DataFrame.from_dict(results)
        return results.agg(['mean', 'std']).to_dict()

    def __call__(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> Dict:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        result = self.evaluate(x, y)
        return result
