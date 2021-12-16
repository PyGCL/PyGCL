import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union, Callable, List, Dict, Optional, Type
from operator import itemgetter
from torch.optim import Optimizer
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, BaseCrossValidator

from GCL.eval.split import iter_split


class BaseTrainableEvaluator:
    """
    Base class for trainable (e.g., logistic regression) evaluation.

    Args:
        model (torch.nn.Module): The evaluation model to train. It should implement a :obj:`predict` function to
            convert logits to predictions.
        optimizer (Optimizer): The optimizer class for training.
        optimizer_params (Dict): The parameters for the optimizer.
        objective (Callable): The objective function to use for training. Its signature should be like
            :obj:`f(logits, y)`.
        split (Union[List[Dict], BaseCrossValidator]): A list of split indices (for multiple folds), or a
            sklearn cross-validator.
        metrics (Dict[str, Callable]): The metrics to evaluate in a dictionary with metric names as keys and
            callables as values.
        device (Union[str, torch.device]): The device to use for training. (default: :obj:`'cpu'`)
        num_epochs (int): The number of epochs to train the model. (default: :obj:`1000`)
        test_interval (int): The number of epochs between each test. (default: :obj:`20`)
        test_metric (Union[Callable, str], optional): The metric to test on the validation set during training.
            It could be a callable function, or a string specifying the key in :obj:`metrics`.
            If set to :obj:`None`, the test metric will be the first in :obj:`metrics`.
            If :obj:`split` is a sklearn cross-validator, this parameter is ignored as no validation set is used.
            (default: :obj:`None`)
    """

    def __init__(
            self, model: torch.nn.Module, optimizer: Type[Optimizer], optimizer_params: Dict,
            objective: Callable, split: Union[Dict, List[Dict], BaseCrossValidator],
            metrics: Dict[str, Callable], device: Union[str, torch.device] = 'cpu',
            num_epochs: int = 1000, test_interval: int = 20, test_metric: Union[Callable, str, None] = None):
        self.model = model
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.objective = objective
        self.split = split
        self.metrics = metrics
        self.device = device
        self.num_epochs = num_epochs
        self.test_interval = test_interval
        if isinstance(test_metric, str):
            self.test_metric = metrics[test_metric]
        else:
            self.test_metric = test_metric

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Dict]:
        """
        Evaluate the model on the given data by training another evaluator model.

        Args:
            x (torch.Tensor): The data.
            y (torch.Tensor): The targets (labels).

        Returns:
            Dict[str, Dict]: Evaluation results with metrics as keys and mean and standard deviation as values.
        """
        results = []
        for split_dict in iter_split(self.split, x, y):
            [v.to(self.device) for v in split_dict.values()]
            x_train, x_test, x_valid = itemgetter('x_train', 'x_test', 'x_valid')(split_dict)
            y_train, y_test, y_valid = itemgetter('y_train', 'y_test', 'y_valid')(split_dict)

            model = self.model.to(self.device)
            model.reset_paramerters()
            optimizer = self.optimizer_class(model.parameters(), **self.optimizer_params)
            criterion = self.objective

            best_val = 0
            best_test = {}

            with tqdm(total=self.num_epochs, desc='(ET)',
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
                for epoch in range(self.num_epochs):
                    model.train()
                    optimizer.zero_grad()
                    y_pred = model(x_train)
                    loss = criterion(y_pred, y_train)
                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % self.test_interval == 0:
                        model.eval()
                        with torch.no_grad():
                            y_pred = model.predict(x_valid).detach().cpu().numpy()
                            val_result = self.test_metric(y_pred, y_valid)
                            if val_result > best_val:
                                best_val = val_result
                                y_pred = model.predict(x_test).detach().cpu().numpy()
                                best_test = {k: v(y_pred, y_test) for k, v in self.metrics.items()}
                                best_model = self.model.state_dict().copy()
                        pbar.set_postfix(best_test)
                        pbar.update(self.test_interval)

            model.load_state_dict(best_model)
            model.eval()
            with torch.no_grad():
                y_pred = model.predict(x_test).detach().cpu().numpy()
                test_result = {k: v(y_pred, y_test) for k, v in self.metrics.items()}
            results.append(test_result)

        results = pd.DataFrame.from_dict(results)
        return results.agg(['mean', 'std']).to_dict()

    def __call__(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> Dict[str, Dict]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        x = x.detach().to(self.device)
        y = y.detach().to(self.device)
        result = self.evaluate(x, y)
        return result


class BaseSKLearnEvaluator:
    """
    Base class for sklearn-based evaluation.

    Args:
        evaluator (BaseEstimator): The sklearn evaluator.
        metrics (Dict[str, Callable]): The metrics to evaluate in a dictionary
            with metric names as keys and callables as values.
        split (Union[List[Dict], BaseCrossValidator]): A list of split indices (for multiple folds), or a
            sklearn cross-validator. If a list of indices is given, the validation set will be ignored.
        params (Dict, optional): Other parameters for the evaluator. (default: :obj:`None`)
        param_grid (List[Dict], optional): The parameter grid for the grid search. (default: :obj:`None`)
        grid_search_scoring (Dict[str, Callable], optional):
            If :obj:`param_grid` is given, provide metrics in grid search.
            If multiple metrics are given, the first one will be used to retrain the best model.
            (default: :obj:`None`)
        grid_search_params (Dict, optional): If :obj:`param_grid` is given, further pass the parameters
            for the sklearn grid search cross-validator. See sklearn `GridSearchCV
            <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
            for details. (default: :obj:`None`)
    """

    def __init__(
            self, evaluator: BaseEstimator, metrics: Dict[str, Callable],
            split: Union[List[Dict], BaseCrossValidator],
            params: Optional[Dict] = None, param_grid: Optional[Dict] = None,
            grid_search_scoring: Optional[Dict[str, Callable]] = None,
            grid_search_params: Optional[Dict] = None):
        if params is not None:
            evaluator.set_params(**params)
        self.evaluator = evaluator
        self.split = split
        self.grid_search_params = grid_search_params
        self.grid_search_scoring = grid_search_scoring
        self.metrics = metrics
        self.param_grid = param_grid

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate the model on the given data using sklearn evaluator.

        Args:
            x (np.ndarray): The data.
            y (np.ndarray): The targets (labels).

        Returns:
            Dict[str, Dict]: Evaluation results with metrics as keys and mean and standard deviation as values.
        """
        results = []
        for split_dict in iter_split(self.split, x, y):
            x_train, x_test = itemgetter('x_train', 'x_test')(split_dict)
            y_train, y_test = itemgetter('y_train', 'y_test')(split_dict)
            if self.param_grid is not None:
                predictor = GridSearchCV(
                    self.evaluator, self.param_grid, scoring=self.grid_search_scoring,
                    verbose=0, refit=next(iter(self.grid_search_scoring)))
                if self.grid_search_params is not None:
                    predictor.set_params(**self.grid_search_params)
            else:
                predictor = self.evaluator
            predictor.fit(x_train, y_train)
            y_pred = predictor.predict(x_test)
            results.append({name: metric(y_test, y_pred) for name, metric in self.metrics.items()})

        results = pd.DataFrame.from_dict(results)
        return results.agg(['mean', 'std']).to_dict()

    def __call__(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> Dict[str, Dict]:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        result = self.evaluate(x, y)
        return result
