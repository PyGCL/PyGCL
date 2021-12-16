import torch

from torch import nn
from typing import Union, List, Callable, Dict, Optional
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator

from GCL.eval import BaseTrainableEvaluator, BaseSKLearnEvaluator


class LRModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LRModel, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        self.output_fn = nn.LogSoftmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        return self.output_fn(self.fc(x))

    def predict(self, x):
        return self.fc(x)


class LRTrainableEvaluator(BaseTrainableEvaluator):
    """
    Evaluate using a trainable logistic regression model.

    Args:
        input_dim (int): The dimension of the input data.
        num_classes (int): The number of classes.
        split (Union[List[Dict], BaseCrossValidator]): A list of split indices (for multiple folds), or a
            sklearn cross-validator.
        metrics (Dict[str, Callable]): The metrics to evaluate in a dictionary with metric names as keys and
            callables as values.
        device (Union[str, torch.device]): The device to use for training. (default: :obj:`'cpu'`)
        num_epochs (int): The number of epochs to train. (default: :obj:`5000`)
        learning_rate (float): The learning rate for the optimizer. (default: :obj:`0.01`)
        weight_decay (float): The weight decay for the optimizer. (default: :obj:`0.0`)
        test_interval (int): The number of epochs between each test. (default: :obj:`20`)
        test_metric (Union[Callable, str], optional): The metric to test on the validation set during training.
            It could be a callable function, or a string specifying the key in :obj:`metrics`.
            If set to :obj:`None`, the test metric will be the first in :obj:`metrics`.
            If :obj:`split` is a sklearn cross-validator, this parameter is ignored as no validation set is used.
            (default: :obj:`None`)
    """

    def __init__(
            self, input_dim: int, num_classes: int,
            split: Union[Dict, List[Dict], BaseCrossValidator],
            metrics: Dict[str, Callable], device: Union[str, torch.device] = 'cpu',
            num_epochs: int = 5000, learning_rate: float = 0.01, weight_decay: float = 0.0,
            test_interval: int = 20, test_metric: Union[Callable, str, None] = None):
        model = LRModel(input_dim, num_classes)
        optimizer = Adam
        optimizer_params = {'lr': learning_rate, 'weight_decay': weight_decay}
        criterion = nn.NLLLoss()
        super(LRTrainableEvaluator).__init__(
            model=model, optimizer=optimizer, optimizer_params=optimizer_params,
            objective=criterion, split=split, metrics=metrics, device=device,
            num_epochs=num_epochs, test_interval=test_interval, test_metric=test_metric)


class LRSklearnEvaluator(BaseSKLearnEvaluator):
    """
    Evaluate using the sklearn logistic regression classifier.

    Parameters:
        metrics (Dict[str, Callable]): The metrics to evaluate in a dictionary
            with metric names as keys and callables as values.
        split (BaseCrossValidator): The sklearn cross-validator to split the data.
        params (Dict, optional): Other parameters for the logistic regression model.
            See sklearn `LogisticRegression
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
            for details. (default: :obj:`None`)
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
            self, metrics: Dict[str, Callable], split: BaseCrossValidator,
            params: Optional[Dict] = None, param_grid: Optional[Dict] = None,
            grid_search_scoring: Optional[Dict[str, Callable]] = None,
            grid_search_params: Optional[Dict] = None):
        super(LRSklearnEvaluator, self).__init__(
            evaluator=LogisticRegression(), metrics=metrics, split=split, params=params,
            param_grid=param_grid, grid_search_scoring=grid_search_scoring, grid_search_params=grid_search_params)
