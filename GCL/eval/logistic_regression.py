import torch

from tqdm import tqdm
from torch import nn
from typing import Union, List, Callable, Dict, Optional
from torch.optim import Adam
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator

from GCL.eval import BaseEvaluator, BaseSKLearnEvaluator


class LRModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LRModel, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LRModel(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }


class LRSklearnEvaluator(BaseSKLearnEvaluator):
    """
    Evaluation using the sklearn logistic regression classifier.

    Parameters:
        metrics (Dict[str, Callable]): The metric(s) to evaluate.
        split (BaseCrossValidator): The sklearn cross-validator to split the data.
        params (Dict, optional): Other parameters for the logistic regression model.
         See sklearn :obj:`LogisticRegression<https://scikit-learn.org/stable/modules/generated/
         sklearn.linear_model.LogisticRegression.html>`_ for details. (default: :obj:`None`)
        param_grid (List[Dict], optional): The parameter grid for the grid search. (default: :obj:`None`)
        grid_search_scoring (Dict[str, Callable], optional):
         If :obj:`param_grid` is given, provide metric(s) in grid search. (default: :obj:`None`)
        cv_params (Dict, optional): If :obj:`param_grid` is given, further pass the parameters
         for the sklearn cross-validator. See sklearn :obj:`GridSearchCV<https://scikit-learn.org/stable/modules/
         generated/sklearn.model_selection.GridSearchCV.html>`_ for details. (default: :obj:`None`)
    """
    def __init__(
            self, metrics: Dict[str, Callable], split: BaseCrossValidator,
            params: Optional[Dict] = None, param_grid: Optional[Dict] = None,
            grid_search_scoring: Optional[Dict[str, Callable]] = None,
            cv_params: Optional[Dict] = None):
        super(LRSklearnEvaluator, self).__init__(
            evaluator=LogisticRegression(), metrics=metrics, split=split, params=params,
            param_grid=param_grid, grid_search_scoring=grid_search_scoring, cv_params=cv_params)
