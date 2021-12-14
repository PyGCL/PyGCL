from typing import Callable, Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator

from GCL.eval import BaseSKLearnEvaluator


class RFEvaluator(BaseSKLearnEvaluator):
    """
    Evaluation using the random forest classifier.

    Parameters:
        metrics (Dict[str, Callable]): The metric(s) to evaluate.
        split (BaseCrossValidator): The sklearn cross-validator to split the data.
        params (Dict, optional): Other parameters for the random forest classifier.
         See sklearn :obj:`RandomForestClassifier<https://scikit-learn.org/stable/modules/generated/
         sklearn.ensemble.RandomForestClassifier.html>`_ for details. (default: :obj:`None`)
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
        super(RFEvaluator, self).__init__(
            evaluator=RandomForestClassifier(), metrics=metrics, split=split, params=params,
            param_grid=param_grid, grid_search_scoring=grid_search_scoring, cv_params=cv_params)
