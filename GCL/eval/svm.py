from typing import Union, Callable, List, Optional, Dict
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import BaseCrossValidator

from GCL.eval import BaseSKLearnEvaluator


class SVMEvaluator(BaseSKLearnEvaluator):
    """
    Evaluation using SVM.

    Parameters:
        linear (bool): Whether to use linear SVM. (default: :obj:`True`)
        metrics (Dict[str, Callable]): The metric(s) to evaluate.
        split (BaseCrossValidator): The sklearn cross-validator, used to split the data.
        param_grid (List[Dict], optional): The parameter grid for the grid search. (default: :obj:`None`)
        grid_search_scoring (Dict[str, Callable]): The metric(s) used in grid search. (default: :obj:`None`)
        cv_params (Dict, optional): Pass the parameters for the sklearn cross-validator.
         See sklearn :obj:`GridSearchCV<https://scikit-learn.org/stable/modules/
         generated/sklearn.model_selection.GridSearchCV.html>`_ for details. (default: :obj:`None`)
    """
    def __init__(
            self, metrics: Dict[str, Callable], split: BaseCrossValidator,
            linear=True, param_grid: Optional[Dict] = None,
            grid_search_scoring: Optional[Dict[str, Callable]] = None,
            cv_params: Optional[Dict] = None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        super(SVMEvaluator, self).__init__(
            evaluator=self.evaluator, metrics=metrics, split=split,
            param_grid=param_grid, grid_search_scoring=grid_search_scoring, cv_params=cv_params)
