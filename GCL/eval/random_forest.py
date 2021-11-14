from sklearn.ensemble import RandomForestClassifier
from GCL.eval import BaseSKLearnEvaluator


class RFEvaluator(BaseSKLearnEvaluator):
    def __init__(self, param_grid=None, cv=None, scoring=None, refit=None):
        if param_grid is None:
            param_grid = {'n_estimators': [100, 200, 500, 1000]}
        super(RFEvaluator, self).__init__(RandomForestClassifier(), param_grid, cv, scoring, refit)
