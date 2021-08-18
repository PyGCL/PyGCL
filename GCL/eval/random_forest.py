from sklearn.ensemble import RandomForestClassifier
from GCL.eval import BaseSKLearnEvaluator


class RFEvaluator(BaseSKLearnEvaluator):
    def __init__(self, params=None):
        if params is None:
            params = {'n_estimators': [100, 200, 500, 1000]}
        super(RFEvaluator, self).__init__(RandomForestClassifier(), params)
