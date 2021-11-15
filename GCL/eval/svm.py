from sklearn.svm import LinearSVC, SVC
from GCL.eval import BaseSKLearnEvaluator


class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, param_grid=None, cv=None, metric=None, refit=None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        if param_grid is None:
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, param_grid, cv, metric, refit)
