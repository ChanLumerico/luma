from typing import Any, List, Literal, Dict
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.interface.util import Matrix, Vector
from luma.classifier.logistic import SoftmaxRegressor
from luma.metric.classification import Accuracy
from luma.model_selection.fold import CrossValidator


__all__ = (
    'StackingClassifier'
)


class StackingClassifier(Estimator, Estimator.Meta, Supervised):
    
    def __init__(self, 
                 estimators: List[Estimator],
                 final_estimator: Estimator = SoftmaxRegressor(), 
                 pass_original: bool = False,
                 drop_threshold: float = None, 
                 verbose: bool = False, 
                 **kwargs: Dict[str, Any]) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.pass_original = pass_original
        self.drop_threshold = drop_threshold
        self.verbose = verbose
        self._final_estimator_params = kwargs
        self._base_estimators: List[Estimator] = []
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'StackingClassifier':
        m, _ = X.shape
        X_new = np.zeros((m, len(self.estimators)))
        for i, est in enumerate(self.estimators):
            est.fit(X, y)
            X_new[:, i] = est.predict(X)
            score = est.score(X, y)
            
            if self.drop_threshold is not None and score < self.drop_threshold:
                if self.verbose:
                    print(f'[StackClassifier] {type(est).__name__} dropped',
                          f'for low score: {score}')
                continue
            
            self._base_estimators.append(est)
            if self.verbose:
                print(f'[StackClassifier] Finished fitting for',
                      f'{type(est).__name__} with score: {score}')
        
        if self.pass_original:
            X_new = np.hstack((X, X_new))
        
        self.final_estimator.set_params(**self._final_estimator_params)
        self.final_estimator.fit(X_new, y)
        
        if self.verbose:
            print(f'[StackClassifier] Finished fitting for',
                  f'{type(self.final_estimator).__name__}(final)')
        
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        X_new = np.zeros((m, len(self._base_estimators)))
        for i, est in enumerate(self._base_estimators):
            X_new[:, i] = est.predict(X)
            
        if self.pass_original:
            X_new = np.hstack((X, X_new))
        
        return self.final_estimator.predict(X_new)
    
    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

