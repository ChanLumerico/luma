import numpy as np
import pandas as pd

from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector, Clone
from luma.interface.exception import NotFittedError
from luma.classifier.tree import DecisionTreeClassifier


class AdaBoostClassifier(Estimator, Supervised):
    def __init__(self, 
                 base_estimator: DecisionTreeClassifier(max_depth=1),
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 random_state: int= None,
                 verbose: bool = False) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'AdaBoostClassifier':
        m, _ = X.shape
        for i in range(self.n_estimators):
            if i == 0: w = np.ones(m) * (1 / m)
            else: w = self._update_weights(...)
            
            estimator = Clone(self.base_estimator).get
            estimator.fit(X, y, sample_weight=w)
    
    def _update_weights(self, 
                        alpha: Vector, 
                        w: Vector, 
                        y: Vector, 
                        y_pred: Vector) -> Vector:
        w *= np.exp(alpha * (np.not_equal(y, y_pred).astype(int)))
        return w

