from typing import Literal
import numpy as np

from luma.interface.util import Matrix
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator
from luma.metric.regression import MeanSquaredError


__all__ = (
    'PolynomialRegressor'
)


class PolynomialRegressor(Estimator):
    
    """
    Polynomial regression is a type of regression analysis used
    in statistics and machine learning to model the relationship between 
    a dependent variable and one or more independent variables.
    
    Parameters
    ----------
    `deg` : Degree of a polynomial function
    `alpha` : Regularization strength
    `l1_ratio` : Balancing parameter between `l1` and `l2`
    `regularization` : Regularization type (e.g. `l1`, `l2`, `elastic-net`)
    
    """
    
    def __init__(self, 
                 deg: int = 2,
                 alpha: float = 0.01,
                 l1_ratio: float = 0.5,
                 regularization: Literal['l1', 'l2', 'elastic-net'] = None) -> None:
        self.deg = deg
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.regularization = regularization
        self._fitted = False

    def _generate_polynomial_features(self, X):
        X_poly = X.copy()
        for d in range(2, self.deg + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X: Matrix, y: Matrix) -> 'PolynomialRegressor':
        X_poly = self._generate_polynomial_features(X)
        reg_matrix = self._regularization_matrix(X_poly.shape[1])
        augmented_matrix = np.vstack([X_poly, np.sqrt(self.alpha) * reg_matrix])
        augmented_target = np.hstack([y, np.zeros(X_poly.shape[1])])
        self.coef_ = np.linalg.lstsq(augmented_matrix, augmented_target, rcond=None)[0]
        self._fitted = True
        return self

    def _regularization_matrix(self, n: int) -> Matrix:
        if self.regularization == 'l2':
            return self.alpha * np.eye(n)
        elif self.regularization == 'l1':
            return self.alpha * np.sign(np.random.randn(n, n))
        elif self.regularization == 'elastic-net':
            l1_mat = self.alpha * self.l1_ratio * np.sign(np.random.randn(n, n))
            l2_mat = self.alpha * (1 - self.l1_ratio) * np.eye(n)
            return l1_mat + l2_mat
        else:
            return np.zeros((n, n))
    
    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        X_poly = self._generate_polynomial_features(X)
        return np.dot(X_poly, self.coef_)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

