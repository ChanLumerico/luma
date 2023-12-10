from typing import *
import numpy as np

from luma.interface.exception import NotFittedError
from luma.interface.super import Estimator, Evaluator
from luma.metric.regression import MeanSquaredError


__all__ = ['PolynomialRegressor']


class PolynomialRegressor(Estimator):
    
    """Polynomial regression is a type of regression analysis used
    in statistics and machine learning to model the relationship between 
    a dependent variable and one or more independent variables."""
    
    def __init__(self, degree: int = 1):
        self.degree = degree
        self._fitted = False

    def _generate_polynomial_features(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegressor':
        X_poly = self._generate_polynomial_features(X)
        self.coefficients = np.linalg.lstsq(X_poly, y, rcond=None)[0]
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        X_poly = self._generate_polynomial_features(X)
        return np.dot(X_poly, self.coefficients)

    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)

    def set_params(self, degree: int = None) -> None:
        self.degree = int(degree)

