from typing import *
from typing_extensions import Self
import numpy as np

from LUMA.Interface.Exception import NotFittedError
from LUMA.Interface.Super import _Estimator


__all__ = ['PolynomialRegressor']


class PolynomialRegressor(_Estimator):
    
    """Polynomial regression is a type of regression analysis used
    in statistics and machine learning to model the relationship between 
    a dependent variable and one or more independent variables."""
    
    def __init__(self, degree: int=1):
        self.degree = degree
        self._fitted = False

    def _generate_polynomial_features(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X_poly = self._generate_polynomial_features(X)
        self.coefficients = np.linalg.lstsq(X_poly, y, rcond=None)[0]
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        X_poly = self._generate_polynomial_features(X)
        return np.dot(X_poly, self.coefficients)

    def set_params(self, degree: int=None) -> None:
        self.degree = int(degree)

