from typing import Literal, Self
import numpy as np

from luma.interface.typing import Matrix
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator
from luma.metric.regression import MeanSquaredError


__all__ = "PolynomialRegressor"


class PolynomialRegressor(Estimator):
    """
    Polynomial regression is a type of regression analysis used
    in statistics and machine learning to model the relationship between
    a dependent variable and one or more independent variables.

    Parameters
    ----------
    `deg` : int, default=2
        Degree of a polynomial function
    `alpha` : float, default=0.01
        Regularization strength
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter between L1 and L2 regularization
    `regularization` : {"l1", "l2", "elastic-net"}, optional, default=None
        Regularization type

    """

    def __init__(
        self,
        deg: int = 2,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        regularization: Literal["l1", "l2", "elastic-net"] | None = None,
    ) -> None:
        self.deg = deg
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.regularization = regularization
        self._fitted = False

        self.set_param_ranges(
            {
                "deg": ("0<,+inf", int),
                "alpha": ("0,+inf", None),
                "l1_ratio": ("0,1", None),
            }
        )
        self.check_param_ranges()

    def _generate_polynomial_features(self, X: Matrix) -> Matrix:
        X_poly = X.copy()
        for d in range(2, self.deg + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X_poly = self._generate_polynomial_features(X)
        reg_matrix = self._regularization_matrix(X_poly.shape[1])

        augmented_matrix = np.vstack([X_poly, np.sqrt(self.alpha) * reg_matrix])
        augmented_target = np.hstack([y, np.zeros(X_poly.shape[1])])

        self.coef_ = np.linalg.lstsq(augmented_matrix, augmented_target, rcond=None)[0]
        self._fitted = True
        return self

    def _regularization_matrix(self, n: int) -> Matrix:
        if self.regularization == "l2":
            return self.alpha * np.eye(n)
        elif self.regularization == "l1":
            return self.alpha * np.sign(np.random.randn(n, n))
        elif self.regularization == "elastic-net":
            l1_mat = self.alpha * self.l1_ratio * np.sign(np.random.randn(n, n))
            l2_mat = self.alpha * (1 - self.l1_ratio) * np.eye(n)
            return l1_mat + l2_mat
        else:
            return np.zeros((n, n))

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X_poly = self._generate_polynomial_features(X)
        return np.dot(X_poly, self.coef_)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
