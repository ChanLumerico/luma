from typing import Literal, Self
from scipy.special import psi
import numpy as np

from luma.interface.typing import Matrix
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.metric.regression import MeanSquaredError


__all__ = (
    "PoissonRegressor",
    "NegativeBinomialRegressor",
    "GammaRegressor",
    "BetaRegressor",
    "InverseGaussianRegressor",
)


class PoissonRegressor(Estimator, Supervised):
    """
    Poisson regression is a model for predicting count or frequency data.
    It models the relationship between features and counts, with a focus
    on count data that follows a Poisson distribution. The model uses
    the natural logarithm of expected counts as a link function and is
    trained to minimize the negative log-likelihood.

    Parameters
    ----------
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter of `elastic-net`
    `alpha` : float, default=0.01
        Regularization strength
    `regularization` : {"l1", "l2", "elastic-net"}, optional, default=None
        Regularization type

    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        regularization: Literal["l1", "l2", "elastic-net"] | None = None,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "l1_ratio": ("0,1", None),
                "alpha": ("0<,+inf", None),
            }
        )
        self.check_param_ranges()

    def link_funciton(self, X: Matrix) -> Matrix:
        return np.exp(np.dot(X, self.weights))

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        m, n = X.shape
        self.weights = np.zeros(n)

        for i in range(self.max_iter):
            predictions = self.link_funciton(X)
            gradient = np.dot(X.T, y - predictions) * self.learning_rate
            gradient /= m

            if self.regularization:
                regularization_term = self._regularization_term()
                gradient -= self.alpha * regularization_term / m

            if self.verbose and i % 10 == 0:
                print(f"[PoissonReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {np.linalg.norm(gradient)}")

            self.weights += gradient
        self._fitted = True
        return self

    def _regularization_term(self) -> Matrix:
        if self.regularization == "l1":
            return np.sign(self.weights)
        elif self.regularization == "l2":
            return self.weights
        elif self.regularization == "elastic-net":
            l1_term = np.sign(self.weights)
            l2_term = 2 * self.weights
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(self.weights)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        predictions = self.link_funciton(X)
        return predictions

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class NegativeBinomialRegressor(Estimator, Supervised):
    """
    Negative Binomial Regression is a statistical method used to
    model count data, like the number of events or occurrences,
    when the data shows overdispersion (variance is greater than mean)
    compared to the Poisson distribution. It estimates how predictors
    influence the count while allowing for flexible variance modeling.

    Parameters
    ----------
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter of `elastic-net`
    `alpha` : float, default=0.01
        Regularization strength
    `regularization` : {"l1", "l2", "elastic-net"}, optional, default=None
        Regularization type
    `phi` : float, default=1.0
        Dispersion parameter

    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        phi: float = 1.0,
        regularization: Literal["l1", "l2", "elastic-net"] = None,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.phi = phi
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "alpha": ("0,+inf", None),
                "l1_ratio": ("0,1", None),
                "phi": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def link_function(self, X: Matrix) -> Matrix:
        return np.log(1 + np.exp(np.dot(X, self.weights)))

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        m, n = X.shape
        self.weights = np.zeros(n)

        for i in range(self.max_iter):
            mu = self.link_function(X)
            gradient = (y - mu) / (self.phi * (1 + mu / self.phi))
            gradient = np.dot(X.T, gradient) * self.learning_rate
            gradient /= m

            if self.regularization:
                regularization_term = self._regularization_term()
                gradient -= self.alpha * regularization_term / m

            if self.verbose and i % 10 == 0:
                print(f"[NegBinReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {np.linalg.norm(gradient)}")

            self.weights += gradient
        self._fitted = True
        return self

    def _regularization_term(self) -> Matrix:
        if self.regularization == "l1":
            return np.sign(self.weights)
        elif self.regularization == "l2":
            return self.weights
        elif self.regularization == "elastic-net":
            l1_term = np.sign(self.weights)
            l2_term = 2 * self.weights
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(self.weights)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        mu = self.link_function(X)
        return mu

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class GammaRegressor(Estimator, Supervised):
    """
    Gamma regression is a statistical method for analyzing
    data with skewed, positively skewed distributions and where
    the variance increases with the mean. It models the
    relationship between variables using the gamma distribution,
    making it suitable for situations where traditional
    linear regression doesn't apply.

    Parameters
    ----------
    `alpha`, `beta` : float, default=1.0
        Shape parameter of gamma distribution
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `reg_strength` : float, default=0.01
        Regularization strength
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter of `elastic-net`
    `regularization` : {"l1", "l2", "elastic-net"}, optional, default=None
        Regularization type

    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        reg_strength: float = 0.01,
        l1_ratio: float = 0.5,
        regularization: Literal["l1", "l2", "elastic-net"] = None,
        verbose: bool = False,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "reg_strength": ("0,+inf", None),
                "l1_ratio": ("0,1", None),
                "alpha": ("-inf,inf", None),
                "beta": ("-inf,inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        m, n = X.shape
        self.alpha = np.ones(n) * self.alpha
        self.beta = np.ones(n) * self.beta

        for i in range(self.max_iter):
            gradient = (self.alpha - 1) / self.alpha - psi(self.alpha)
            gradient += np.log(self.beta) - (y / self.beta)

            self.alpha += self.learning_rate * np.mean(gradient * X, axis=0) / m
            self.beta += (
                self.learning_rate
                / m
                * np.mean((self.alpha / self.beta**2 - y / self.beta) * X, axis=0)
            )

            if self.regularization:
                self.alpha -= (
                    self.reg_strength * self._regularization_term(self.alpha) / m
                )
                self.beta -= (
                    self.reg_strength * self._regularization_term(self.beta) / m
                )

            if self.verbose and i % 10 == 0:
                print(f"[GammaReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {np.linalg.norm(gradient)}")

        self._fitted = True
        return self

    def _regularization_term(self, weights: Matrix) -> Matrix:
        if self.regularization == "l1":
            return np.sign(weights)
        elif self.regularization == "l2":
            return self.weights
        elif self.regularization == "elastic-net":
            l1_term = np.sign(weights)
            l2_term = 2 * weights
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(weights)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return np.dot(X, self.alpha / self.beta)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class BetaRegressor(Estimator, Supervised):
    """
    Beta regression is a statistical model used in machine learning to
    analyze and predict data that follows a bounded interval between
    0 and 1, such as proportions, probabilities, or percentages.
    It's particularly useful for modeling data with heteroscedasticity,
    where the spread of the data varies across different input values.

    Parameters
    ----------
    `alpha`, `beta` : float, default=1.0
        Shape parameters of beta function
    `learning_rate` : float, default=1.0
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `reg_strength` : float, default=0.01
        Regularization strength
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balacing parameter of `elastic-net`
    `regularization` : {"l1", "l2", "elastic-net"}, optional, default=None
        Regularization type

    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        reg_strength: float = 0.01,
        l1_ratio: float = 0.5,
        regularization: Literal["l1", "l2", "elastic-net"] = None,
        verbose: bool = False,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "reg_strength": ("0,+inf", None),
                "l1_ratio": ("0,1", None),
                "alpha": ("-inf,inf", None),
                "beta": ("-inf,inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        m, n = X.shape
        self.alpha = np.ones(n) * self.alpha
        self.beta = np.ones(n) * self.beta

        for i in range(self.max_iter):
            gradient_alpha = psi(self.alpha) - psi(self.alpha + self.beta)
            gradient_alpha += np.log(self.beta) - np.log(1 - y)
            gradient_beta = psi(self.beta) - psi(self.alpha + self.beta)
            gradient_beta += np.log(self.alpha) - np.log(y)

            self.alpha -= self.learning_rate * np.mean(gradient_alpha * X, axis=0) / m
            self.beta -= self.learning_rate * np.mean(gradient_beta * X, axis=0) / m

            if self.regularization:
                self.alpha -= (
                    self.reg_strength * self._regularization_term(self.alpha) / m
                )
                self.beta -= (
                    self.reg_strength * self._regularization_term(self.beta) / m
                )

            if self.verbose and i % 10 == 0:
                gradient_norm = np.linalg.norm(gradient_alpha + gradient_beta)
                print(f"[BetaReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {gradient_norm}")

        self._fitted = True
        return self

    def _regularization_term(self, weights: Matrix) -> Matrix:
        if self.regularization == "l1":
            return np.sign(weights)
        elif self.regularization == "l2":
            return self.weights
        elif self.regularization == "elastic-net":
            l1_term = np.sign(weights)
            l2_term = 2 * weights
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(weights)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> float:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return np.dot(X, self.alpha / (self.alpha + self.beta))

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class InverseGaussianRegressor(Estimator, Supervised):
    """
    Inverse Gaussian regression is a statistical technique applied in
    machine learning to model relationships between predictor variables
    and continuous response variables when the response data exhibits
    a skewed or non-normally distributed pattern.

    Parameters
    ----------
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter of `elastic-net`
    `alpha` : float, default=0.01
        Regularization strength
    `regularization` : {"l1", "l2", "elastic-net"}, optional, default=None
        Regularization type
    `phi` : float, default=1.0
        Dispersion parameter

    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        phi: float = 1.0,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        regularization: Literal["l1", "l2", "elastic-net"] = None,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.phi = phi
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "phi": ("0<,+inf", None),
                "l1_ratio": ("0,1", None),
                "alpha": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        m, n = X.shape
        self.weights = np.ones(n)

        for i in range(self.max_iter):
            error = y - np.dot(X, self.weights)

            gradient = np.dot(X.T, (error * self.phi) / (2 * self.weights * y**2))
            gradient += 0.5 * error**2 / (2 * self.weights * y)
            gradient -= 0.5 * (self.phi / self.weights)
            gradient *= self.learning_rate / m

            if self.regularization:
                regularization_term = self._regularization_term()
                gradient += self.alpha * regularization_term / m

            if self.verbose and i % 10 == 0:
                print(f"[InvGaussianReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {np.linalg.norm(gradient)}")

            self.weights -= gradient
        self._fitted = True
        return self

    def _regularization_term(self):
        if self.regularization == "l1":
            return np.sign(self.weights)
        elif self.regularization == "l2":
            return self.weights
        elif self.regularization == "elastic-net":
            l1_term = np.sign(self.weights)
            l2_term = 2 * self.weights
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(self.weights)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return 1 / (self.phi * X) * np.dot(X, self.weights)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
