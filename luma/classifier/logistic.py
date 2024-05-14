from typing import Literal, Self
import numpy as np

from luma.interface.typing import Matrix
from luma.metric.classification import Accuracy
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.core.super import Estimator, Evaluator, Supervised


__all__ = ("LogisticRegressor", "SoftmaxRegressor")


class LogisticRegressor(Estimator, Supervised):
    """
    Logistic regression is a statistical model used for binary classification,
    which means it's employed to predict one of two possible outcomes.
    It does this by modeling the relationship between one or more
    input variables and the probability of the binary outcome.

    Parameters
    ----------
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter of elastic-net
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
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.regularization = regularization
        self.alpha = alpha
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "l1_ratio": ("0,1", None),
                "alpha": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def sigmoid(self, z: Matrix) -> Matrix:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.insert(X, 0, 1, axis=1)
        m, n = X.shape
        self.theta = np.zeros(n)

        for i in range(self.max_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) * self.learning_rate
            gradient += self.alpha * self._regularization_term() / m
            gradient[0] -= self.alpha * self._regularization_term()[0]
            self.theta -= gradient

            if self.verbose and i % 10 == 0:
                print(f"[LogisticReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {np.linalg.norm(gradient)}")

        self._fitted = True
        return self

    def _regularization_term(self) -> Matrix:
        if self.regularization == "l1":
            return np.sign(self.theta)
        elif self.regularization == "l2":
            return self.theta
        elif self.regularization == "elastic-net":
            l1_term = np.sign(self.theta)
            l2_term = 2 * self.theta
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(self.theta)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        predictions = (h >= 0.5).astype(int)
        return predictions

    def predict_proba(self, X: Matrix) -> Matrix:
        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        return h

    def score(self, X: Matrix, y: Matrix, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class SoftmaxRegressor(Estimator, Supervised):
    """
    Softmax regression, also known as multinomial logistic regression,
    is a supervised machine learning technique used for classification tasks.
    It's an extension of logistic regression,
    but it can handle multiple classes (more than two).

    Parameters
    ----------
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update
    `max_iter` : int, default=100
        Number of iteration
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter of elastic-net
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
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
                "l1_ratio": ("0,1", None),
                "alpha": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def softmax(self, z: Matrix) -> Matrix:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.insert(X, 0, 1, axis=1)
        m, n = X.shape
        num_classes = len(np.unique(y))
        self.theta = np.zeros((n, num_classes))

        for i in range(self.max_iter):
            z = np.dot(X, self.theta)
            h = self.softmax(z)
            gradient = np.dot(X.T, (h - self._one_hot_encode(y)))
            gradient *= self.learning_rate
            gradient += self.alpha * self._regularization_term() / m
            gradient[0] -= self.alpha * self._regularization_term()[0]
            self.theta -= gradient

            if self.verbose and i % 10 == 0:
                print(f"[SoftmaxReg] iteration: {i}/{self.max_iter}", end="")
                print(f" - gradient-norm: {np.linalg.norm(gradient)}")

        self._fitted = True
        return self

    def _one_hot_encode(self, y: Matrix) -> Matrix:
        num_classes = self.theta.shape[1]
        one_hot = np.zeros((len(y), num_classes))
        for i in range(len(y)):
            one_hot[i, y[i]] = 1
        return one_hot

    def _regularization_term(self) -> Matrix:
        if self.regularization == "l1":
            return np.sign(self.theta)
        elif self.regularization == "l2":
            return self.theta
        elif self.regularization == "elastic-net":
            l1_term = np.sign(self.theta)
            l2_term = 2 * self.theta
            return (1 - self.l1_ratio) * l2_term + self.l1_ratio * l1_term
        elif self.regularization is None:
            return np.zeros_like(self.theta)
        else:
            raise UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.theta)
        h = self.softmax(z)
        predictions = np.argmax(h, axis=1)
        return predictions

    def predict_proba(self, X: Matrix) -> Matrix:
        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.theta)
        h = self.softmax(z)
        return h

    def score(self, X: Matrix, y: Matrix, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
