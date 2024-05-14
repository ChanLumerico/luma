from typing import Self
import numpy as np

from luma.interface.typing import Matrix
from luma.interface.util import KernelUtil
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.metric.regression import MeanSquaredError


__all__ = ("SVR", "KernelSVR")


class SVR(Estimator, Supervised):
    """
    Support Vector Regressor (SVR) is a supervised machine learning
    algorithm used for regression tasks. It operates by determining a
    hyperplane that best fits the input data, aiming to minimize the
    difference between the predicted and actual values.

    Parameters
    ----------
    `C` : float, default=1.0
        Regularization parameter
    `epsilon` : float, default=0.1
        Epsilon-tube in the training loss function
    `batch_size` : int, default=100
        Size of a single batch
    `learning_rate` : float, default=0.001
        Step-size of gradient descent update
    `max_iter` : int, default=1000
        Number of iteration

    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        batch_size: int = 100,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        verbose: bool = False,
    ) -> None:
        self.C = C
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "C": ("0,+inf", None),
                "epsilon": ("0,+inf", None),
                "batch_size": ("0<,+inf", int),
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        m, n = X.shape
        id = np.arange(m)
        np.random.shuffle(id)

        self.bias = 0.0
        self.weight = np.zeros(n)

        for i in range(self.max_iter):
            for batch_point in range(0, m, self.batch_size):
                gradient_w, gradient_b = 0, 0
                for j in range(batch_point, batch_point + self.batch_size):
                    if j >= m:
                        continue
                    idx = id[j]
                    pred = np.dot(self.weight, X[idx].T) + self.bias
                    loss = pred - y[idx]

                    if np.abs(loss) > self.epsilon:
                        gradient_w += self.C * np.sign(loss) * X[idx]
                        gradient_b += self.C * np.sign(loss)

                self.weight -= self.learning_rate * self.weight
                self.weight -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b

            if self.verbose and i % 100 == 0 and i:
                print(f"[SVR] Finished iteration {i}/{self.max_iter}", end=" ")
                print(f"with weight-norm of {np.linalg.norm(self.weight)}")

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        return np.dot(X, self.weight) + self.bias

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class KernelSVR(Estimator, Supervised):
    """
    Kernel Support Vector Regression (KernelSVR) is an extension of the
    Support Vector Regression (SVR) algorithm that enables regression on
    non-linearly related data by mapping it into a higher-dimensional space
    using a kernel function. The kernel function allows the algorithm to
    establish a flexible and non-linear regression model in the transformed space.

    Parameters
    ----------
    `C` : float, default=1.0
        Regularization parameter
    `deg` : int, default=3
        Polynomial Degree for polynomial kernel
    `gamma` : float, default=1.0
        Shape parameter of Gaussian curve for RBF kernel
    `coef` : float, default=1.0
        Coefficient for polynomial, sigmoid kernel
    `learning_rate` : float, default=0.001
        Step-size for gradient descent update
    `max_iter` : int, default=1000
        Number of iteration
    `kernel` : FuncType, default="rbf"
        Type of kernel

    """

    def __init__(
        self,
        C: float = 1.0,
        deg: int = 3,
        gamma: float = 1.0,
        coef: float = 1.0,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        kernel: KernelUtil.FuncType = "rbf",
        verbose: bool = False,
    ) -> None:
        self.C = C
        self.deg = deg
        self.gamma = gamma
        self.coef = coef
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.kernel = kernel
        self.verbose = verbose
        self._kernel_func = None
        self._fitted = False

        self.set_param_ranges(
            {
                "C": ("0<,+inf", None),
                "deg": ("0<,+inf", int),
                "gamma": ("0<,+inf", None),
                "coef": ("-inf,+inf", None),
                "learning_rate": ("0<,+inf", None),
                "max_iter": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        m, _ = X.shape
        self._set_kernel_func()
        self.alpha = np.random.random(m)
        self.bias = 0

        y_mul_kernel = np.outer(y, y) * self._kernel_func(X, X)
        for i in range(self.max_iter):
            gradient = np.ones(m) - y_mul_kernel.dot(self.alpha)
            self.alpha += self.learning_rate * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)

            if self.verbose and i % 100 == 0 and i:
                print(f"[KernelSVR] Finished iteration {i}/{self.max_iter}", end=" ")
                print(f"with alpha-norm: {np.linalg.norm(self.alpha)}")

        alpha_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        bias_list = []
        for idx in alpha_idx:
            _append = y[idx] - (self.alpha * y).dot(self._kernel_func(X, X[idx]))
            bias_list.append(_append)
        self.bias = np.mean(bias_list)

        self._X = X
        self._y = y
        self._fitted = True
        return self

    def _linear_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        return xi.dot(xj.T)

    def _poly_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        return (self.coef + xi.dot(xj.T)) ** self.deg

    def _rbf_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        norm = np.linalg.norm(xi[:, np.newaxis] - xj[np.newaxis, :], axis=2)
        return np.exp(-self.gamma * norm**2)

    def _sigmoid_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        return np.tanh(2 * xi.dot(xj.T) + self.coef)

    def _set_kernel_func(self) -> None:
        if self.kernel == "linear":
            self._kernel_func = self._linear_kernel
        elif self.kernel == "poly":
            self._kernel_func = self._poly_kernel
        elif self.kernel == "rbf":
            self._kernel_func = self._rbf_kernel
        elif self.kernel == "sigmoid":
            self._kernel_func = self._sigmoid_kernel
        else:
            raise UnsupportedParameterError(self.kernel)

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        return (self.alpha * self._y).dot(self._kernel_func(self._X, X)) + self.bias

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
