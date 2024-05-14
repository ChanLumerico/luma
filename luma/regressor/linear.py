from typing import Self
import numpy as np

from luma.interface.typing import Matrix, Vector
from luma.interface.util import KernelUtil
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.metric.regression import MeanSquaredError


__all__ = (
    "LinearRegressor",
    "RidgeRegressor",
    "LassoRegressor",
    "ElasticNetRegressor",
    "KernelRidgeRegressor",
    "BayesianRidgeRegressor",
)


class RidgeRegressor(Estimator, Supervised):
    """
    Ridge regression is a linear regression technique used to
    prevent overfitting in predictive models.
    It adds a penalty term called "L2 regularization" to help
    reduce the complexity of the model and prevent it from
    fitting noise in the data.

    Parameters
    ----------
    `alpha` : float, default=1.0
        L2-regularization strength

    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._fitted = False

        self.set_param_ranges({"alpha": ("0,+inf", None)})
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        identity_matrix = np.identity(X.shape[1])
        self.coef_ = np.linalg.inv(X.T.dot(X) + self.alpha * identity_matrix)
        self.coef_ = self.coef_.dot(X.T).dot(y)
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X.dot(self.coef_)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class LassoRegressor(Estimator, Supervised):
    """
    Lasso regression is a linear regression technique used for
    feature selection and regularization. It adds a penalty term,
    called "L1 regularization," to the standard linear regression
    objective function. Lasso encourages some of the model's
    coef_ to become exactly zero, effectively eliminating
    certain features from the model.

    Parameters
    ----------
    `alpha` : float, default=1.0
        L1-regularization strength
    `max_iter` : int, default=100
        Number of iteration
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update

    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._fitted = False

        self.set_param_ranges(
            {
                "alpha": ("0,+inf", None),
                "max_iter": ("0<,+inf", int),
                "learning_rate": ("0<,+inf", None),
            }
        )
        self.check_param_ranges()

    def _soft_threshold(self, x: Matrix, threshold: float) -> Matrix:
        return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.coef_ = np.zeros(X.shape[1])

        for i in range(self.max_iter):
            coefficients_prev = self.coef_.copy()
            y_pred = X.dot(self.coef_)
            gradient = -(X.T.dot(y - y_pred)) * self.learning_rate
            self.coef_ = self.coef_ - (1.0 / X.shape[0]) * gradient
            self.coef_ = self._soft_threshold(self.coef_, self.alpha / X.shape[0])

            if self.verbose and i % 10 == 0:
                print(f"[LassoReg] iteration: {i}/{self.max_iter}", end="")
                print(
                    f" - delta-coeff norm: {np.linalg.norm(self.coef_ - coefficients_prev)}"
                )

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X.dot(self.coef_)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class ElasticNetRegressor(Estimator, Supervised):
    """
    Elastic-Net regression is a linear regression technique
    that combines both L1 (Lasso) and L2 (Ridge) regularization methods.
    It adds a combination of L1 and L2 penalty terms to the
    standard linear regression objective function.

    Parameters
    ----------
    `alpha` : float, default=1.0
        Regularization strength
    `l1_ratio` : float, default=0.5, range=[0,1]
        Balancing parameter between `l1` and `l2`
    `max_iter` : int, default=100
        Number of iteration
    `learning_rate` : float, default=0.01
        Step size of the gradient descent update

    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.coef_ = None
        self._fitted = False

        self.set_param_ranges(
            {
                "alpha": ("0,+inf", None),
                "l1_ratio": ("0,1", None),
                "max_iter": ("0<,+inf", int),
                "learning_rate": ("0<,+inf", None),
            }
        )
        self.check_param_ranges()

    def _soft_threshold(self, x: Matrix, alpha: float) -> Matrix:
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

    def fit(self, X: Matrix, y: Matrix) -> Self:
        N, p = X.shape
        self.coef_ = np.zeros(p)

        for i in range(self.max_iter):
            y_pred = X.dot(self.coef_)
            gradient = -(1 / N) * X.T.dot(y - y_pred)
            lefthand = self.coef_ - self.learning_rate * gradient
            righthand = self.alpha * self.l1_ratio

            self.coef_ = self._soft_threshold(lefthand, righthand)
            self.coef_ /= 1 + self.alpha * (1 - self.l1_ratio)

            if self.verbose and i % 10 == 0:
                print(f"[ElasticReg] iteration: {i}/{self.max_iter}")

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        if self.coef_ is None:
            raise ValueError()
        return X.dot(self.coef_)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class LinearRegressor(Estimator, Supervised):
    """
    An Ordinary Least Squares (OLS) Linear Regressor is a statistical method
    used in linear regression analysis. It estimates the coef_ of the
    linear equation, minimizing the sum of the squared differences between
    observed and predicted values. This results in a line of best fit through
    the data points in multidimensional space. OLS is widely used for its
    simplicity and efficiency in modeling linear relationships.
    """

    def __init__(self):
        self.coef_ = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X_T = np.transpose(X)
        self.coef_ = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X.dot(self.coef_)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class KernelRidgeRegressor(Estimator, Supervised):
    """
    Kernel Ridge Regression (KRR) combines ridge regression's linear coefficient
    penalization with kernel methods to handle non-linear data. By using a kernel
    function, KRR maps input data into a higher-dimensional feature space. In this
    transformed space, it minimizes the regularized loss function to find linear
    relationships. KRR excels in capturing complex, non-linear patterns in data,
    making it versatile for various regression tasks.

    Parameters
    ----------
    `alpha` : float, default=1.0
        Regularization strength
    `deg` : int, default=2
        Degree for polynomial kernel
    `gamma` : float, default=2
        Scaling factor for Tanh, RBF, sigmoid kernels
    `coef` : float, default=1.0
        Base coefficient for linear and polynomial kernels
    `kernel` : FuncType, default="rbf"
        Kernel functions

    """

    def __init__(
        self,
        alpha: float = 1.0,
        deg: int = 2,
        gamma: float = 1.0,
        coef: float = 1.0,
        kernel: KernelUtil.FuncType = "rbf",
    ) -> None:
        self.alpha = alpha
        self.deg = deg
        self.gamma = gamma
        self.coef = coef
        self.kernel = kernel
        self._X = None
        self._fitted = False

        self.kernel_params = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "deg": self.deg,
            "coef": self.coef,
        }

        self.set_param_ranges(
            {
                "alpha": ("0,+inf", None),
                "deg": ("0<,+inf", int),
                "gamma": ("0<,+inf", None),
                "coef": ("-inf,+inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Vector) -> Self:
        m, _ = X.shape
        self._X = X
        self.ku_ = KernelUtil(self.kernel, **self.kernel_params)

        K = self.ku_.kernel_func(X)
        self.alpha_mat = np.eye(m) * self.alpha
        self.dual_coef = np.linalg.inv(K + self.alpha_mat).dot(y)

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        K = self.ku_.kernel_func(X, self._X)

        return K.dot(self.dual_coef)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class BayesianRidgeRegressor(Estimator, Supervised):
    """
    Bayesian Ridge Regression applies Bayesian methods to linear models,
    estimating parameters with uncertainty. It automates regularization,
    balancing data fit and model simplicity. The approach provides both
    point estimates and confidence measures. It's effective for predictive
    modeling with inherent uncertainty.

    Parameters
    ----------
    `alpha_init` : float, optional, default=None
        Initial value for the precision of the distribution of noise
    `lambda_init` : float, optional, default=None
        Initial value for the precision of the distribution of weights

    """

    def __init__(
        self,
        alpha_init: float | None = None,
        lambda_init: float | None = None,
    ) -> None:
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self._fitted = False

        self.set_param_ranges(
            {"alpha_init": ("0<,+inf", None), "lambda_init": ("0<,+inf", None)}
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Vector) -> Self:
        if self.alpha_init is None:
            self.alpha_init = 1 / np.var(y)
        if self.lambda_init is None:
            self.lambda_init = 1.0

        m, n = X.shape
        X = np.column_stack((np.ones(m), X))
        A = np.eye(n + 1) * self.alpha_init

        sigma_inv = A + self.lambda_init * X.T.dot(X)
        sigma = np.linalg.inv(sigma_inv)
        mu = self.lambda_init * sigma.dot(X.T).dot(y)

        self.coef_ = mu
        self.sigma_ = sigma

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X.dot(self.coef_)

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
