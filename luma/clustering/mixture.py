from typing import Self
import numpy as np

from luma.core.super import Estimator, Evaluator, Unsupervised
from luma.interface.typing import Matrix, Vector
from luma.metric.clustering import SilhouetteCoefficient
from luma.interface.exception import NotFittedError


__all__ = ("GaussianMixture", "MultinomialMixture")


class GaussianMixture(Estimator, Unsupervised):
    """
    A Gaussian Mixture Model (GMM) is a probabilistic model for representing
    an ensemble of multiple Gaussian distributions within a dataset. It is
    used in clustering by assuming each cluster follows a different Gaussian
    distribution. GMM provides soft clustering, assigning probabilities to
    each data point for belonging to each cluster. The model parameters are
    estimated using the Expectation-Maximization algorithm.

    Parameters
    ----------
    `n_clusters` : int
        Number of clusters to estimate
    `max_iter` : int, default=100
        Maximum amount of iteration
    `tol` : float, default=1e-5
        Tolerance for early convergence
    `random_state` : int, optional, default=None
        Seed for random permutation

    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self._X = None
        self._fitted = False

        self.set_param_ranges(
            {
                "n_clusters": ("0<,+inf", int),
                "max_iter": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self._X = X
        self._initialize_params(X)
        for i in range(self.max_iter):
            R = self._E_step(X)
            means_old = self.means.copy()

            self._M_step(X, R)
            norm_ = np.linalg.norm(self.means - means_old)

            if self.verbose and i % 20 and i:
                print(f"[GMM] Finished iteration {i}/{self.max_iter}", end="")
                print(f" with delta-means-norm: {norm_}")

            if norm_ < self.tol:
                if self.verbose:
                    print(f"[GMM] Early-convergence at iteration", end="")
                    print(f" {i}/{self.max_iter}")
                break

        self._fitted = True
        return self

    def _initialize_params(self, X: Matrix) -> None:
        m, _ = X.shape
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(m)[: self.n_clusters]

        self.means = X[random_indices]
        self.covs = Matrix([np.cov(X.T) for _ in range(self.n_clusters)])
        self.weights = np.full(self.n_clusters, 1 / self.n_clusters)

    def _E_step(self, X: Matrix) -> Matrix:
        m, n = X.shape
        resp = np.zeros((m, self.n_clusters))

        for i in range(self.n_clusters):
            cov_inv = np.linalg.inv(self.covs[i])
            diff = X - self.means[i]
            exp_term = np.exp(-0.5 * np.sum(diff.dot(cov_inv) * diff, axis=1))
            coeff = 1 / np.sqrt(np.linalg.det(self.covs[i]) * (2 * np.pi) ** n)
            resp[:, i] = self.weights[i] * coeff * exp_term

        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _M_step(self, X: Matrix, R: Matrix) -> None:
        self.weights = R.mean(axis=0)
        self.means = R.T.dot(X) / np.sum(R, axis=0)[:, np.newaxis]
        for i in range(self.n_clusters):
            diff = X - self.means[i]
            self.covs[i] = (R[:, i, np.newaxis] * diff).T.dot(diff) / R[:, i].sum()

    @property
    def labels(self) -> Vector:
        return self.predict(self._X)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        R = self._E_step(X)
        return np.argmax(R, axis=1)

    def predict_proba(self, X: Matrix) -> Vector:
        return self._E_step(X)

    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)


class MultinomialMixture(Estimator, Unsupervised):
    """
    Multinomial Mixture Models (MMM) cluster categorical data using a mixture
    of multinomial distributions. Each cluster is modeled with a distinct
    multinomial distribution, representing the probability of each category
    within that cluster. The Expectation-Maximization (EM) algorithm is
    typically used to estimate model parameters, including cluster
    probabilities and multinomial parameters. MMM is commonly applied in text
    analysis and other scenarios involving discrete data, like document
    clustering or topic modeling.

    Parameters
    ----------
    `n_clusters` : int
        Number of clusters to estimate
    `max_iter` : int, default=100
        Maximum amount of iteration
    `tol` : float, default=1e-5
        Tolerance for early convergence

    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.pi = None
        self.theta = None
        self.X_ = None
        self._fitted = False

        self.set_param_ranges(
            {
                "n_clusters": ("0<,+inf", int),
                "max_iter": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self.X_ = X
        self._initialize_parameters(X)
        logL = None

        for i in range(self.max_iter):
            resp = self._E_step(X)
            self._M_step(X, resp)

            log_prob_cluster = np.log(self.theta) @ X.T
            log_prob_cluster += np.log(self.pi[:, np.newaxis])
            log_prob_samples = self._logsumexp(log_prob_cluster, axis=0)
            current_logL = np.sum(log_prob_samples)

            if logL is not None and np.abs(current_logL - logL) < self.tol:
                if self.verbose:
                    print(
                        f"EM converged at iteration {i+1}",
                        f"with log-likelihood: {current_logL}",
                    )
                break

            logL = current_logL
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}, log-likelihood: {logL}")

        self._fitted = True
        return self

    def _logsumexp(self, a: Matrix, axis: int = None, keepdims: bool = False) -> Matrix:
        a_max = np.max(a, axis=axis, keepdims=True)
        tmp = np.exp(a - a_max)
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        out = np.log(s)
        if not keepdims:
            a_max = np.squeeze(a_max, axis=axis)
        out += a_max
        return out

    def _initialize_parameters(self, X: Matrix) -> None:
        _, n = X.shape
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        self.theta = np.random.dirichlet(alpha=np.ones(n), size=self.n_clusters)

    def _E_step(self, X: Matrix) -> Matrix:
        log_prob = np.log(self.theta) @ X.T + np.log(self.pi[:, np.newaxis])
        log_resp = log_prob - self._logsumexp(log_prob, axis=0, keepdims=True)

        return np.exp(log_resp).T

    def _M_step(self, X: Matrix, resp: Matrix) -> None:
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.pi = nk / nk.sum()
        self.theta = (resp.T @ X) / nk[:, np.newaxis]

    @property
    def labels(self) -> Vector:
        return self.predict(self.X_)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        resp = self._E_step(X)
        return np.argmax(resp, axis=1)

    def fit_predict(self, X: Matrix) -> Vector:
        self.fit(X)
        return self.predict(X)

    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)
