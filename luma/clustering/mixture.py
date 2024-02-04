import numpy as np

from luma.core.super import Estimator, Evaluator, Unsupervised
from luma.interface.util import Matrix, Vector
from luma.metric.clustering import SilhouetteCoefficient
from luma.interface.exception import NotFittedError


__all__ = (
    'GaussianMixture',
    'MultinomialMixture'
)


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
    `n_clusters` : Number of clusters to estimate
    `max_iter` : Maximum amount of iteration
    `tol` : Tolerance for early convergence
    `random_state` : Seed for random permutation
    
    """
    
    def __init__(self, 
                 n_clusters : int,
                 max_iter: int = 100,
                 tol: float = 1e-5,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'GaussianMixture':
        self._X = X
        self._initialize_params(X)
        for i in range(self.max_iter):
            R = self._E_step(X)
            means_old = self.means.copy()
            
            self._M_step(X, R)
            norm_ = np.linalg.norm(self.means - means_old)
            
            if self.verbose and i % 20 and i:
                print(f'[GMM] Finished iteration {i}/{self.max_iter}', end='')
                print(f' with delta-means-norm: {norm_}')
            
            if norm_ < self.tol:
                if self.verbose:
                    print(f'[GMM] Early-convergence at iteration', end='')
                    print(f' {i}/{self.max_iter}')
                break
        
        self._fitted = True
        return self
    
    def _initialize_params(self, X: Matrix) -> None:
        m, _ = X.shape
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(m)[:self.n_clusters]
        
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
        if not self._fitted: raise NotFittedError(self)
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
    `n_clusters` : Number of clusters to estimate
    `max_iter` : Maximum amount of iteration
    `tol` : Tolerance for early convergence
    
    """
    
    def __init__(self, 
                 n_clusters: int, 
                 max_iter: int = 100, 
                 tol: float = 1e-5,
                 verbose: bool = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._X = None
        self._fitted = False
        
    def fit(self, X: Matrix) -> 'MultinomialMixture':
        self._X = X
        m, _ = X.shape
        
        prev_logL = None
        self._initialize_parameters(X)
        for i in range(self.max_iter):
            resp = self._E_step(X)
            self._M_step(X, resp)
            
            logL = 0
            for i in range(m):
                likelihoods = []
                for j in range(self.n_clusters):
                    cluster_likelihood = np.prod(self.theta[j] ** X[i, :])
                    weighted_likelihood = self.pi[j] * cluster_likelihood
                    likelihoods.append(weighted_likelihood)

                L_sum = np.sum(likelihoods)
                logL += np.log(L_sum)
            
            if prev_logL is not None:
                diff = np.abs(logL - prev_logL)
                if self.verbose and i % 20 == 0 and i:
                    print(f'[MMM] Finished iteration {i}/{self.max_iter}', end='')
                    print(f' with delta-likelihood: {diff}')
                    
                if diff < self.tol:
                    if self.verbose:
                        print(f'[MMM] Early-convergence at {i}/{self.max_iter}', end='')
                        print(f' with delta-likelihood: {diff}')
                    break
            
            prev_logL = logL
        
        self._fitted = True
        return self
    
    def _initialize_parameters(self, X: Matrix) -> None:
        _, n = X.shape
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        self.theta = np.random.dirichlet(alpha=np.ones(n), size=self.n_clusters)
    
    def _E_step(self, X: Matrix) -> Matrix:
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_clusters))
        
        for j in range(self.n_clusters):
            likelihood = np.prod(self.theta[j] ** X, axis=1)
            resp[:, j] = self.pi[j] * likelihood
        
        resp /= resp.sum(axis=1, keepdims=True)
        return resp
    
    def _M_step(self, X: Matrix, R: Matrix) -> None:
        self.pi = R.mean(axis=0)
        for j in range(self.n_clusters):
            weighted_sum = np.dot(R[:, j], X)
            self.theta[j] = weighted_sum / np.sum(R[:, j])

    @property
    def labels(self) -> Vector:
        return self.predict(self._X)
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        R = self._E_step(X)
        return np.argmax(R, axis=1)
    
    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)

