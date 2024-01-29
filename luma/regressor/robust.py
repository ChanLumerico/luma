from typing import Tuple
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector
from luma.interface.exception import NotFittedError
from luma.metric.regression import MeanSquaredError
from luma.regressor.linear import LinearRegressor


__all__ = (
    'RANSAC',
    'MLESAC'
)


class RANSAC(Estimator, Supervised):
    
    """
    RANSAC (Random Sample Consensus) is an iterative algorithm used for robust 
    estimation of parameters from a set of data containing outliers. It works by 
    randomly selecting a subset of the original data and fitting a model to this 
    subset. The algorithm then tests all other data points against this model, 
    classifying them as inliers or outliers based on a predefined threshold. 
    This process is repeated multiple times, each time selecting a different 
    subset. The best model is chosen based on the maximum number of inliers 
    it includes.
    
    Parameters
    ----------
    `estimator` : Regression estimator (Default `LinearRegressor()`)
    `min_points` : Mininum sample size for each random sample
    `max_iter` : Maximum iteration
    `min_inliers` : Minimum number of inliers for a model to be considered valid
    `threshold` : Maximum distance a point can have to be considered an inlier
    `random_state` : Seed for random sampling the data
    
    Properties
    ----------
    Get best estimator:
    ```py
        @property
        def best_estimator(self) -> Estimator
    ```
    """
    
    def __init__(self, 
                 estimator: Estimator = LinearRegressor(),
                 min_points: int | float = 2,
                 max_iter: int = 1000,
                 min_inliers: int | float = 0.5,
                 threshold: float = None,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.min_points = min_points
        self.max_iter = max_iter
        self.min_inliers = min_inliers
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose
        self.inliers_ = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'RANSAC':
        m, _ = X.shape
        best_err = np.inf
        best_indices = None
        
        np.random.seed(self.random_state)
        Xy = np.hstack((X, y.reshape(-1, 1)))
        
        if self.min_points < 1: self.min_points = np.ceil(m * self.min_points)
        if self.min_inliers < 1: self.min_inliers = np.ceil(m * self.min_inliers)
        if self.threshold is None: self.threshold = self._mean_absdev(y)
        
        for i in range(self.max_iter):
            hypo_indices, test_indices = self._random_partition(X)
            hypo_inliers = Xy[hypo_indices]
            test_points = Xy[test_indices]
            
            self.estimator.fit(hypo_inliers[:, :-1], hypo_inliers[:, -1])
            test_err = self._error(test_points[:, :-1], test_points[:, -1])
            
            true_indices = test_indices[test_err < self.threshold]
            true_inliers = Xy[true_indices]
            
            if len(true_inliers) > self.min_inliers:
                Xy_better = np.concatenate((hypo_inliers, true_inliers))
                
                self.estimator.fit(Xy_better[:, :-1], Xy_better[:, -1])
                better_err = self._error(Xy_better[:, :-1], Xy_better[:, -1])
                this_err = np.mean(better_err)
                
                if this_err < best_err:
                    best_err = this_err
                    best_indices = np.concatenate((hypo_indices, true_indices))
            
            if self.verbose and i % 100 == 0:
                print(f'[RANSAC] Iteration {i}/{self.max_iter}',
                      f'with current best-error: {best_err}')
        
        self.inliers_ = Xy[best_indices, :-1]
        self._fitted = True
        return self
    
    def _random_partition(self, X: Matrix) -> Tuple[Vector, Vector]:
        all_indices = np.arange(X.shape[0])
        np.random.shuffle(all_indices)
        indices_l = all_indices[:self.min_points]
        indices_r = all_indices[self.min_points:]
        
        return indices_l, indices_r

    def _error(self, X: Matrix, y: Vector) -> Vector:
        y_pred = self.estimator.predict(X)
        return np.abs(y - y_pred)
    
    def _mean_absdev(self, y: Vector) -> float:
        y_median = np.median(y)
        abs_dev = np.abs(y - y_median)
        mad = np.median(abs_dev)
        return mad
    
    @property
    def best_estimator(self) -> Estimator:
        if not self._fitted: raise NotFittedError(self)
        return self.estimator
    
    def predict(self, X: Matrix) -> Vector:
        best = self.best_estimator
        return best.predict(X)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class MLESAC(Estimator, Supervised):
    
    """
    MLESAC (Maximum Likelihood Estimation SAmple Consensus) is an iterative algorithm 
    used for robust estimation of parameters from a set of data containing outliers. 
    Unlike RANSAC which focuses on maximizing the number of inliers, MLESAC seeks 
    to maximize the likelihood of the model given the data, considering both the 
    number of inliers and the quality of the fit.
    
    Parameters
    ----------
    `estimator` : Regression estimator (Default `LinearRegressor()`)
    `min_points` : Minimum sample size for each random sample
    `max_iter` : Maximum number of iterations
    `min_inliers` : Minimum number of inliers for a model to be considered valid
    `threshold` : Maximum distance a point can have to be considered an inlier
    `random_state` : Seed for random sampling the data
    
    Properties
    ----------
    Get best estimator:
    ```py
        @property
        def best_estimator(self) -> Estimator
    ```
    
    Notes
    -----
    * `MLESAC` assumes that inliers follow a multivariate Gaussian model
    
    """
    
    def __init__(self, 
                 estimator: Estimator = LinearRegressor(),
                 min_points: int | float = 2,
                 max_iter: int = 1000,
                 min_inliers: int | float = 0.5,
                 threshold: float = None,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.min_points = min_points
        self.max_iter = max_iter
        self.min_inliers = min_inliers
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose
        self.inliers_ = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'MLESAC':
        m, _ = X.shape
        best_likelihood = -np.inf
        best_indices = None
        
        np.random.seed(self.random_state)
        Xy = np.hstack((X, y.reshape(-1, 1)))
        
        if self.min_points < 1: self.min_points = np.ceil(m * self.min_points)
        if self.min_inliers < 1: self.min_inliers = np.ceil(m * self.min_inliers)
        if self.threshold is None: self.threshold = self._mean_absdev(y)
        
        for i in range(self.max_iter):
            hypo_indices, test_indices = self._random_partition(X)
            hypo_inliers = Xy[hypo_indices]
            test_points = Xy[test_indices]
            
            self.estimator.fit(hypo_inliers[:, :-1], hypo_inliers[:, -1])
            test_err = self._error(test_points[:, :-1], test_points[:, -1])
            
            true_indices = test_indices[test_err < self.threshold]
            true_inliers = Xy[true_indices]
            
            if len(true_inliers) >= self.min_inliers:
                Xy_better = np.concatenate((hypo_inliers, true_inliers))
                
                self.estimator.fit(Xy_better[:, :-1], Xy_better[:, -1])
                better_err = self._error(Xy_better[:, :-1], Xy_better[:, -1])
                likelihood = self._calculate_likelihood(better_err)
                
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_indices = np.concatenate((hypo_indices, true_indices))

            if self.verbose and i % 100 == 0:
                print(f'[MLESAC] Iteration {i}/{self.max_iter}',
                      f'with current best-likelihood: {best_likelihood}')

        self.inliers_ = Xy[best_indices, :-1]
        self._fitted = True
        return self
    
    def _random_partition(self, X: Matrix) -> Tuple[Vector, Vector]:
        all_indices = np.arange(X.shape[0])
        np.random.shuffle(all_indices)
        indices_l = all_indices[:self.min_points]
        indices_r = all_indices[self.min_points:]
        
        return indices_l, indices_r

    def _error(self, X: Matrix, y: Vector) -> Vector:
        y_pred = self.estimator.predict(X)
        return np.abs(y - y_pred)
    
    def _mean_absdev(self, y: Vector) -> float:
        y_median = np.median(y)
        abs_dev = np.abs(y - y_median)
        mad = np.median(abs_dev)
        return mad
    
    def _calculate_likelihood(self, error: Vector) -> float:
        if error.ndim == 1: error = error[:, np.newaxis]
        m, n = error.shape
        cov_matrix = np.cov(error, rowvar=False)
        
        if n == 1: cov_matrix = cov_matrix.reshape(1, 1)
        det_cov_matrix = np.linalg.det(cov_matrix)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        likelihood = -m * n * np.log(2 * np.pi) / 2
        likelihood -= m * np.log(det_cov_matrix) / 2
        for error in error:
            likelihood -= 0.5 * np.dot(error, np.dot(inv_cov_matrix, error))

        return likelihood
    
    @property
    def best_estimator(self) -> Estimator:
        if not self._fitted: raise NotFittedError(self)
        return self.estimator
    
    def predict(self, X: Matrix) -> Vector:
        best = self.best_estimator
        return best.predict(X)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

