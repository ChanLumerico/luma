from typing import Tuple
from scipy.spatial.distance import cdist
import numpy as np

from luma.core.super import Transformer, Supervised
from luma.interface.util import Matrix, Vector
from luma.interface.exception import NotFittedError


__all__ = (
    'LocalOutlierFactor'
)


class LocalOutlierFactor(Transformer, Transformer.Both, Supervised):
    
    """
    The Local Outlier Factor (LOF) algorithm identifies outliers by comparing 
    the local density of a data point with the densities of its neighbors. 
    Points with significantly lower density than their neighbors are considered 
    outliers. LOF scores greater than 1 indicate potential outliers. It is 
    effective in datasets with varying densities and does not require a prior 
    assumption about the data distribution.
    
    Parameters
    ----------
    `n_neighbors` : Number of neighbors to estimate the local densities
    
    Examples
    --------
    >>> lof = LocalOutlierFactor()
    >>> lof.fit(X)
    
    Getting LOF Scores
    >>> lof_scores = lof.get_scores(X)
    
    Filtering Dataset
    >>> X_, y_ = lof.filter(X, y) # or lof.transform(X, y)
    
    """
    
    def __init__(self, 
                 n_neighbors: int = 10,
                 threshold: float = 1.5):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.lrd_ = None
        self.neighbors_ = None
        self._fitted = False
    
    def fit(self, X: Matrix, _ = None) -> 'LocalOutlierFactor':
        n_samples = len(X)
        self.lrd_ = np.zeros(n_samples)
        self.neighbors_ = []

        for i in range(n_samples):
            distances = cdist([X[i]], X)[0]
            neighbors_idx = np.argsort(distances)[1:self.n_neighbors + 1]
            neighbors = X[neighbors_idx]

            self.lrd_[i] = self._local_reachability_density(X[i], neighbors)
            self.neighbors_.append(neighbors_idx)

        self._fitted = True
        return self

    def _k_distance(self, x: Matrix, neighbors: Matrix) -> Matrix:
        sorted_ = np.sort(cdist([x], neighbors)[0]) 
        return sorted_[self.n_neighbors - 1]

    def _reachability_distance(self, x: Matrix, 
                               neighbor: Matrix, 
                               neighbors: Matrix) -> Matrix:
        k_dist_neighbor = self._k_distance(neighbor, neighbors)
        return max(k_dist_neighbor, np.linalg.norm(x - neighbor))

    def _local_reachability_density(self, x: Matrix, 
                                    neighbors: Matrix) -> Matrix:
        reach_distances = [self._reachability_distance(x, neighbor, neighbors) 
                           for neighbor in neighbors]
        return 1 / (np.sum(reach_distances) / len(neighbors) 
                    if len(neighbors) > 0 else 1)

    def get_scores(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        lof = np.zeros(len(X))
        for i in range(len(X)):
            lrd_sum = sum(self.lrd_[idx] for idx in self.neighbors_[i])
            lof[i] = lrd_sum / (self.n_neighbors * self.lrd_[i])

        return lof
    
    def filter(self, X: Matrix, y: Matrix) -> Tuple[Matrix, Vector]:
        lof_scores = self.get_scores(X)
        condition = lof_scores < self.threshold
        
        return X[condition], y[condition]
    
    def transform(self, X: Matrix, y: Vector) -> Tuple[Matrix, Vector]:
        return self.filter(X, y)
    
    def fit_transform(self, X: Matrix, y: Vector) -> Tuple[Matrix, Vector]:
        self.fit(X)
        return self.transform(X, y)

