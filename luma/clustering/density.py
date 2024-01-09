from typing import Literal
import numpy as np

from luma.interface.super import Estimator, Evaluator, Unsupervised
from luma.interface.util import Matrix
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.distance import Euclidean, Minkowski
from luma.metric.clustering import SilhouetteCoefficient


__all__ = (
    'DBSCAN'
)


class DBSCAN(Estimator, Unsupervised):
    
    """
    DBSCAN, short for Density-Based Spatial Clustering of Applications 
    with Noise, is a clustering algorithm that groups points in a dataset 
    by their proximity and density. It identifies clusters as areas of high 
    point density, separated by regions of low density. Points in sparse 
    regions are classified as noise. DBSCAN is particularly effective at 
    discovering clusters of arbitrary shapes and dealing with outliers.
    
    Parameters
    ----------
    `epsilon` : Radius of a neighborhood hypersphere
    `min_points` : Minimum required points to form a cluster
    
    """
    
    def __init__(self, 
                 epsilon: float = 0.1, 
                 min_points: int = 5,
                 metric: Literal['euclidean', 'minkowski'] = 'euclidean') -> None:
        self.epsilon = epsilon
        self.min_points = min_points
        self.metric = metric
        self._X = None
        self._fitted = False
        
        self.metric_func = None
        if self.metric == 'euclidean': self.metric_func = Euclidean
        elif self.metric == 'minkowski': self.metric_func = Minkowski
        else: raise UnsupportedParameterError(self.metric)
        
    def fit(self, X: Matrix) -> 'DBSCAN':
        self._X = X
        clusters = [0] * X.shape[0]
        curPt = 0
        
        for i in range(X.shape[0]):
            if clusters[i]: continue
            neighbors = self._generate_neighbors(X, idx=i)
            if len(neighbors) < self.min_points:
                clusters[i] = -1
            else:
                curPt += 1
                self._expand_cluster(X, neighbors, clusters, i, curPt)
        
        self._cluster_labels = clusters
        self._fitted = True
        return self
    
    def _generate_neighbors(self, X: Matrix, idx: int) -> Matrix:
        neighbors = []
        for i in range(X.shape[0]):
            if self.metric_func.distance(X[idx], X[i]) < self.epsilon:
                neighbors.append(i)

        return np.array(neighbors)

    def _expand_cluster(self, X: Matrix, neighbors: Matrix, clusters: Matrix, 
                        idx: int, current: int) -> None:
        i = 0
        clusters[idx] = current
        
        while i < len(neighbors):
            next = neighbors[i]
            if clusters[next] == -1:
                clusters[next] = current
                
            elif clusters[next] == 0:
                clusters[next] = current
                next_neighbors = self._generate_neighbors(X, idx=next)
                
                if len(next_neighbors) > self.min_points:
                    neighbors = np.concatenate([neighbors, next_neighbors], axis=0)
            
            i += 1
    
    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return np.array(self._cluster_labels)
    
    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")
    
    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.compute(self._X, self.labels)
    
    def set_params(self, 
                   epsilon: float = None,
                   min_points: int = None) -> None:
        if epsilon is not None: self.epsilon = float(epsilon)
        if min_points is not None: self.min_points = int(min_points)

