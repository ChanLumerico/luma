from typing import Literal
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

from luma.interface.super import Estimator, Evaluator, Unsupervised
from luma.interface.util import Matrix, Scalar
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.distance import Euclidean, Minkowski
from luma.metric.clustering import SilhouetteCoefficient


__all__ = (
    'DBSCAN',
    'OPTICS'
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


class OPTICS(Estimator, Unsupervised):
    
    """
    OPTICS (Ordering Points To Identify the Clustering Structure) is an 
    unsupervised learning algorithm for identifying cluster structures 
    in spatial data. It creates an ordered list of points based on 
    core-distance and reachability-distance, allowing it to find 
    clusters of varying densities. Unlike algorithms like DBSCAN, it 
    doesn't require a global density threshold, making it versatile 
    for complex datasets. The result is often visualized as a 
    reachability plot, revealing the data's clustering hierarchy and 
    density variations.
    
    Parameters
    ----------
    `epsilon` : Radius of neighborhood hypersphere
    `min_points` : Minimum nuber of points to form a cluster
    `threshold` : Threshold for filtering samples with large reachabilities
    
    """
    
    def __init__(self, 
                 epsilon: float = 1.0,
                 min_points: int = 5,
                 threshold: float = 1.5,
                 verbose: bool = False) -> None:
        self.epsilon = epsilon
        self.min_points = min_points
        self.threshold = threshold
        self.verbose = verbose
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'OPTICS':
        self._X = X
        m, _ = X.shape
        
        self.processed = np.full(m, False, dtype=bool)
        self.reachability = np.full(m, np.inf)
        self.ordered_points = []
        
        for i in range(m):
            if self.verbose and i % 50 == 0 and i:
                print(f"[OPTICS] Finished for point {i}/{m}",
                      f"with reachability {self.reachability[i]}")
            
            if self.processed[i]: continue
            seeds = []
            point_neighbors = self._neighbors(X, i)
            core_dist = self._core_distance(point_neighbors)
            
            self.processed[i] = True
            self.ordered_points.append(i)
            
            if not np.isinf(core_dist):
                self._update(X, core_dist, i, seeds)
                seeds.sort(key=lambda x: x[1])
                
                while seeds:
                    next_, _ = seeds.pop(0)
                    self.processed[next_] = True
                    self.ordered_points.append(next_)
                    next_neighbors = self._neighbors(X, next_)
                    core_dist = self._core_distance(next_neighbors)
                    
                    if not np.isinf(core_dist):
                        self._update(X, core_dist, next_, seeds)
        
        self._fitted = True
        return self
    
    def _core_distance(self, neighbors: Matrix) -> Matrix | Scalar:
        if len(neighbors) >= self.min_points:
            return sorted(neighbors)[self.min_points - 1]
        return np.inf
    
    def _neighbors(self, X: Matrix, idx: int) -> Matrix:
        distances = cdist([X[idx]], X)[0]
        return distances[distances <= self.epsilon]
    
    def _update(self, X: Matrix, core_dist: Scalar, 
                idx: int, seeds: list) -> None:
        distances = cdist([X[idx]], X)[0]
        for i, dist in enumerate(distances):
            if dist <= self.epsilon and not self.processed[i]:
                new_reach_dist = max(core_dist, dist)
                
                if np.isinf(self.reachability[i]):
                    self.reachability[i] = new_reach_dist
                    seeds.append((i, new_reach_dist))
                elif new_reach_dist < self.reachability[i]:
                    self.reachability[i] = new_reach_dist
    
    def plot_reachability(self, color: str = 'royalblue') -> None:
        m = range(len(self.ordered_points))
        vals = self.reachability[self.ordered_points]
        
        plt.figure(figsize=(8, 5))
        plt.plot(m, vals, color=color)
        plt.fill_between(m, vals, color=color, alpha=0.5)
        
        plt.title('Reachability Plot')
        plt.xlim(m[0], m[-1])
        
        plt.xlabel('Order of Points')
        plt.ylabel('Reachability Distance')
        plt.tight_layout()
        plt.show()
    
    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        cluster_labels = np.full_like(self.reachability, -1, dtype=int)
        current_ = 0
        
        for point in self.ordered_points:
            if self.reachability[point] <= self.threshold:
                cluster_labels[point] = current_
            else: current_ += 1

        return cluster_labels
    
    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")
    
    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.compute(self._X, self.labels)
    
    def set_params(self, 
                   epsilon: float = None,
                   min_points: int = None,
                   threshold: float = None) -> None:
        if epsilon is not None: self.epsilon = float(epsilon)
        if min_points is not None: self.min_points = int(min_points)
        if threshold is not None: self.threshold = str(threshold)

