from typing import *
import numpy as np

from luma.interface.exception import NotFittedError
from luma.interface.super import Estimator, Evaluator, Unsupervised
from luma.metric.classification import Accuracy


__all__ = ['KMeansClustering', 'KMeansClusteringPlus', 'KMediansClustering']


class KMeansClustering(Estimator, Unsupervised):
    
    """
    K-means clustering is a machine learning algorithm that  groups similar data 
    points into clusters. It works by iteratively assigning data points to the 
    nearest cluster center (centroid) and updating the centroids based on 
    the assigned data points. This process continues 
    until convergence.
    
    Parameters
    ----------
    ``n_clusters`` : Number of clusters \n
    ``max_iter`` : Number of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 max_iter: int = 100, 
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        init_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[init_indices]

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = [X[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            if np.all(np.array(new_centroids) == self.centroids): 
                if self.verbose: print(f'[K-Means] Ealry convergence at itertaion {i}')
                break
            
            if self.verbose and i % 10 == 0: 
                diff_norm = np.linalg.norm(np.array(new_centroids) - np.array(self.centroids))
                print(f'[K-Means] iteration: {i}/{self.max_iter}', end='')
                print(f' - delta-centroid norm: {diff_norm}')
            self.centroids = new_centroids
            
        self.centroids = np.array(self.centroids)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)

    def set_params(self, n_clusters: int = None, max_iter: int = None) -> None:
        if n_clusters is not None: self.n_clusters = int(n_clusters)
        if max_iter is not None: self.max_iter = int(max_iter)


class KMeansClusteringPlus(Estimator, Unsupervised):
    
    """
    K-means++ is an improved version of the original K-means clustering algorithm, 
    designed to address some of its shortcomings and produce more robust and 
    efficient clustering results. K-means++ was introduced by David Arthur and 
    Sergei Vassilvitskii in a 2007 research paper titled "k-means++: 
    The Advantages of Careful Seeding."
    
    Parameters
    ----------
    ``n_clusters`` : Number of clusters \n
    ``max_iter`` : Number of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 max_iter: int = 100,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False

    def _initialize_centroids(self, X: np.ndarray) -> None:
        self.centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = [min([np.linalg.norm(x - c) ** 2 for c in self.centroids]) for x in X]
            distances = np.array(distances)
            
            probs = distances / distances.sum()
            next_centroid = np.random.choice(X.shape[0], p=probs)
            self.centroids.append(X[next_centroid])
        
    def fit(self, X: np.ndarray) -> 'KMeansClusteringPlus':
        self._initialize_centroids(X)
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0: continue
                self.centroids[i] = np.mean(cluster_points, axis=0)
            
            if self.verbose and i % 10 == 0: 
                print(f'[K-Means++] iteration: {i}/{self.max_iter}', end='')
        
        self.centroids = np.array(self.centroids)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)

    def set_params(self, n_clusters: int = None, max_iter: int = None) -> None:
        if n_clusters is not None: self.n_clusters = int(n_clusters)
        if max_iter is not None: self.max_iter = int(max_iter)


class KMediansClustering(Estimator, Unsupervised):
    
    """
    K-median clustering is a data clustering method that divides a dataset into 
    K clusters with each cluster having a median point as its representative. 
    It uses distance metrics like Manhattan or Euclidean distance to minimize 
    the sum of distances between data points and their cluster medians, 
    making it less sensitive to outliers and adaptable to non-Euclidean data.
    
    Parameters
    ----------
    ``n_clusters`` : Number of clusters \n
    ``max_iter`` : Number of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 max_iter: int = 100,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False
        
    def fit(self, X: np.ndarray) -> 'KMediansClustering':
        self.medians = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            distances = np.abs(X[:, np.newaxis] - self.medians)
            labels = np.argmin(distances.sum(axis=2), axis=1)

            new_medians = [np.median(X[labels == i], axis=0) for i in range(self.n_clusters)]
            new_medians = np.array(new_medians)
            
            if np.all(np.array(new_medians) == self.medians): 
                if self.verbose: print(f'[K-Medians] Ealry convergence at itertaion {i}')
                break
            
            if self.verbose and i % 10 == 0: 
                diff_norm = np.linalg.norm(np.array(new_medians) - np.array(self.medians))
                print(f'[K-Medians] iteration: {i}/{self.max_iter}', end='')
                print(f' - delta-centroid norm: {diff_norm}')
            
            self.medians = new_medians
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        distances = np.abs(X[:, np.newaxis] - self.medians)
        labels = np.argmin(distances.sum(axis=2), axis=1)
        return labels
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, n_clusters: int = None, max_iter: int = None) -> None:
        if n_clusters is not None: self.n_clusters = int(n_clusters)
        if max_iter is not None: self.max_iter = int(max_iter)

