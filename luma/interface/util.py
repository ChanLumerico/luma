from typing import Any
import numpy as np


__all__ = (
    'Matrix', 
    'Vector', 
    'Constant', 
    'TreeNode', 
    'NearestNeighbors', 
    'SilhouetteUtil', 
    'DBUtil'
)


class Matrix(np.ndarray):

    """
    Internal class that extends `numpy.ndarray`.

    This class provides a way to create matrix objects that have 
    all the capabilities of numpy arrays with the potential for 
    additional functionalities and readability.s
    
    Example
    -------
    >>> m = Matrix([1, 2, 3])
    >>> isinstance(m, numpy.ndarray) # True
    
    """
    
    def __new__(cls, array_like: Any) -> 'Matrix':
        if isinstance(array_like, list): obj = np.array(array_like)
        else: obj = array_like    
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if obj is None: return


class Vector(Matrix):
    
    """
    Internal class for vector that extends `Matrix`.
    
    This class represents a single row/column vector with its
    type of `numpy.ndarray` or `Matrix`.
    
    """
    
    def __new__(cls, array_like: Any) -> 'Vector':
        if isinstance(array_like, list): obj = Matrix(array_like)
        else: obj = array_like    
        return obj


class Constant:
    
    """
    A placeholder class for constant type.
    
    This class encompasses `int` and `float`.
    """
    
    def __new__(cls, value: int | float) -> 'Constant':
        return float(value)


class TreeNode:
    
    """
    Internal class for node used in tree-based models.
    
    Parameters
    ----------
    `feature` : Feature of node
    `threshold` : Threshold for split point
    `left` : Left-child node
    `right` : Right-child node
    `value` : Most popular label of leaf node
    
    """
    
    def __init__(self,
                 feature: int = None,
                 threshold: float = None,
                 left: 'TreeNode' = None,
                 right: 'TreeNode' = None,
                 value: int | float = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def isLeaf(self) -> bool:
        return self.value is not None


class NearestNeighbors:
    
    """
    Internal class for computing nearest neighbors of given data.
    
    Parameters
    ----------
    `data` : Data to be handled
    `n_neighbors` : Number of nearest neighbors
    
    """
    
    def __init__(self,
                 data: Matrix,
                 n_neighbors: int) -> None:
        self.data = data
        self.n_neighbors = n_neighbors
        self._size = data.shape[0]
    
    @property
    def index_matrix(self) -> Matrix:
        data = self.data
        dist = np.linalg.norm(data[:, np.newaxis, :] - data, axis=2)
        sorted_indices = np.argsort(dist, axis=1)
        return sorted_indices[:, 1:self.n_neighbors + 1]
    
    @property
    def adjacent_matrix(self) -> Matrix:
        indices = self.index_matrix
        adj_mat = np.zeros((self._size, self._size))
        for i in range(self._size):
            adj_mat[i, indices[i]] = 1
        
        return adj_mat.astype(int)


class SilhouetteUtil:
    
    """
    Internal class for computing various distances used in 
    Silhouette Coefficient calculation.
    
    Parameters
    ----------
    `idx` : Index of a single data point
    `cluster` : Current cluster number
    `labels` : Labels assigned by clustering estimator
    `distances` : Square-form distance matrix of the data
    
    """
    
    def __init__(self, 
                 idx: int,
                 cluster: int,
                 labels: Matrix,
                 distances: Matrix) -> None:
        self.idx = idx
        self.cluster = cluster
        self.labels = labels
        self.distances = distances

    @property
    def avg_dist_others(self) -> Matrix:
        others = set(self.labels) - {self.cluster}
        sub_avg = [np.mean(self.distances[self.idx][self.labels == other]) 
                   for other in others]

        return np.mean(sub_avg)

    @property
    def avg_dist_within(self) -> Matrix | int:
        within_cluster = self.distances[self.idx][self.labels == self.cluster]
        if len(within_cluster) <= 1: return 0
        return np.mean([dist for dist in within_cluster if dist != 0])


class DBUtil:
    
    """
    Internal class for supporting Davies-Bouldin Index (DBI) computation.
    
    Parameters
    ----------
    `data` : Original data
    `labels` : Labels assigned by clustering estimator
    
    """
    
    def __init__(self,
                 data: Matrix, 
                 labels: Matrix) -> None:
        self.data = data
        self.labels = labels
    
    @property
    def cluster_centroids(self) -> Matrix:
        unique_labels = np.unique(self.labels)
        centroids = np.array([self.data[self.labels == label].mean(axis=0) 
                              for label in unique_labels])
        return centroids

    @property
    def within_cluster_scatter(self) -> Matrix:
        centroids = self.cluster_centroids
        scatter = np.zeros(len(centroids))
        
        for i, centroid in enumerate(centroids):
            cluster_points = self.data[self.labels == i]
            diff_sq =  (cluster_points - centroid) ** 2
            scatter[i] = np.mean(np.sqrt(np.sum(diff_sq, axis=1)))
            
        return scatter

    @property
    def separation(self) -> Matrix:
        centroids = self.cluster_centroids
        n_clusters = len(centroids)
        separation = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j: continue
                diff_sq = (centroids[i] - centroids[j]) ** 2
                separation[i, j] = np.sqrt(np.sum(diff_sq))
                    
        return separation

