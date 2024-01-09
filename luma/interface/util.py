from typing import Any, Callable
import numpy as np

from luma.interface.exception import UnsupportedParameterError


__all__ = (
    'Matrix', 
    'Vector', 
    'Scalar', 
    'TreeNode', 
    'NearestNeighbors', 
    'SilhouetteUtil', 
    'DBUtil',
    'KernelUtil'
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


class Scalar:
    
    """
    A placeholder class for scalar type.
    
    This class encompasses `int` and `float`.
    """
    
    def __new__(cls, value: int | float) -> 'Scalar':
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


class KernelUtil:
    
    """
    Internal class for kernel methods(tricks).
    
    This class facilitates transferring kernel type strings
    into actual specific kernel function.
    
    Example
    -------
    >>> util = KernelUtil(kernel='rbf')
    >>> util.kernel_func
    KernelUtil.rbf_kernel: Callable[[...], Matrix]
    
    """
    
    def __init__(self, kernel: str) -> None:
        self.kernel = kernel
    
    def linear_kernel(self, X: Matrix) -> Matrix:
        return np.dot(X, X.T)
    
    def polynomial_kernel(self, X: Matrix, 
                          gamma: float = 1.0,
                          coef: float = 0.0,
                          deg: int = 2) -> Matrix:
        return (gamma * np.dot(X, X.T) + coef) ** deg
    
    def rbf_kernel(self, X: Matrix, gamma: float = 1.0) -> Matrix:
        _left = np.sum(X ** 2, axis=1).reshape(-1, 1)
        _right = np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
        
        return np.exp(-gamma * (_left + _right))
    
    def sigmoid_kernel(self, X: Matrix,
                       gamma: float = 1.0,
                       coef: float = 0.0) -> Matrix:
        return np.tanh(gamma * np.dot(X, X.T) + coef)
    
    def laplacian_kernel(self, X: Matrix, gamma: float = 1.0) -> Matrix:
        manhattan_dists = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        return np.exp(-gamma * manhattan_dists)
    
    @property
    def kernel_func(self) -> Callable[[Matrix, Ellipsis], Matrix]:
        if self.kernel in ('linear', 'lin'):
            return self.linear_kernel
        elif self.kernel in ('poly', 'polynomial'):
            return self.polynomial_kernel
        elif self.kernel in ('rbf', 'gaussian', 'Gaussian'):
            return self.rbf_kernel
        elif self.kernel in ('sigmoid', 'tanh'):
            return self.sigmoid_kernel
        elif self.kernel in ('laplacian', 'lap'):
            return self.laplacian_kernel
        else:
            raise UnsupportedParameterError(self.kernel)

