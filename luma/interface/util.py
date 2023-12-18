from typing import *
import numpy as np


__all__ = ['Matrix', 'TreeNode', 'NearestNeighbors']


class Matrix(np.ndarray):

    """
    Internal class that extends numpy.ndarray.

    This class provides a way to create matrix objects that have all the capabilities 
    of numpy arrays with the potential for additional functionalities and readability.
    
    Example
    -------
    >>> m = Matrix([1, 2, 3])
    >>> isinstance(m, numpy.ndarray) # True
    
    """
    
    def __new__(cls, array_like: Any) -> 'Matrix':
        if isinstance(array_like, list): obj = np.array(array_like)
        else: obj = array_like    
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None: return


class TreeNode:
    
    """
    Internal class for node used in tree-based models.
    
    Parameters
    ----------
    ``feature`` : Feature of node \n
    ``threshold`` : Threshold for split point \n
    ``left`` : Left-child node \n
    ``right`` : Right-child node \n
    ``value`` : Most popular label of leaf node
    
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
    ``data`` : Data to be handled \n
    ``n_neighbors`` : Number of nearest neighbors
    
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

