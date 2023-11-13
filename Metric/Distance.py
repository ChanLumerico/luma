from typing import *
import numpy as np

from LUMA.Interface.Super import _Distance


class Euclidean(_Distance):
    
    """
    Suitable for continuous data. \n
    Most used distance metric.
    """
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)


class Manhattan(_Distance):
    
    """
    Suitable for data with outliers or in cases where movement 
    can only occur along grid lines.
    """
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(x - y))


class Chebyshev(_Distance):
    
    """Emphasizes differences in the largest dimension."""
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.max(np.abs(x - y))


class Minkowski(_Distance):
    
    """
    Generalization of Euclidean and Manhattan distances. \n
    Euclidean distance is a special case for `p=2`, 
    Manhattan distance for ``p=1``
    """
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray, p: int | float) -> float:
        if p is None: raise ValueError('[Minkowski] Empty p-value!'); return
        return np.power(np.sum(np.abs(x - y) ** p), 1 / p)


class CosineSimilarity(_Distance):
    
    """
    Measures the cosine of the angle between two non-zero vectors. \n
    Suitable for text data and other high-dimensional data.
    """
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


class Correlation(_Distance):
    
    """
    Measures the similarity between two vectors, taking into account 
    their linear relationships.
    """
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        correlation_matrix = np.corrcoef(x, y)
        return 1 - correlation_matrix[0, 1]


class Mahalanobis(_Distance):
    
    """
    Takes into account the correlations between variables and 
    the variances of individual variables. \n
    Useful when dealing with multivariate data.
    """
    
    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray, cov: np.ndarray) -> float:
        diff = x - y
        inv_covariance_matrix = np.linalg.inv(cov)
        mahalanobis_distance = np.sqrt(diff.T @ inv_covariance_matrix @ diff)
        return mahalanobis_distance

