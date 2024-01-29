from typing import *
import numpy as np

from luma.interface.util import Matrix
from luma.core.super import Distance


__all__ = (
    'Euclidean', 
    'Manhattan', 
    'Chebyshev', 
    'Minkowski', 
    'CosineSimilarity', 
    'Correlation', 
    'Mahalanobis'
)


class Euclidean(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix) -> float:
        return np.linalg.norm(x - y)


class Manhattan(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix) -> float:
        return np.sum(np.abs(x - y))


class Chebyshev(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix) -> float:
        return np.max(np.abs(x - y))


class Minkowski(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix, p: int | float) -> float:
        if p is None: raise ValueError('[Minkowski] Empty p-value!'); return
        return np.power(np.sum(np.abs(x - y) ** p), 1 / p)


class CosineSimilarity(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix) -> float:
        return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


class Correlation(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix) -> float:
        correlation_matrix = np.corrcoef(x, y)
        return 1 - correlation_matrix[0, 1]


class Mahalanobis(Distance):
    @staticmethod
    def compute(x: Matrix, y: Matrix, cov: Matrix) -> float:
        diff = x - y
        inv_covariance_matrix = np.linalg.inv(cov)
        mahalanobis_distance = np.sqrt(diff.T @ inv_covariance_matrix @ diff)
        return mahalanobis_distance

