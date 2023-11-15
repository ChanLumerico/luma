from typing import *
import numpy as np

from LUMA.Interface.Super import _Distance


class Euclidean(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)


class Manhattan(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(x - y))


class Chebyshev(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.max(np.abs(x - y))


class Minkowski(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray, p: int | float) -> float:
        if p is None: raise ValueError('[Minkowski] Empty p-value!'); return
        return np.power(np.sum(np.abs(x - y) ** p), 1 / p)


class CosineSimilarity(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


class Correlation(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        correlation_matrix = np.corrcoef(x, y)
        return 1 - correlation_matrix[0, 1]


class Mahalanobis(_Distance):
    @classmethod
    def distance(x: np.ndarray, y: np.ndarray, cov: np.ndarray) -> float:
        diff = x - y
        inv_covariance_matrix = np.linalg.inv(cov)
        mahalanobis_distance = np.sqrt(diff.T @ inv_covariance_matrix @ diff)
        return mahalanobis_distance

