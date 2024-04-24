import numpy as np

from luma.interface.typing import Matrix, ClassType
from luma.core.super import Distance


__all__ = (
    "Euclidean",
    "Manhattan",
    "Chebyshev",
    "Minkowski",
    "CosineSimilarity",
    "Correlation",
    "Mahalanobis",
)


@ClassType.non_instantiable()
class Euclidean(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix) -> float:
        return np.linalg.norm(x - y)


@ClassType.non_instantiable()
class Manhattan(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix) -> float:
        return np.sum(np.abs(x - y))


@ClassType.non_instantiable()
class Chebyshev(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix) -> float:
        return np.max(np.abs(x - y))


@ClassType.non_instantiable()
class Minkowski(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix, p: int | float) -> float:
        if p is None:
            raise ValueError("[Minkowski] Empty p-value!")
        return np.power(np.sum(np.abs(x - y) ** p), 1 / p)


@ClassType.non_instantiable()
class CosineSimilarity(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix) -> float:
        return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


@ClassType.non_instantiable()
class Correlation(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix) -> float:
        correlation_matrix = np.corrcoef(x, y)
        return 1 - correlation_matrix[0, 1]


@ClassType.non_instantiable()
class Mahalanobis(Distance):
    @classmethod
    def score(cls, x: Matrix, y: Matrix, cov: Matrix) -> float:
        diff = x - y
        inv_covariance_matrix = np.linalg.inv(cov)
        mahalanobis_distance = np.sqrt(diff.T @ inv_covariance_matrix @ diff)
        return mahalanobis_distance
