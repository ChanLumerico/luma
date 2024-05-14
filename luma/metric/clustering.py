from typing import Optional
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np

from luma.core.super import Evaluator, Visualizer
from luma.interface.util import SilhouetteUtil, DBUtil
from luma.interface.typing import Matrix, Vector, ClassType


__all__ = ("SilhouetteCoefficient", "DaviesBouldin", "Inertia")


class SilhouetteCoefficient(Evaluator, Visualizer):
    """
    The Silhouette Coefficient is a measure used to evaluate the quality of
    clusters in a dataset. It calculates how similar each data point is to its
    own cluster compared to other clusters. The coefficient ranges from -1
    (poorly clustered) to +1 (well clustered), with values near 0 indicating
    overlapping clusters. High average scores across a dataset suggest clear,
    well-separated clusters.

    Parameters
    ----------
    `data` : Matrix
        Original data
    `labels` : Vector
        Labels assigned by clustering estimator

    Examples
    --------
    With Instantiation
    ```py
        sil = SilhouetteCoefficient(data, labels)
        score = sil.score(data, labels) # compute() is a static method
        sil.plot(...)
    ```
    Without Instantiation
    ```py
        score = SilhouetteCoefficient.score(data, labels)
        SilhouetteCoefficient.plot(...) # Error; plot() is an instance method
    ```
    """

    def __init__(self, data: Matrix, labels: Vector) -> None:
        self.data = data
        self.labels = labels
        self.dist = squareform(pdist(self.data))

    @staticmethod
    def score(data: Matrix, labels: Matrix) -> Matrix[float]:
        scores = []
        dist = squareform(pdist(data))
        for idx, label in enumerate(labels):
            util = SilhouetteUtil(idx, label, labels, dist)
            a = util.avg_dist_within
            b = util.avg_dist_others
            score = (b - a) / max(a, b) if max(a, b) != 0 else 0
            scores.append(score)

        return np.mean(scores)

    def _individual_silhouette(self) -> Matrix:
        silhouette_values = []
        for idx, label in enumerate(self.labels):
            util = SilhouetteUtil(idx, label, self.labels, self.dist)
            a = util.avg_dist_within
            b = util.avg_dist_others
            score = (b - a) / max(a, b) if max(a, b) != 0 else 0
            silhouette_values.append(score)

        return Matrix(silhouette_values)

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        sample_silhouette = self._individual_silhouette()
        y_lower = 10

        for i in range(len(np.unique(self.labels))):
            values = sample_silhouette[self.labels == i]
            values.sort()

            size_cluster_i = values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / len(set(self.labels)))
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            y_lower = y_upper + 10

        ax.set_title("Silhouette Coefficient Plot")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=self.score(self.data, self.labels), color="red", linestyle="--")

        ax.set_yticks([])
        ax.set_xticks(np.arange(0.0, 1.2, 0.1))
        plt.tight_layout()

        if show:
            plt.show()
        return ax


@ClassType.non_instantiable()
class DaviesBouldin(Evaluator):
    """
    The Davies-Bouldin Index (DBI) is a metric for evaluating clustering
    algorithms. It compares the average distance within clusters to the
    distance between clusters. Lower DBI values indicate better clustering,
    with compact and well-separated clusters.

    Parameters
    ----------
    `data` : Matrix
        Original data
    `labels` : Vector
        Labels assigned by clustering estimator

    """

    @classmethod
    def score(cls, data: Matrix, labels: Vector) -> float:
        util = DBUtil(data=data, labels=labels)
        centroids = util.cluster_centroids
        scatter = util.within_cluster_scatter
        separation = util.separation

        n_clusters = len(centroids)
        db_values = np.zeros(n_clusters)

        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i == j:
                    continue
                ratio = (scatter[i] + scatter[j]) / separation[i, j]
                if ratio > max_ratio:
                    max_ratio = ratio

            db_values[i] = max_ratio
        db_index = np.mean(db_values)

        return db_index


@ClassType.non_instantiable()
class Inertia(Evaluator):
    """
    Inertia in clustering quantifies the compactness of clusters by measuring
    the sum of squared distances between data points and their respective
    cluster centroids. It serves as a key metric to evaluate the quality of
    cluster assignments in algorithms like K-means. A lower inertia value
    indicates more cohesive clusters, where data points are closer to their
    centroids. However, minimizing inertia excessively can lead to overfitting,
    with a large number of clusters.

    Parameters
    ----------
    `data` : Matrix
        Original data
    `centroids`: Matrix
        Centroids(or medoids for certain algorithm)

    """

    @classmethod
    def score(cls, data: Matrix, centroids: Matrix) -> float:
        sq_dist = (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2
        dist = np.sqrt(sq_dist.sum(axis=2))

        closest = np.argmin(dist, axis=1)
        inertia = sum((dist[i, closest[i]] ** 2) for i in range(len(data)))

        return inertia
