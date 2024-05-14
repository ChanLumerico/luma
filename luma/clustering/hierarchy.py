from typing import Literal, Optional, Self, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

from luma.interface.typing import Matrix
from luma.core.super import Estimator, Evaluator, Unsupervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.clustering.kmeans import KMeansClusteringPlus
from luma.metric.clustering import SilhouetteCoefficient


__all__ = ("AgglomerativeClustering", "DivisiveClustering")


class AgglomerativeClustering(Estimator, Unsupervised):
    """
    Agglomerative clustering is a hierarchical clustering technique that starts
    by treating each data point as a single cluster. It iteratively merges the
    closest pairs of clusters based on a chosen distance metric. This process
    continues until a specified number of clusters is reached or all points
    are merged into a single cluster.

    Parameters
    ----------
    `n_clusters` : int, default=2
        Number of clusters to estimate
    `linkage` : {"single", "complete", "average"}, default="single"
        Linkage method

    Methods
    -------
    Plot hierarchical dendrogram:
    ```py
    def plot_dendrogram(
        self,
        ax: Optional[plt.Axes] = None,
        hide_indices: bool = True,
        show: bool = False
    ) -> plt.Axes
    ```
    Examples
    --------
    >>> agg = AgglomerativeClustering()
    >>> agg.fit(X, y)
    >>> lables = agg.labels # Get assigned labels

    """

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: Literal["single", "complete", "average"] = "single",
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self._X = None
        self._fitted = False

        self.set_param_ranges({"n_clusters": ("0<,+inf", int)})
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self._X = X
        n_samples = X.shape[0]
        dist_matrix = squareform(pdist(X, metric="euclidean"))
        clusters = {i: [i] for i in range(n_samples)}

        while len(clusters) > self.n_clusters:
            closest_pair, _ = self._find_closest_clusters(dist_matrix, clusters)

            self._merge_clusters(clusters, closest_pair)
            self._update_distance_matrix(dist_matrix, closest_pair, clusters)

        self.clusters = clusters
        self._n = n_samples

        self._fitted = True
        return self

    def _find_closest_clusters(
        self, dist_matrix: Matrix, clusters: dict
    ) -> Tuple[tuple, float]:
        min_dist = np.inf
        closest_pair = (None, None)

        for i in clusters:
            for j in clusters:
                if i != j and dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
                    closest_pair = (i, j)

        return closest_pair, min_dist

    def _merge_clusters(self, clusters: Matrix, pair: tuple) -> None:
        i, j = pair
        clusters[i].extend(clusters[j])
        del clusters[j]

    def _update_distance_matrix(
        self, dist_matrix: Matrix, pair: tuple, clusters: dict
    ) -> None:
        i, j = pair
        for k in clusters:
            if k == i:
                continue
            value = None
            if self.linkage == "single":
                value = min(dist_matrix[i, k], dist_matrix[j, k])
            elif self.linkage == "complete":
                value = max(dist_matrix[i, k], dist_matrix[j, k])
            elif self.linkage == "average":
                value = np.mean([dist_matrix[i, k], dist_matrix[j, k]])
            else:
                raise UnsupportedParameterError(self.linkage)

            dist_matrix[i, k] = dist_matrix[k, i] = value
        dist_matrix[i, i] = np.inf

    def _assign_labels(self, clusters: dict, n_samples: int) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)

        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_samples in enumerate(clusters.values()):
            for sample in cluster_samples:
                labels[sample] = cluster_id

        return labels

    def plot_dendrogram(
        self,
        ax: Optional[plt.Axes] = None,
        hide_indices: bool = True,
        show: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        m = self._X.shape[0]
        labels = [""] * m if hide_indices else list(range(m))
        Z = linkage(self._X, method=self.linkage)
        dendrogram(Z, labels=labels)

        ax.set_xlabel("Samples")
        ax.set_ylabel("Distance")
        ax.set_title("Dendrogram")
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

    @property
    def labels(self) -> Matrix:
        return self._assign_labels(self.clusters, self._n)

    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)


class DivisiveClustering(Estimator, Unsupervised):
    """
    Divisive clustering is a "top-down" approach in hierarchical clustering,
    starting with all data points in one large cluster. It iteratively splits
    clusters into smaller ones based on similarity measures, typically using
    algorithms like K-means for splitting. The process continues until a
    specified number of clusters is reached or other stopping criteria are met.
    This method is particularly effective for datasets with well-defined,
    separate clusters.

    Parameters
    ----------
    `n_clusters` : int, default=2
        Number of clusters to estimate

    Examples
    --------
    >>> div = DivisiveClustering()
    >>> div.fit(X)
    >>> labels = div.labels # Get assigned labels

    """

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.clusters = []
        self._X = None
        self._fitted = False

        self.set_param_ranges({"n_clusters": ("0<,+inf", int)})
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        initial_cluster = [0] * X.shape[0]
        self._X = X
        self.clusters.append(initial_cluster)
        self._recursive_split(X)

        self._fitted = True
        return self

    def _recursive_split(self, X: Matrix, depth: int = 0) -> None:
        if depth == self.n_clusters - 1:
            return

        max_cluster_idx = self._find_largest_cluster()
        split_idx = [i for i, v in enumerate(self.clusters[-1]) if v == max_cluster_idx]

        if len(split_idx) > 1:
            kmeans = KMeansClusteringPlus(n_clusters=2).fit(X[split_idx])
            new_labels = kmeans.predict(X[split_idx])

            new_cluster = self.clusters[-1].copy()
            new_cluster_id = max(new_cluster) + 1
            for idx, label in zip(split_idx, new_labels):
                new_cluster[idx] = new_cluster_id if label == 1 else max_cluster_idx

            self.clusters.append(new_cluster)
            self._recursive_split(X, depth + 1)

    def _find_largest_cluster(self) -> Matrix:
        return np.argmax(np.bincount(self.clusters[-1]))

    @property
    def labels(self) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        return Matrix(self.clusters[-1])

    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)
