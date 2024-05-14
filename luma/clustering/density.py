from typing import Literal, Optional, Self, Tuple, List
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

from luma.core.super import Estimator, Evaluator, Unsupervised
from luma.interface.typing import Matrix, Vector, Scalar
from luma.interface.exception import NotFittedError, NotConvergedError
from luma.interface.exception import UnsupportedParameterError

from luma.metric.distance import Euclidean, Minkowski
from luma.metric.clustering import SilhouetteCoefficient


__all__ = ("DBSCAN", "OPTICS", "DENCLUE", "MeanShiftClustering")


class DBSCAN(Estimator, Unsupervised):
    """
    DBSCAN, short for Density-Based Spatial Clustering of Applications
    with Noise, is a clustering algorithm that groups points in a dataset
    by their proximity and density. It identifies clusters as areas of high
    point density, separated by regions of low density. Points in sparse
    regions are classified as noise. DBSCAN is particularly effective at
    discovering clusters of arbitrary shapes and dealing with outliers.

    Parameters
    ----------
    `epsilon` : float, default=0.1
        Radius of a neighborhood hypersphere
    `min_points` : int, default=5
        Minimum required points to form a cluster
    `metric` : {"euclidean", "minkowski"}, default="euclidean"
        Distance metric for radius measurement

    """

    def __init__(
        self,
        epsilon: float = 0.1,
        min_points: int = 5,
        metric: Literal["euclidean", "minkowski"] = "euclidean",
    ) -> None:
        self.epsilon = epsilon
        self.min_points = min_points
        self.metric = metric
        self._X = None
        self._fitted = False

        self.metric_func = None
        if self.metric == "euclidean":
            self.metric_func = Euclidean
        elif self.metric == "minkowski":
            self.metric_func = Minkowski
        else:
            raise UnsupportedParameterError(self.metric)

        self.set_param_ranges(
            {"epsilon": ("0<,+inf", None), "min_points": ("0<,+inf", int)}
        )
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self._X = X
        clusters = [0] * X.shape[0]
        curPt = 0

        for i in range(X.shape[0]):
            if clusters[i]:
                continue
            neighbors = self._generate_neighbors(X, idx=i)
            if len(neighbors) < self.min_points:
                clusters[i] = -1
            else:
                curPt += 1
                self._expand_cluster(X, neighbors, clusters, i, curPt)

        self._cluster_labels = clusters
        self._fitted = True
        return self

    def _generate_neighbors(self, X: Matrix, idx: int) -> Matrix:
        neighbors = []
        for i in range(X.shape[0]):
            if self.metric_func.compute(X[idx], X[i]) < self.epsilon:
                neighbors.append(i)

        return Matrix(neighbors)

    def _expand_cluster(
        self, X: Matrix, neighbors: Matrix, clusters: Matrix, idx: int, current: int
    ) -> None:
        i = 0
        clusters[idx] = current

        while i < len(neighbors):
            next = neighbors[i]
            if clusters[next] == -1:
                clusters[next] = current

            elif clusters[next] == 0:
                clusters[next] = current
                next_neighbors = self._generate_neighbors(X, idx=next)

                if len(next_neighbors) > self.min_points:
                    neighbors = np.concatenate([neighbors, next_neighbors], axis=0)

            i += 1

    @property
    def labels(self) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        return Matrix(self._cluster_labels)

    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)


class OPTICS(Estimator, Unsupervised):
    """
    OPTICS (Ordering Points To Identify the Clustering Structure) is an
    unsupervised learning algorithm for identifying cluster structures
    in spatial  It creates an ordered list of points based on
    core-distance and reachability-distance, allowing it to find
    clusters of varying density. Unlike algorithms like DBSCAN, it
    doesn't require a global density threshold, making it versatile
    for complex datasets. The result is often visualized as a
    reachability plot, revealing the data's clustering hierarchy and
    density variations.

    Parameters
    ----------
    `epsilon` : float, default=1.0
        Radius of neighborhood hypersphere
    `min_points` : int, default=5
        Minimum nuber of points to form a cluster
    `threshold` : float, default=1.5
        Threshold for filtering samples with large reachabilities

    """

    def __init__(
        self,
        epsilon: float = 1.0,
        min_points: int = 5,
        threshold: float = 1.5,
        verbose: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.min_points = min_points
        self.threshold = threshold
        self.verbose = verbose
        self._X = None
        self._fitted = False

        self.set_param_ranges(
            {
                "epsilon": ("0<,+inf", None),
                "min_points": ("0<,+inf", int),
                "threshold": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self._X = X
        m, _ = X.shape

        self.processed = np.full(m, False, dtype=bool)
        self.reachability = np.full(m, np.inf)
        self.ordered_points = []

        for i in range(m):
            if self.verbose and i % 50 == 0 and i:
                print(
                    f"[OPTICS] Finished for point {i}/{m}",
                    f"with reachability {self.reachability[i]}",
                )

            if self.processed[i]:
                continue
            seeds = []
            point_neighbors = self._neighbors(X, i)
            core_dist = self._core_distance(point_neighbors)

            self.processed[i] = True
            self.ordered_points.append(i)

            if not np.isinf(core_dist):
                self._update(X, core_dist, i, seeds)
                seeds.sort(key=lambda x: x[1])

                while seeds:
                    next_, _ = seeds.pop(0)
                    self.processed[next_] = True
                    self.ordered_points.append(next_)
                    next_neighbors = self._neighbors(X, next_)
                    core_dist = self._core_distance(next_neighbors)

                    if not np.isinf(core_dist):
                        self._update(X, core_dist, next_, seeds)

        self._fitted = True
        return self

    def _core_distance(self, neighbors: Matrix) -> Matrix | Scalar:
        if len(neighbors) >= self.min_points:
            return sorted(neighbors)[self.min_points - 1]
        return np.inf

    def _neighbors(self, X: Matrix, idx: int) -> Matrix:
        distances = cdist([X[idx]], X)[0]
        return distances[distances <= self.epsilon]

    def _update(self, X: Matrix, core_dist: Scalar, idx: int, seeds: list) -> None:
        distances = cdist([X[idx]], X)[0]
        for i, dist in enumerate(distances):
            if dist <= self.epsilon and not self.processed[i]:
                new_reach_dist = max(core_dist, dist)

                if np.isinf(self.reachability[i]):
                    self.reachability[i] = new_reach_dist
                    seeds.append((i, new_reach_dist))
                elif new_reach_dist < self.reachability[i]:
                    self.reachability[i] = new_reach_dist

    def plot_reachability(
        self,
        ax: Optional[plt.Axes] = None,
        color: str = "royalblue",
        show: bool = False,
    ) -> plt.Axes:
        m = range(len(self.ordered_points))
        vals = self.reachability[self.ordered_points]

        if ax is None:
            _, ax = plt.subplots()
            show = True

        ax.plot(m, vals, color=color)
        ax.fill_between(m, vals, color=color, alpha=0.5)

        ax.set_title("Reachability Plot")
        ax.set_xlim(m[0], m[-1])

        ax.set_xlabel("Order of Points")
        ax.set_ylabel("Reachability Distance")
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

    @property
    def labels(self) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        cluster_labels = np.full_like(self.reachability, -1, dtype=int)
        current_ = 0

        for point in self.ordered_points:
            if self.reachability[point] <= self.threshold:
                cluster_labels[point] = current_
            else:
                current_ += 1

        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        mapping = {original: new for new, original in enumerate(unique_labels)}
        labels = [mapping[label] if label != -1 else -1 for label in cluster_labels]

        return Vector(labels)

    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)


class DENCLUE(Estimator, Unsupervised):
    """
    DENCLUE (Density-Based Clustering) is a machine learning algorithm
    for identifying clusters in large datasets. It uses mathematical
    functions, typically Gaussian kernels, to estimate data density.
    Cluster centers are determined by the peaks of these density functions.
    The algorithm excels in handling clusters of arbitrary shapes and noise,
    but is sensitive to parameter settings.

    Parameters
    ----------
    `h` : float or {"auto"}, default="auto"
        Smoothing parameter of local density estimation
    `tol` : float, default=1e-3
        Threshold for early convergence
    `max_climb` : int, default=100
        Maximum number of climbing process for finding local maxima
    `min_density` : float, default=0.0
        Minimum local densities to be considered
    `sample_weight` : Vector, optional, default=None
        Custom individual weights for sample data

    Reference
    ---------
    1. Hinneburg, A., & Gabriel, H. H. (2007, September). Denclue 2.0: Fast
    clustering based on kernel density estimation. In International symposium
    on intelligent data analysis (pp. 70-80). Berlin, Heidelberg: Springer
    Berlin Heidelberg.

    """

    def __init__(
        self,
        h: float | Literal["auto"] = "auto",
        tol: float = 1e-3,
        max_climb: int = 100,
        min_density: float = 0.0,
        sample_weight: Vector | None = None,
    ) -> None:
        self.h = h
        self.tol = tol
        self.max_climb = max_climb
        self.min_density = min_density
        self.sample_weight = sample_weight
        self._X = None
        self._fitted = False

        self.set_param_ranges(
            {"max_climb": ("0<,+inf", int), "min_density": ("0,+inf", None)}
        )
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self._m, self._n = X.shape

        attractors: Vector = np.zeros((self._m, self._n))
        rad: Vector = np.zeros((self._m, 1))
        density: Vector = np.zeros((self._m, 1))

        if self.h == "auto":
            self.h = np.std(X) / 5
        if self.sample_weight is None:
            self.sample_weight = np.ones((self._m, 1))

        labels = -np.ones(X.shape[0])
        for i in range(self._m):
            attractors[i], density[i], rad[i] = self._climb_hill(X[i], X)

        adj_list = [[] for _ in range(self._m)]
        for i in range(self._m):
            for j in range(i + 1, self._m):
                diff = np.linalg.norm(attractors[i] - attractors[j])
                if diff <= rad[i] + rad[j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        num_clusters = 0
        for cl in self._find_connected_components(adj_list):
            max_instance = max(cl, key=lambda x: density[x])
            max_density = density[max_instance]

            if max_density >= self.min_density:
                labels[cl] = num_clusters
            num_clusters += 1

        self.labels = labels
        self._fitted = True
        return self

    def _climb_hill(self, x: Vector, X: Matrix) -> Tuple[Vector, float, float]:
        error, prob = 99.0, 0.0
        x1 = np.copy(x)

        iters = 0
        r_new, r_old, r_2old = 0.0, 0.0, 0.0
        while True:
            r_3old, r_2old, r_old = r_2old, r_old, r_new
            x0 = np.copy(x1)
            x1, density = self._step(x0, X)

            error, prob = density - prob, density
            r_new = np.linalg.norm(x1 - x0)
            radius = r_3old + r_2old + r_old + r_new

            iters += 1
            if iters > 3 and error < self.tol:
                break
            if iters == self.max_climb:
                raise NotConvergedError(self)

        return x1, prob, radius

    def _step(self, x: Vector, X: Matrix) -> Tuple[Vector, float]:
        sup_weight = 0.0
        x1 = np.zeros((1, self._n))

        for j in range(self._m):
            kernel = self._kernel_func(x, X[j], self._n)
            kernel = kernel * self.sample_weight[j] / (self.h**self._n)

            sup_weight = sup_weight + kernel
            x1 = x1 + (kernel * X[j])

        x1 = x1 / sup_weight
        density = sup_weight / np.sum(self.sample_weight)

        return x1, density

    def _kernel_func(self, xi: Vector, xj: Vector, deg: int):
        kernel = np.exp(-((np.linalg.norm(xi - xj) / self.h) ** 2) / 2)
        kernel /= (2 * np.pi) ** (deg / 2)

        return kernel

    def _find_connected_components(self, adj_list: List[list]) -> list:
        def __dfs(node: int) -> Vector:
            stack, path = [node], []
            while stack:
                vertex = stack.pop()
                if not visited[vertex]:
                    visited[vertex] = True
                    path.append(vertex)
                    stack.extend(set(adj_list[vertex]) - set(path))

            return Matrix(path)

        clusters = []
        visited = [False] * self._m
        for node in range(self._m):
            if visited[node]:
                continue
            cluster = __dfs(node)
            clusters.append(cluster)

        return clusters

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)


class MeanShiftClustering(Estimator, Unsupervised):
    """
    Mean Shift is a non-parametric, iterative clustering algorithm
    that identifies clusters in a dataset by updating candidate cluster
    centers to the mean of points within a given region (bandwidth).
    The process repeats until convergence, where centers no longer
    significantly shift. It automatically determines the number of
    clusters based on the data distribution. Mean Shift is especially
    effective in discovering clusters of arbitrary shapes and sizes.

    Parameters
    ----------
    `bandwidth` : float, default=2.0
        Window size for kernel density estimation
    `max_iter` : int, default=300
        Maximum iteration
    `tol` : float, default=1e-3
        Tolerence threshold for early convergence

    """

    def __init__(
        self,
        bandwidth: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._X = None
        self._fitted = False

        self.set_param_ranges(
            {"bandwidth": ("0<,+inf", None), "max_iter": ("0<,+inf", int)}
        )
        self.check_param_ranges()

    def fit(self, X: Matrix) -> Self:
        self._X = X
        m, _ = X.shape

        centers = np.copy(X)
        for iter in range(self.max_iter):
            new_centers = []
            for i in range(m):
                points = X[np.linalg.norm(X - centers[i], axis=1) < self.bandwidth]
                new_center = points.mean(axis=0)
                new_centers.append(new_center)
            new_centers = Matrix(new_centers)

            diff = np.linalg.norm(new_centers - centers)
            if self.verbose and iter % 10 == 0 and iter:
                print(f"[MeanShift] iteration: {iter}/{self.max_iter}", end="")
                print(f" with delta-centers-norm {diff}")

            if diff < self.tol:
                if self.verbose:
                    print(f"[MeanShift] Early-convergence occurred at", end="")
                    print(f" iteration {iter}/{self.max_iter}")
                break
            centers = new_centers

        self.centers = centers
        self._fitted = True
        return self

    @property
    def labels(self) -> Vector:
        return self.predict(self._X)

    def predict(self, X: Matrix) -> Vector:
        norm_ = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        cluster_labels = np.argmin(norm_, axis=1)

        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        mapping = {original: new for new, original in enumerate(unique_labels)}
        labels = [mapping[label] if label != -1 else -1 for label in cluster_labels]

        return Vector(labels)

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)
