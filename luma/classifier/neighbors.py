from typing import Self
from scipy.spatial import distance_matrix
import numpy as np

from luma.core.super import Estimator, Supervised, Evaluator
from luma.interface.typing import Matrix
from luma.interface.util import NearestNeighbors
from luma.interface.exception import NotFittedError
from luma.metric.classification import Accuracy


__all__ = ("KNNClassifier", "AdaptiveKNNClassifier", "WeightedKNNClassifier")


class KNNClassifier(Estimator, Supervised):
    """
    The K-Nearest Neighbors (KNN) classifier is a simple and intuitive
    machine learning algorithm that classifies a new data point based on
    the majority class among its closest neighbors. It operates by
    calculating the distance (such as Euclidean distance) between the
    new point and all points in the training set, identifying the 'k'
    nearest neighbors, and then performing a majority vote among these
    neighbors to determine the class label.

    Parameters
    ----------
    `n_neighbors` : int, default=5
        Number of neighbors to be considered close

    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._y = None
        self._fitted = False

        self.set_param_ranges({"n_neighbors": ("0,+inf", int)})
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        self._neighbors = NearestNeighbors(X, self.n_neighbors)
        self._y = y
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        predictions = []

        for x in X:
            distances = np.linalg.norm(self._neighbors.data - x, axis=1)
            nearest_neighbor_ids = np.argsort(distances)[: self.n_neighbors]
            nearest_neighbor_labels = self._y[nearest_neighbor_ids]

            most_common_label = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(most_common_label)

        return Matrix(predictions)

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        proba = []

        for x in X:
            distances = np.linalg.norm(self._neighbors.data - x, axis=1)
            nearest_neighbor_ids = np.argsort(distances)[: self.n_neighbors]
            nearest_neighbor_labels = self._y[nearest_neighbor_ids]

            class_votes = np.bincount(
                nearest_neighbor_labels, minlength=np.max(self._y) + 1
            )

            class_probabilities = class_votes / self.n_neighbors
            proba.append(class_probabilities)

        return Matrix(proba)

    def score(self, X: Matrix, y: Matrix, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class AdaptiveKNNClassifier(Estimator, Supervised):
    """
    The Adaptive K-Nearest Neighbors (AdaKNN) Classifier is an extension
    of the conventional KNN algorithm for classification. It classifies a
    new data point based on the majority class among its neighbors, where
    the number of neighbors (k) is adaptively determined based on the local
    density of the training data. This adaptive approach allows the algorithm
    to be more flexible and effective, especially in datasets with varying
    densities.

    Parameters
    ----------
    `n_density` : int, default=10
        Number of nearest neighbors to estimate the local density
    `min_neighbors` : int, default=5
        Minimum number of neighbors to be considered for averaging
    `max_neighbors` : int, default=20
        Maximum number of neighbors to be considered

    """

    def __init__(
        self, n_density: int = 10, min_neighbors: int = 5, max_neighbors: int = 20
    ):
        self.n_density = n_density
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self._X = None
        self._y = None
        self._fitted = False

        self.set_param_ranges(
            {
                "n_density": ("0<,+int", int),
                "min_neighbors": ("0<,+int", int),
                "max_neighbors": ("0<,+int", int),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        self._X = X
        self._y = y

        self.dist_matrix = distance_matrix(X, X)
        self.local_density = np.sort(self.dist_matrix, axis=1)
        self.local_density = self.local_density[:, self.n_density]

        self.k_values = np.clip(
            np.ceil(self.max_neighbors / self.local_density),
            self.min_neighbors,
            self.max_neighbors,
        ).astype(int)

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        dist_matrix = distance_matrix(X, self._X)

        predictions = []
        for point_distances in dist_matrix:
            nearest_indices = np.argsort(point_distances)
            votes = np.zeros(len(np.unique(self._y)))

            for idx, _ in enumerate(point_distances[nearest_indices]):
                if idx >= self.k_values[nearest_indices[idx]]:
                    break
                votes[self._y[nearest_indices[idx]]] += 1

            most_common = np.argmax(votes)
            predictions.append(most_common)

        return Matrix(predictions)

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        dist_matrix = distance_matrix(X, self._X)

        proba_predictions = []
        for i, point_distances in enumerate(dist_matrix):
            nearest_indices = np.argsort(point_distances)
            k_value = min(self.k_values[i], len(nearest_indices))
            nearest_indices = nearest_indices[:k_value]

            votes = np.zeros(np.max(self._y) + 1)
            for idx in nearest_indices:
                votes[self._y[idx]] += 1

            class_probabilities = votes / votes.sum()
            proba_predictions.append(class_probabilities)

        return Matrix(proba_predictions)

    def score(self, X: Matrix, y: Matrix, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class WeightedKNNClassifier(Estimator, Supervised):
    """
    The Weighted KNN Classifier is a variation of the k-Nearest Neighbors algorithm
    where neighbors contribute to the classification decision based on their distance
    from the query point. Closer neighbors have more influence as they are assigned
    higher weights, typically using inverse distance weighting. This approach enhances
    prediction accuracy, especially in unevenly distributed datasets.

    Parameters
    ----------
    `n_neighbors` : int, default=5
        Number of neighbors to be considered close

    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._X = None
        self._y = None
        self._fitted = False

        self.set_param_ranges({"n_neighbors": ("0,+inf", int)})
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        self._X = X
        self._y = y

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        predictions = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            distances = np.linalg.norm(self._X - x, axis=1)
            nearest_neighbors = np.argsort(distances)[: self.n_neighbors]

            weights = 1 / (distances[nearest_neighbors] + 1e-5)
            weighted_votes = np.zeros(np.unique(self._y).shape[0])
            for idx, neighbor in enumerate(nearest_neighbors):
                weighted_votes[self._y[neighbor]] += weights[idx]

            predictions[i] = np.argmax(weighted_votes)

        return predictions.astype(int)

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        proba_predictions = []

        for x in X:
            distances = np.linalg.norm(self._X - x, axis=1)
            nearest_neighbor_indices = np.argsort(distances)[: self.n_neighbors]

            weights = 1 / (distances[nearest_neighbor_indices] + 1e-5)
            all_classes = np.unique(self._y)
            weighted_votes = np.zeros(len(all_classes))

            for i, cl in enumerate(all_classes):
                class_weights = weights[self._y[nearest_neighbor_indices] == cl]
                weighted_votes[i] = np.sum(class_weights)

            class_probabilities = weighted_votes / np.sum(weighted_votes)
            proba_predictions.append(class_probabilities)

        return Matrix(proba_predictions)

    def score(self, X: Matrix, y: Matrix, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
