from scipy.spatial import distance_matrix
import numpy as np

from luma.interface.util import Matrix
from luma.core.super import Estimator, Supervised, Evaluator
from luma.interface.util import NearestNeighbors
from luma.interface.exception import NotFittedError
from luma.metric.regression import MeanSquaredError


__all__ = (
    'KNNRegressor', 
    'AdaptiveKNNRegressor', 
    'WeightedKNNRegressor'
)


class KNNRegressor(Estimator, Supervised):
     
    """
    The K-Nearest Neighbors (KNN) regressor is a straightforward and 
    intuitive machine learning algorithm that predicts the value of 
    a new data point based on the average values of its closest 
    neighbors. It functions by calculating the distance (such as 
    Euclidean distance) between the new point and all points in the 
    training set, determining the 'k' nearest neighbors, and then 
    averaging the values of these neighbors to predict the continuous 
    output.
    
    Parameters
    ----------
    `n_neighbors` : Number of neighbors to be considered close
    
    """
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._y = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'KNNRegressor':
        self._neighbors = NearestNeighbors(X, self.n_neighbors)
        self._y = y
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        predictions = []

        for point in X:
            distances = np.linalg.norm(self._neighbors.data - point, axis=1)
            nearest_neighbor_ids = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbor_values = self._y[nearest_neighbor_ids]
            average_value = np.mean(nearest_neighbor_values)
            predictions.append(average_value)

        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class AdaptiveKNNRegressor(Estimator, Supervised):
    
    """
    The Adaptive K-Nearest Neighbors (AdaKNN) Regressor is an enhanced version 
    of the traditional KNN algorithm for regression. It predicts the value of a 
    new data point based on the average values of its neighbors, where the 
    number of neighbors (k) is adaptively determined based on the local density 
    of the training data. This method allows the algorithm to be more flexible 
    and effective, particularly in datasets with varying densities.

    Parameters
    ----------
    `n_density` : Number of nearest neighbors to estimate the local density
    `min_neighbors` : Minimum number of neighbors to be considered for averaging
    `max_neighbors` : Maximum number of neighbors to be considered

    """
    
    def __init__(self, 
                 n_density: int = 10, 
                 min_neighbors: int = 5, 
                 max_neighbors: int = 20):
        self.n_density = n_density
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self._X = None
        self._y = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'AdaptiveKNNRegressor':
        self._X = X
        self._y = y
        self.dist_matrix = distance_matrix(X, X)
        self.local_density = np.sort(self.dist_matrix, axis=1)
        self.local_density = self.local_density[:, self.n_density]
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        dist_matrix = distance_matrix(X, self._X)
        k_values = np.clip(np.ceil(self.max_neighbors / self.local_density), 
                           self.min_neighbors, self.max_neighbors).astype(int)
        
        predictions = []
        for point_distances in dist_matrix:
            nearest_indices = np.argsort(point_distances)
            adaptive_neighbors = min(len(nearest_indices), k_values[nearest_indices[0]])
            nearest_indices = nearest_indices[:adaptive_neighbors]

            average_value = np.mean(self._y[nearest_indices])
            predictions.append(average_value)

        return Matrix(predictions)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class WeightedKNNRegressor(Estimator, Supervised):
    
    """
    The Weighted KNN Regressor is an adaptation of the k-Nearest Neighbors algorithm
    tailored for regression tasks, where neighbors influence the predicted value
    based on their distance from the query point. Closer neighbors exert greater
    influence due to higher weights, commonly calculated using inverse distance weighting.
    This method improves prediction precision, particularly in datasets with
    irregular distributions.
    
    Parameters
    ----------
    `n_neighbors` : Number of neighbors to be considered close
    
    """
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._X = None
        self._y = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'WeightedKNNRegressor':
        self._X = X
        self._y = y
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise not NotFittedError(self)
        predictions = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = np.linalg.norm(self._X - x, axis=1)
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]

            weights = 1 / (distances[nearest_neighbors] + 1e-5)
            weighted_sum = np.dot(weights, self._y[nearest_neighbors])
            total_weight = np.sum(weights)
            predictions[i] = weighted_sum / total_weight

        return predictions

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

