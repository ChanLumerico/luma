import numpy as np

from luma.interface.super import Estimator, Supervised, Evaluator
from luma.interface.util import NearestNeighbors
from luma.interface.exception import NotFittedError
from luma.metric.regression import MeanSquaredError


__all__ = ['KNNRegressor']


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
    ``n_neighbors`` : Number of neighbors to be considered close
    
    """
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._y = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNRegressor':
        self._neighbors = NearestNeighbors(X, self.n_neighbors)
        self._y = y
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        predictions = []

        for point in X:
            distances = np.linalg.norm(self._neighbors.data - point, axis=1)
            nearest_neighbor_ids = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbor_values = self._y[nearest_neighbor_ids]
            average_value = np.mean(nearest_neighbor_values)
            predictions.append(average_value)

        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, n_neighbors: int = None) -> None:
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)

