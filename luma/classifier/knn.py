import numpy as np

from luma.interface.super import Estimator, Supervised, Evaluator
from luma.interface.util import NearestNeighbors
from luma.interface.exception import NotFittedError
from luma.metric.classification import Accuracy


__all__ = ['KNNClassifier']


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
    ``n_neighbors`` : Number of neighbors to be considered close
    
    """
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._y = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        self._neighbors = NearestNeighbors(X, self.n_neighbors)
        self._y = y
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        predictions = []
        
        for x in X:
            distances = np.linalg.norm(self._neighbors.data - x, axis=1)
            nearest_neighbor_ids = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbor_labels = self._y[nearest_neighbor_ids]
            most_common_label = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(most_common_label)

        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, n_neighbors: int = None) -> None:
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)

