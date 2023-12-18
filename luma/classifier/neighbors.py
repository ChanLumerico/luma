from scipy.spatial import distance_matrix
import numpy as np

from luma.interface.util import Matrix
from luma.interface.super import Estimator, Supervised, Evaluator
from luma.interface.util import NearestNeighbors
from luma.interface.exception import NotFittedError
from luma.metric.classification import Accuracy


__all__ = ['KNNClassifier', 'AdaptiveKNNClassifier']


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

    def fit(self, X: Matrix, y: Matrix) -> 'KNNClassifier':
        self._neighbors = NearestNeighbors(X, self.n_neighbors)
        self._y = y
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        predictions = []
        
        for x in X:
            distances = np.linalg.norm(self._neighbors.data - x, axis=1)
            nearest_neighbor_ids = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbor_labels = self._y[nearest_neighbor_ids]
            most_common_label = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(most_common_label)

        return np.array(predictions)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, n_neighbors: int = None) -> None:
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)


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
    ``n_density`` : Number of nearest neighbors to estimate the local density \n
    ``min_neighbors`` : Minimum number of neighbors to be considered for averaging \n
    ``max_neighbors`` : Maximum number of neighbors to be considered
    
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

    def fit(self, X: Matrix, y: Matrix) -> 'AdaptiveKNNClassifier':
        self._X = X
        self._y = y
        
        self.dist_matrix = distance_matrix(X, X)
        self.local_density = np.sort(self.dist_matrix, axis=1)
        self.local_density = self.local_density[:, self.n_density]
        
        self.k_values = np.clip(np.ceil(self.max_neighbors / self.local_density), 
                           self.min_neighbors, self.max_neighbors).astype(int)
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        dist_matrix = distance_matrix(X, self._X)
        
        predictions = []
        for point_distances in dist_matrix:
            nearest_indices = np.argsort(point_distances)
            votes = np.zeros(len(np.unique(self._y)))

            for idx, _ in enumerate(point_distances[nearest_indices]):
                if idx >= self.k_values[nearest_indices[idx]]: break
                votes[self._y[nearest_indices[idx]]] += 1

            most_common = np.argmax(votes)
            predictions.append(most_common)

        return np.array(predictions)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, 
                   n_density: int = None,
                   min_neighbors: int = None,
                   max_neighbors: int = None) -> None:
        if n_density is not None: self.n_density = int(n_density)
        if min_neighbors is not None: self.min_neighbors = int(min_neighbors)
        if max_neighbors is not None: self.max_neighbors = int(max_neighbors)

