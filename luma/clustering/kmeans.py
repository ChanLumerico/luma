import numpy as np

from luma.interface.util import Matrix, Vector
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator, Unsupervised
from luma.metric.clustering import SilhouetteCoefficient


__all__ = (
    'KMeansClustering', 
    'KMeansClusteringPlus', 
    'KMediansClustering', 
    'KMedoidsClustering',
    'MiniBatchKMeansClustering',
    'FuzzyCMeansClustering'
)


class KMeansClustering(Estimator, Unsupervised):
    
    """
    K-means clustering is a machine learning algorithm that  groups similar data 
    points into clusters. It works by iteratively assigning data points to the 
    nearest cluster center (centroid) and updating the centroids based on 
    the assigned data points. This process continues 
    until convergence.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters
    `max_iter` : Number of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 max_iter: int = 100, 
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'KMeansClustering':
        init_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[init_indices]
        self._X = X
        
        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = [X[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            if np.all(Matrix(new_centroids) == self.centroids): 
                if self.verbose: print(f'[K-Means] Ealry convergence at itertaion {i}')
                break
            
            if self.verbose and i % 10 == 0: 
                diff_norm = np.linalg.norm(Matrix(new_centroids) - Matrix(self.centroids))
                print(f'[K-Means] iteration: {i}/{self.max_iter}', end='')
                print(f' - delta-centroid norm: {diff_norm}')
            self.centroids = new_centroids
            
        self.centroids = Matrix(self.centroids)
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    @property
    def labels(self) -> Matrix:
        return self.predict(self._X)
    
    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)


class KMeansClusteringPlus(Estimator, Unsupervised):
    
    """
    K-means++ is an improved version of the original K-means clustering algorithm, 
    designed to address some of its shortcomings and produce more robust and 
    efficient clustering results. K-means++ was introduced by David Arthur and 
    Sergei Vassilvitskii in a 2007 research paper titled "k-means++: 
    The Advantages of Careful Seeding."
    
    Parameters
    ----------
    `n_clusters` : Number of clusters
    `max_iter` : Number of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 max_iter: int = 100,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._X = None
        self._fitted = False

    def _initialize_centroids(self, X: Matrix) -> None:
        self.centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = [min([np.linalg.norm(x - c) ** 2 for c in self.centroids]) for x in X]
            distances = Matrix(distances)
            
            probs = distances / distances.sum()
            next_centroid = np.random.choice(X.shape[0], p=probs)
            self.centroids.append(X[next_centroid])
        
    def fit(self, X: Matrix) -> 'KMeansClusteringPlus':
        self._X = X
        self._initialize_centroids(X)
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0: continue
                self.centroids[i] = np.mean(cluster_points, axis=0)
            
            if self.verbose and i % 10 == 0: 
                print(f'[K-Means++] iteration: {i}/{self.max_iter}', end='')
        
        self.centroids = Matrix(self.centroids)
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    @property
    def labels(self) -> Matrix:
        return self.predict(self._X)
    
    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)


class KMediansClustering(Estimator, Unsupervised):
    
    """
    K-median clustering is a data clustering method that divides a dataset into 
    K clusters with each cluster having a median point as its representative. 
    It uses distance metrics like Manhattan or Euclidean distance to minimize 
    the sum of distances between data points and their cluster medians, 
    making it less sensitive to outliers and adaptable to non-Euclidean data.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters
    `max_iter` : Number of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 max_iter: int = 100,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._X = None
        self._fitted = False
        
    def fit(self, X: Matrix) -> 'KMediansClustering':
        self._X = X
        self.medians = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for i in range(self.max_iter):
            distances = np.abs(X[:, np.newaxis] - self.medians)
            labels = np.argmin(distances.sum(axis=2), axis=1)

            new_medians = [np.median(X[labels == i], axis=0) for i in range(self.n_clusters)]
            new_medians = Matrix(new_medians)
            
            if np.all(Matrix(new_medians) == self.medians): 
                if self.verbose: print(f'[K-Medians] Ealry convergence at itertaion {i}')
                break
            
            if self.verbose and i % 10 == 0: 
                diff_norm = np.linalg.norm(Matrix(new_medians) - Matrix(self.medians))
                print(f'[K-Medians] iteration: {i}/{self.max_iter}', end='')
                print(f' - delta-centroid norm: {diff_norm}')
            
            self.medians = new_medians
        
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        distances = np.abs(X[:, np.newaxis] - self.medians)
        labels = np.argmin(distances.sum(axis=2), axis=1)
        return labels
    
    @property
    def labels(self) -> Matrix:
        return self.predict(self._X)
    
    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)


class KMedoidsClustering(Estimator, Unsupervised):
    
    """
    K-Medoids is a clustering algorithm similar to K-Means, but it uses actual 
    data points as cluster centers (medoids) instead of centroids. It minimizes 
    the sum of dissimilarities between points labeled to be in a cluster and a 
    point designated as the medoid of that cluster. During each iteration, it 
    reassigns points to the closest medoid and updates medoids based on the 
    current cluster assignments. K-Medoids is more robust to noise and outliers 
    compared to K-Means.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters to estimate
    `max_iter` : Maximum iteration for PAM (Partitioning Around Medoids) algorithm
    `random_state` : Seed for randomized initialization of medoids
    
    """
    
    def __init__(self, 
                 n_clusters: int, 
                 max_iter: int = 300, 
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids = None
        self.verbose = verbose
        self._X = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'KMedoidsClustering': 
        self._X = X
        m, _ = X.shape
        np.random.seed(self.random_state)
        initial_indices = np.random.choice(m, self.n_clusters, replace=False)
        self.medoids = X[initial_indices]
        
        for i in range(self.max_iter):
            labels = self._closest_medoid(X, self.medoids)
            new_medoids = Matrix([X[labels == n_clusters].mean(axis=0) 
                                  for n_clusters in range(self.n_clusters)])

            if self.verbose and i % 10 == 0 and i:
                print(f'[K-Medoids] iteration: {i}/{self.max_iter}', end='')
                print(f' with medoids-norm {np.linalg.norm(self.medoids)}')
            
            if np.all(new_medoids == self.medoids): 
                if self.verbose:
                    print(f'[K-Medoids] Early-convergence at iteration')
                    print(f'{i}/{self.max_iter}')
                break
            
            self.medoids = new_medoids
        
        self._fitted = True
        return self

    def _closest_medoid(self, X: Matrix, medoids: Matrix) -> Matrix:
        distances = np.zeros((X.shape[0], self.n_clusters))
        for idx, medoid in enumerate(medoids):
            distances[:, idx] = np.linalg.norm(X - medoid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    @property
    def labels(self) -> Vector:
        return self._closest_medoid(self._X, self.medoids)
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        return self._closest_medoid(X, self.medoids)
    
    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)


class MiniBatchKMeansClustering(Estimator, Unsupervised):
    
    """
    Mini-Batch K-Means is an efficient variation of the traditional K-Means 
    clustering algorithm, designed to handle large datasets more effectively. 
    It operates by randomly selecting small subsets of the dataset (mini-batches) 
    and using these subsets, rather than the entire dataset, to update the 
    cluster centroids in each iteration. This approach significantly reduces 
    the computational cost and memory requirements, making it well-suited 
    for big data applications.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters to estimate
    `batch_size` : Size of a single mini-batch
    `max_iter` : Maximum amount of iteration
    
    """
    
    def __init__(self, 
                 n_clusters: int = None, 
                 batch_size: int = 100, 
                 max_iter: int = 100):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.centroids = None
        self._X = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'MiniBatchKMeansClustering':
        m, _ = X.shape
        self._X = X
        
        rand_idx = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = X[rand_idx]

        for _ in range(self.max_iter):
            batch_idx = np.random.choice(m, self.batch_size, replace=False)
            batch = X[batch_idx]
            
            distances = np.linalg.norm(batch[:, np.newaxis] - self.centroids, axis=2)
            closest_cluster_idx = np.argmin(distances, axis=1)
            
            for i in range(self.n_clusters):
                cluster_points = batch[closest_cluster_idx == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    @property
    def labels(self) -> Matrix:
        return self.predict(self._X)

    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)


class FuzzyCMeansClustering(Estimator, Unsupervised):
    
    """
    Fuzzy C-Means (FCM) is a clustering algorithm where each data point can 
    belong to multiple clusters with varying degrees of membership. It 
    iteratively updates cluster centroids and membership levels based on the 
    distance of points to centroids, weighted by their membership. The 
    algorithm is suitable for data with overlapping or unclear boundaries 
    between clusters. FKM provides a soft partitioning of the data, allowing 
    more flexible cluster assignments compared to hard clustering methods.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters to estimate
    `max_iter` : Maximum iteration
    `m`: Fuzziness parameter (larger makes labels softer)
    `tol` : Threshold for early convergence
    `random_state` : Seed for randomized initialization of centers
    
    """
    
    def __init__(self, 
                 n_clusters: int, 
                 max_iter: int = 100, 
                 m: float = 2.0, 
                 tol: float = 1e-5,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self._X = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'FuzzyCMeansClustering':
        self._X = X
        m, _ = X.shape

        np.random.seed(self.random_state)
        self.centers = X[np.random.choice(m, self.n_clusters, replace=False)]
        self.membership = np.zeros((m, self.n_clusters))

        for iter in range(self.max_iter):
            for i in range(m):
                for j in range(self.n_clusters):
                    sum_term = self._sum(X, i, j)
                    self.membership[i, j] = 1 / sum_term if sum_term != 0 else 0

            prev_centers = np.copy(self.centers)
            for j in range(self.n_clusters):
                num_ = sum([self.membership[i, j] ** self.m * X[i] for i in range(m)])
                denom_ = sum([self.membership[i, j] ** self.m for i in range(m)])
                self.centers[j] = num_ / denom_ if denom_ else 0
            
            diff = np.linalg.norm(self.centers - prev_centers)
            if self.verbose and iter % 10 == 0 and iter:
                print(f'[FKM] iteration: {iter}/{self.max_iter}', end='')
                print(f' with delta-center-norm: {diff}')

            if diff < self.tol: 
                if self.verbose:
                    print(f'[FKM] Early-convergnece at iteration', end='')
                    print(f' {iter}/{self.max_iter}')
                break
        
        self._fitted = True
        return self
    
    def _sum(self, X: Matrix, i: int, j: int) -> Matrix:
        sum_term = 0
        for k in range(self.n_clusters):
            distance_ratio = np.linalg.norm(X[i] - self.centers[j])
            distance_ratio /= (np.linalg.norm(X[i] - self.centers[k]) + self.tol)
            sum_term += (distance_ratio ** (2 / (self.m - 1)))
        
        return sum_term

    @property
    def labels(self) -> Vector:
        return self.predict(self._X)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        predictions = np.zeros(m)
        for i in range(m):
            distances = [np.linalg.norm(X[i] - center) for center in self.centers]
            predictions[i] = np.argmin(distances)
        
        return predictions

    def score(self, X: Matrix, metric: Evaluator = SilhouetteCoefficient) -> float:
        X_pred = self.predict(X)
        return metric.score(X, X_pred)

