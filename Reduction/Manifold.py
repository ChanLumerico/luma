from typing import Any, Callable, Literal

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree
from scipy.linalg import eigh, svd
import numpy as np

from luma.reduction.linear import PCA
from luma.interface.super import Transformer, Unsupervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.clustering.kmeans import *
from luma.metric.distance import *


__all__ = ['TSNE', 'MDS', 'MetricMDS', 'LandmarkMDS', 'LLE',
           'ModifiedLLE', 'HessianLLE', 'SammonMapping', 'LaplacianEigenmap',
           'Isomap', 'ConformalIsomap', 'LTSA']


class TSNE(Transformer, Unsupervised):
    
    """
    t-SNE, which stands for t-distributed Stochastic Neighbor Embedding, 
    is a machine learning technique commonly used for data visualization 
    and dimensionality reduction.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of lower spcae \n
    ``max_iter`` : Number of iteration \n
    ``learning_rate`` : Updating factor of embedding optimization \n
    ``perplexity`` : Perplexity parameter of Gaussian kernel \n
    ``verbose`` : Provided details of each iteration when set ``True``
    
    """
    
    def __init__(self, 
                 n_components: int = None, 
                 max_iter: int = 1000, 
                 learning_rate: int = 300, 
                 perplexity: int = 30,
                 verbose: bool = False) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.perplexity = perplexity
        self.verbose = verbose
        self._fitted = False
        
    def fit(self, X: np.ndarray) -> 'TSNE':
        size = X.shape[0]
        P = self._P_joint_probabilities(X)
        y = np.random.normal(0.0, 1e-4, (size, self.n_components))
        Y = [y, y]

        for i in range(1, self.max_iter + 1):
            Q = self._Q_joint_probabilities(Y[-1])
            gradient = self._compute_gradient(P, Q, Y[-1])
            y = Y[-1] - self.learning_rate * gradient
            y += self._momentum(i) * (Y[-1] - Y[-2])
            Y.append(y)
            
            if i % 10 == 0: Q = np.maximum(Q, 1e-12)
            if self.verbose and i % 50 == 0: 
                print(f'[t-SNE] main iteration: {i}/{self.max_iter}', end='')
                print(f' - gradient-norm: {np.linalg.norm(gradient)}')
            
        self.y = y
        self._fitted = True
        return self
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return self.y
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()
    
    def _P_conditional(self, distances: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        exponent = np.exp(-distances / (2 * np.square(sigmas.reshape((-1,1)))))
        np.fill_diagonal(exponent, 0.0)
        exponent += 1e-8
        
        return exponent / exponent.sum(axis=1).reshape([-1,1])
    
    def _P_joint_probabilities(self, X: np.ndarray) -> np.ndarray:
        size = X.shape[0]
        distances = self._compute_pairwise_dist(X)
        sigmas = self._find_sigma_vector(distances)
        conditional_prob = self._P_conditional(distances, sigmas)
        
        return (conditional_prob + conditional_prob.T) / (2.0 * size)
    
    def _Q_joint_probabilities(self, y: np.ndarray) -> np.ndarray:
        distances = self._compute_pairwise_dist(y)
        nominator = 1 / (1 + distances)
        np.fill_diagonal(nominator, 0.0)
        
        return nominator / np.sum(np.sum(nominator))
    
    def _find_sigma_vector(self, distances: np.ndarray) -> np.ndarray:
        N = distances.shape[0]
        opt_sigmas = np.zeros(N)
        
        for i in range(N):
            func = lambda sig: self._compute_perplexity(
                self._P_conditional(distances[i:i + 1, :], np.array([sig])))
            
            opt_sigmas[i] = self._binary_search(func)
            if self.verbose and i % 50 == 0: 
                print(f'[t-SNE] sigma iteration: {i}/{N}', end='')
                print(f' - sigma-norm: {np.linalg.norm(opt_sigmas[i])}')
            
        if self.verbose: print('-' * 70)
        return opt_sigmas
    
    def _binary_search(self, func: Callable, 
                       tol: float = 1e-10, max_iter: int = 1000, 
                       low: float = 1e-10, high: float = 1e3) -> np.ndarray:
        for _ in range(max_iter):
            guess = (high + low) / 2.0
            val = func(guess)
            if val > self.perplexity: high = guess
            else: low = guess
            if np.abs(val - self.perplexity) <= tol: 
                return guess
        return guess
    
    def _compute_pairwise_dist(self, X: np.ndarray) -> np.ndarray:
        return np.sum((X[None, :] - X[:, None]) ** 2, axis=2)
    
    def _compute_gradient(self, P: np.ndarray, Q: np.ndarray, y: np.ndarray) -> np.ndarray:
        pq_diff = P - Q
        y_diff = np.expand_dims(y, axis=1) - np.expand_dims(y, axis=0)
        distances = self._compute_pairwise_dist(y)
        aux = 1 / (1 + distances)
        
        return 4 * (np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux, 2)).sum(axis=1)
    
    def _compute_perplexity(self, cond_matrix: np.ndarray) -> np.ndarray:
        entropy = -np.sum(cond_matrix * np.log2(cond_matrix), axis=1)
        return 2 ** entropy
    
    def _momentum(self, iteration: int):
        return 0.5 if iteration < 250 else 0.8
    
    def set_params(self,
                   n_components: int = None,
                   perplexity: float = None,
                   learning_rate: float = None,
                   max_iter: int = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if perplexity is not None: self.perplexity = float(perplexity)
        if learning_rate is not None: self.learning_rate = float(learning_rate)
        if max_iter is not None: self.max_iter = int(max_iter)


class MDS(Transformer, Unsupervised):
    
    """
    Multidimensional Scaling (MDS) is a data analysis technique that visualizes 
    high-dimensional data in a lower-dimensional space while preserving the 
    relationships between data points. It uses a distance matrix to find a 
    lower-dimensional representation, making it easier to understand and 
    explore data patterns.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space
    
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'MDS':
        D = self._compute_dissimilarity(X)
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H.dot(D).dot(H)
        
        eigvals, eigvecs = np.linalg.eigh(B)
        sorted_indices = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[sorted_indices]
        self.eigvecs = eigvecs[:, sorted_indices]
        
        self.eigvals = self.eigvals[:self.n_components]
        self.eigvecs = self.eigvecs[:, :self.n_components]
        self._fitted = True
        self.stress = self._compute_stress(D)
        return self
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        sqrt_eigvals = np.diag(np.sqrt(np.maximum(self.eigvals, 0)))
        return self.eigvecs.dot(sqrt_eigvals)
    
    def fit_transform(self, X) -> Any:
        self.fit(X)
        return self.transform()
    
    def _compute_dissimilarity(self, X: np.ndarray) -> np.ndarray:
        return squareform(pdist(X, metric='euclidean'))

    def _compute_stress(self, D: np.ndarray) -> float:
        Z = self.transform()
        stress = np.sum((D - self._compute_dissimilarity(Z)) ** 2)
        stress /= np.sum(D ** 2)
        stress = np.sqrt(stress)
        return stress
    
    def set_params(self, n_components: int = None) -> None:
        if n_components is not None: self.n_components = int(n_components)


class MetricMDS(Transformer, Unsupervised):
    
    """
    Metric Multidimensional Scaling (MDS) is a data analysis method that focuses 
    on accurately preserving the original pairwise distances or dissimilarities
    between objects in a high-dimensional space when projecting them onto a 
    lower-dimensional space. 
    
    Parameter
    ---------
    ``n_components`` : Dimensionality of low-space \n
    ``metric`` : Distance metric 
    (e.g. `euclidean`, `manhattan`, `chebyshev`, `minkowski`, \n
    `cos_similarity`, `correation`, `mahalanobis`) \n
    ``p`` : Power factor for `minkowski`
    
    """
    
    def __init__(self,
                 n_components: int = None,
                 p: float | int = None,
                 metric: Literal['euclidean', 'manhattan', 'chebyshev',
                                 'minkowski', 'cos_similarity', 'correlation',
                                 'mahalanobis'] = 'euclidean') -> None:
        self.n_components = n_components
        self.metric = metric
        self.p = p
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'MetricMDS':
        D = self._compute_dissimilarity(X)
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H.dot(D).dot(H)
        
        eigvals, eigvecs = np.linalg.eigh(B)
        sorted_indices = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[sorted_indices]
        self.eigvecs = eigvecs[:, sorted_indices]
        
        self.eigvals = self.eigvals[:self.n_components]
        self.eigvecs = self.eigvecs[:, :self.n_components]
        self._fitted = True
        self.stress = self._compute_stress(D)
        return self
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        sqrt_eigvals = np.diag(np.sqrt(np.maximum(self.eigvals, 0)))
        return self.eigvecs.dot(sqrt_eigvals)
    
    def fit_transform(self, X) -> Any:
        self.fit(X)
        return self.transform()
    
    def _compute_dissimilarity(self, X: np.ndarray) -> np.ndarray:
        m, _ = X.shape
        cov = np.cov(X, rowvar=False)
        D = np.ones((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                join = (X[i], X[j])
                if self.metric == 'euclidean': d = Euclidean.distance(*join)
                elif self.metric == 'manhattan': d = Manhattan.distance(*join)
                elif self.metric == 'chebyshev': d = Chebyshev.distance(*join)
                elif self.metric == 'minkowski': d = Minkowski.distance(*join, self.p)
                elif self.metric == 'cos_similarity': d = CosineSimilarity.distance(*join)
                elif self.metric == 'correlation': d = Correlation.distance(*join)
                elif self.metric == 'mahalanobis': d = Mahalanobis.distance(*join, cov)
                else: raise ValueError(f'[mMDS] Invalid distance metric: {self.metric}')
                D[i, j] = D[j, i] = d
        
        return D

    def _compute_stress(self, D: np.ndarray) -> float:
        Z = self.transform()
        stress = np.sum((D - self._compute_dissimilarity(Z)) ** 2)
        stress /= np.sum(D ** 2)
        stress = np.sqrt(stress)
        return stress
    
    def set_params(self, 
                   n_components: int = None,
                   p: float | int = None,
                   metric: Literal = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if p is not None: self.p = float(p)
        if metric is not None: self.metric = str(metric)


class LandmarkMDS(Transformer, Unsupervised):
    
    """
    Landmark Multi-Dimensional Scaling (LMDS) is a dimensionality reduction method 
    that selects a subset of "landmark" points to efficiently approximate the 
    lower-dimensional representation of a dataset while preserving pairwise distances. 
    It simplifies the computation by focusing on distances between data points and 
    landmarks, making it suitable for large datasets.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space \n
    ``n_landmarks`` : Number of landmarks \n
    ``method`` : Algorithm for landmark initialization
    (e.g. `random`, `kmeans`, `kmeans++`, `kmedians`)
    
    """
    
    def __init__(self, 
                 n_components: int = None, 
                 n_landmarks: int = 10,
                 method: Literal['random', 'kmeans', 'kmeans++', 'kmedians'] = 'random',
                 verbose: bool = False) -> None:
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.method = method
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'LandmarkMDS':
        self.landmarks = self._initialize_landmarks(X)
        D = self._compute_dissimilarity(self.landmarks)
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H.dot(D).dot(H)
        
        eigvals, eigvecs = np.linalg.eigh(B)
        sorted_indices = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[sorted_indices]
        self.eigvecs = eigvecs[:, sorted_indices]
        
        self.eigvals = self.eigvals[self.n_components]
        self.eigvecs = self.eigvecs[:, :self.n_components]
        
        D_mean = np.mean(D, axis=0)
        D_xland = np.sum((X[:, np.newaxis, :] - self.landmarks) ** 2, axis=2)
        self.mean_diff = D_mean - D_xland
        self._fitted = True
        return self
    
    def _initialize_landmarks(self, X: np.ndarray) -> np.ndarray:
        m, _ = X.shape
        if self.method == 'random':
            if self.verbose: print(f'[LMDS] Initializing landmarks with random choice')
            landmark = X[np.random.choice(m, self.n_landmarks, replace=False)]
            
        elif self.method == 'kmeans':
            if self.verbose: print(f'[LMDS] Initializing landmarks with K-Means centroids')
            kmeans = KMeansClustering(n_clusters=self.n_landmarks, verbose=self.verbose)
            kmeans.fit(X)
            landmark = kmeans.centroids
        
        elif self.method == 'kmeans++':
            if self.verbose: print(f'[LMDS] Initializing landmarks with K-Means++ centroids')
            kmeanspp = KMeansClusteringPlus(n_clusters=self.n_landmarks, verbose=self.verbose)
            kmeanspp.fit(X)
            landmark = kmeanspp.centroids
        
        elif self.method == 'kmedians':
            if self.verbose: print(f'[LMDS] Initializing landmarks with K-Medians medians')
            kmedians = KMediansClustering(n_clusters=self.n_landmarks, verbose=self.verbose)
            kmedians.fit(X)
            landmark = kmedians.medians
            
        else: raise UnsupportedParameterError(self.method)
        return landmark
    
    def _compute_dissimilarity(self, X: np.ndarray) -> np.ndarray:
        return squareform(pdist(X, metric='euclidean'))
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        L_sharp = self.eigvecs / np.sqrt(self.eigvals)
        return 0.5 * self.mean_diff.dot(L_sharp)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()
    
    def set_params(self,
                   n_components: int = None,
                   n_landmarks: int = None,
                   method: Literal = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if n_landmarks is not None: self.n_landmarks = int(n_landmarks)
        if method is not None: self.method = str(method)


class LLE(Transformer, Unsupervised):
    
    """
    Locally Linear Embedding (LLE) is a dimensionality reduction technique that aims 
    to find a lower-dimensional representation of data while preserving the local 
    relationships between data points. It works by approximating each data point 
    as a linear combination of its nearest neighbors, revealing the underlying 
    structure of the data in a lower-dimensional space. 
    
    Parameters
    ----------
    ``n_neighbors`` : Number of neighbors to be considered 'close' \n
    ``n_components`` : Dimensionality of low-space
    
    """
    
    def __init__(self, 
                 n_components: int = None, 
                 n_neighbors: int = 5, 
                 verbose: bool = False) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'LLE':
        m, _ = X.shape
        distances = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                distances[i, j] = np.linalg.norm(X[i] - X[j])

        W = np.zeros((m, m))
        neighbors = np.argsort(distances, axis=1)[:, 1:self.n_neighbors + 1]
        for i in range(m):
            Xi = X[neighbors[i]] - X[i]
            Z = np.dot(Xi, Xi.T)
            Z += np.pi * 1e-3 * np.identity(self.n_neighbors)
            
            w = np.linalg.solve(Z, np.ones(self.n_neighbors))
            w /= np.sum(w)
            W[i, neighbors[i]] = w
            
            if self.verbose and i % 100 == 0:
                print(f'[LLE] Optimized weight for instance {i}/{m}', end='')
                print(f' - weight-norm: {np.linalg.norm(w)}')
            
        M = np.identity(m) - W
        self.eigvals, self.eigvecs = np.linalg.eig(np.dot(M.T, M))
        self._fitted = True
        return self
        
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        indices = np.argsort(self.eigvals)[1:self.n_components + 1]
        return self.eigvecs[:, indices]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()

    def set_params(self, 
                   n_neighbors: int = None, 
                   n_components: int = None) -> None:
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)
        if n_components is not None: self.n_components = int(n_components)


class ModifiedLLE(Transformer, Unsupervised):
    
    """
    Modified Locally Linear Embedding (MLLE) is a dimensionality reduction technique 
    that extends the LLE method by introducing regularization for improved stability. 
    MLLE aims to find a lower-dimensional representation of data while preserving the 
    local relationships between data points. It works by approximating each data 
    point as a linear combination of its nearest neighbors, revealing the underlying 
    structure of the data in a lower-dimensional space. 

    Parameters
    ----------
    ``n_neighbors`` : Number of neighbors to be considered 'close' \n
    ``n_components`` : Dimensionality of the lower-dimensional space \n
    ``regularization`` : Regularization parameter for stability \n
    
    """

    def __init__(self, 
                 n_neighbors: int = 5, 
                 n_components: int = None, 
                 regularization: float = 1e-3, 
                 verbose: bool = False) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'ModifiedLLE':
        m, _ = X.shape
        distances = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                distances[i, j] = np.linalg.norm(X[i] - X[j])

        W = np.zeros((m, m))
        neighbors = np.argsort(distances, axis=1)[:, 1 : self.n_neighbors + 1]
        for i in range(m):
            Xi = X[neighbors[i]] - X[i]
            Z = np.dot(Xi, Xi.T)
            Z += self.regularization * np.identity(self.n_neighbors)

            w = np.linalg.solve(Z, np.ones(self.n_neighbors))
            w /= np.sum(w)
            W[i, neighbors[i]] = w

            if self.verbose and i % 100 == 0:
                print(f'[ModifiedLLE] Optimized weight for instance {i}/{m}', end='')
                print(f' - weight-norm: {np.linalg.norm(w)}')

        M = np.identity(m) - W
        self.eigvals, self.eigvecs = np.linalg.eig(np.dot(M.T, M))
        self._fitted = True
        return self

    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        indices = np.argsort(self.eigvals)[1 : self.n_components + 1]
        return self.eigvecs[:, indices]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()
    
    def set_params(self, 
                   n_neighbors: int = None, 
                   n_components: int = None,
                   regularization: float = None) -> None:
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)
        if n_components is not None: self.n_components = int(n_components)
        if regularization is not None: self.regularization = float(regularization)


class HessianLLE(Transformer, Unsupervised):
    
    """
    Hessian Locally Linear Embedding (Hessian LLE) is a dimensionality reduction 
    technique that extends Locally Linear Embedding (LLE). It aims to reduce the 
    dimensionality of high-dimensional data while preserving local linear relationships 
    and capturing the curvature of these relationships using the Hessian matrix.
    
    Parameters
    ----------
    ``n_neighbors`` : Number of neighbors to be considered 'close' \n
    ``n_components`` : Dimensionality of the lower-dimensional space \n
    ``regularization`` : Regularization parameter for stability \n
    
    """
    
    def __init__(self, 
                 n_neighbors: int = 5, 
                 n_components: int = None,
                 regularization: float = 1e-5,
                 verbose: bool = False) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.regularization = regularization
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'HessianLLE':
        m, _ = X.shape
        dp = self.n_components * (self.n_components + 1) // 2
        
        if self.n_neighbors <= self.n_components + dp:
            self.n_neighbors = self.n_components * (self.n_components + 3)
            self.n_neighbors /= 2
            print(f"[HessianLLE] n_neighbors set to {self.n_neighbors}",
                  "due to Hessian condition mismatch.")
        
        index_mat = self._construct_graph(X)
        W = np.zeros((dp * m, m))
        Yi = np.empty((self.n_neighbors, self.n_components + dp + 1))
        Yi[:, 0] = 1
        for i in range(m):
            neighbors = index_mat[i]
            Gi = X[neighbors]
            Gi -= Gi.mean(axis=0)
            
            U, _, _ = np.linalg.svd(Gi, full_matrices=0)
            Yi[:, 1:self.n_components + 1] = U[:, :self.n_components]
            j = self.n_components + 1
            for k in range(self.n_components):
                Yi[:, j:self.n_components + j - k] = \
                    U[:, k:k + 1] * U[:, k:self.n_components]
                j += self.n_components - k
            
            Q, R = np.linalg.qr(Yi)
            w = np.array(Q[:, self.n_components + 1:])
            S = w.sum(axis=0)
            S[np.where(np.abs(S) < self.regularization)] = 1.0
            w /= S
            
            xn, yn = np.meshgrid(neighbors, neighbors)
            W[xn, yn] += w.dot(w.T)
            
            if self.verbose and i % 100 == 0:
                print(f'[HessianLLE] Optimized weight for instance {i}/{m}', end='')
                print(f' - weight-norm: {np.linalg.norm(w)}')
        
        _, sig, VT = np.linalg.svd(W, full_matrices=0)
        indices = np.argsort(sig)[1:self.n_components + 1]
        Y = VT[indices, :] * np.sqrt(m)
        
        _, sig, VT = np.linalg.svd(Y, full_matrices=0)
        S = np.matrix(np.diag(sig ** (-1)))
        R = VT.T * S * VT
        self.Y, self.R = Y, R
        self._fitted = True
        return self
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return np.array(self.Y * self.R).T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()
    
    def _construct_graph(self, X: np.ndarray) -> np.ndarray:
        m, _ = X.shape
        kd_tree = KDTree(X.copy())
        index_mat = np.ones((m, self.n_neighbors), dtype=int)
        for i in range(m):
            _, neighbors = kd_tree.query(X[i], k=self.n_neighbors + 1, p=2)
            neighbors = neighbors.tolist()
            neighbors.remove(i)
            index_mat[i] = np.array([neighbors])
        
        return index_mat

    def set_params(self, 
                   n_neighbors: int = None, 
                   n_components: int = None,
                   regularization: float = None) -> None:
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)
        if n_components is not None: self.n_components = int(n_components)
        if regularization is not None: self.regularization = float(regularization)


class SammonMapping(Transformer, Unsupervised):
    
    """
    Sammon mapping is a dimensionality reduction algorithm designed to transform 
    high-dimensional data into a lower-dimensional space. Sammon mapping focuses 
    on preserving the pairwise distances or dissimilarities between data points, 
    making it well-suited for capturing non-linear relationships in the data.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space \n
    ``max_iter`` : Number of iteration \n
    ``max_halves`` : Number of halving of step-size \n
    ``tol`` : Threshold for early-stopping \n
    ``initialize`` : Methodology for embedding-space initialization \n
    
    """
    
    def __init__(self,
                 n_components: int = None,
                 max_iter: int = 100,
                 max_halves: int = 20,
                 tol: float = 1e-5,
                 initialize: Literal['pca'] | None = 'pca',
                 verbose: bool = False) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.max_halves = max_halves
        self.tol = tol
        self.initialize = initialize
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'SammonMapping':
        m, _ = X.shape
        X_dist = cdist(X, X)
        scale = 0.5 / X_dist.sum()
        
        Y = np.random.normal(0, 1, (m, self.n_components))
        if self.initialize == 'pca':
            Y = PCA(n_components=self.n_components).fit_transform(X)
            if self.verbose:
                print('[SammonMap] Finished PCA initialization')
        
        ones = np.ones((m, self.n_components))
        X_dist += np.eye(m)
        X_dist_inv = 1 / X_dist
        self._remove_inf(X_dist_inv, X_dist_inv)
        
        Y_dist = cdist(Y, Y) + np.eye(m)
        Y_dist_inv = 1 / Y_dist
        
        self._fill_diagnoal(X_dist, 1)
        self._fill_diagnoal(Y_dist, 1)
        self._fill_diagnoal(X_dist_inv, 0)
        self._fill_diagnoal(Y_dist_inv, 0)
        
        self._remove_inf(matrix=X_dist_inv, filter=X_dist_inv)
        self._remove_inf(matrix=Y_dist_inv, filter=X_dist_inv)
        self._remove_inf(matrix=X_dist_inv, filter=Y_dist_inv)
        self._remove_inf(matrix=Y_dist_inv, filter=Y_dist_inv)
        
        E = np.sum(((X_dist - Y_dist) ** 2) * X_dist_inv)
        for i in range(self.max_iter):
            delta = Y_dist_inv - X_dist_inv
            delta_one = delta.dot(ones)
            
            gradient = np.dot(delta, Y) - (Y * delta_one)
            hessian = np.dot(Y_dist_inv ** 3, Y ** 2) - delta_one
            hessian -= np.dot(2, Y) * np.dot(Y_dist_inv ** 3, Y)
            hessian += Y ** 2 * np.dot(Y_dist_inv ** 3, ones)
            
            step = -gradient.flatten(order='F') / np.abs(hessian.flatten(order='F'))
            Y_old = Y
            
            for j in range(self.max_halves):
                step_reshape = step.reshape(2, round(len(step) / 2)).T
                y = Y_old + step_reshape
                
                dist = cdist(Y, y) + np.eye(m)
                dist_inv = 1 / dist
                self._remove_inf(matrix=dist_inv, filter=dist_inv)
                
                delta = X_dist - dist
                E_new = np.sum((delta ** 2) * X_dist_inv)
                if E_new < E: break
                else: step = np.dot(0.5, step)
            
            if j == self.max_halves:
                print(f'[SammonMap] Warning: max halves reached.', end='')
                print(' Optimization may not converge!')
            
            if np.abs((E - E_new) / E) < self.tol:
                print(f'[SammonMap] Early convergence at iteration {i}/{self.max_iter}')
                break
            
            if self.verbose and i % 10 == 0:
                print(f'[SammonMap] Iteration: {i}/{self.max_iter}', end='')
                print(f' - relative delta-error: {np.abs((E - E_new) / E)}')
            
            E = E_new
        self.error = E * scale
        self.Y = y
        
        if self.verbose: print(f'[SammonMap] Final error: {self.error}')
        self._fitted = True
        return self
    
    def _fill_diagnoal(self, matrix: np.ndarray, value: float) -> None:
        np.fill_diagonal(matrix, value)
    
    def _remove_inf(self, matrix: np.ndarray, filter: np.ndarray) -> None:
        matrix[np.isinf(filter)] = 0
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return self.Y
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()

    def set_params(self,
                   n_components: int = None,
                   max_iter: int = None,
                   max_halves: int = None,
                   initialize: Literal = None,
                   tol: float = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if max_iter is not None: self.max_iter = int(max_iter)
        if max_halves is not None: self.max_halves = int(max_halves)
        if initialize is not None: self.initialize = str(initialize)
        if tol is not None: self.tol = float(tol)


class LaplacianEigenmap(Transformer, Unsupervised):
    
    """
    Laplacian Eigenmap is a dimensionality reduction technique that 
    transforms high-dimensional data into a lower-dimensional space while 
    preserving local relationships. It constructs a weighted graph based 
    on pairwise similarities between data points, computes the Laplacian 
    matrix from the graph, and performs eigenvalue decomposition to 
    obtain eigenvectors.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of the lower-dimensional space \n
    ``sigma`` : Width parameter for Gaussian kernel in affinity calculation
    
    """

    def __init__(self, n_components: int = None, sigma: float = 1.0) -> None:
        self.n_components = n_components
        self.sigma = sigma
        self.embedding = None
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'LaplacianEigenmap':
        pairwise = cdist(X, X)
        weights = np.exp(-pairwise / (2 * self.sigma ** 2))
        np.fill_diagonal(weights, 0)
        
        laplacian = self._compute_laplacian(weights)
        normalized_laplacian = self._normalize_laplacian(weights, laplacian)
        
        eigvals, eigvecs = np.linalg.eigh(normalized_laplacian)
        sorted_indices = np.argsort(eigvals)
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        self.embedding = eigvecs[:, :self.n_components]
        self._fitted = True
        return self

    def _compute_laplacian(self, W: np.ndarray) -> np.ndarray:
        return np.diag(np.sum(W, axis=1)) - W
    
    def _normalize_laplacian(self, W: np.ndarray, L: np.ndarray) -> np.ndarray:
        D_sqrt_inv = np.linalg.inv(np.sqrt(np.diag(np.sum(W, axis=1))))
        return D_sqrt_inv.dot(L).dot(D_sqrt_inv)
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return self.embedding
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()

    def set_params(self, n_components: int = None, sigma: float = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if sigma is not None: self.sigma = float(sigma)


class Isomap(Transformer, Unsupervised):
    
    """
    Isomap, or Isometric Mapping, is a nonlinear dimensionality reduction technique 
    employed in machine learning. It focuses on preserving the geodesic distances 
    between data points in high-dimensional datasets. By constructing a neighborhood 
    graph and calculating pairwise geodesic distances, Isomap transforms the data 
    into a lower-dimensional space, offering insights into complex, nonlinear 
    relationships within the dataset.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space \n
    ``epsilon`` : Boundary radius of the neighbor hypersphere \n
    ``algorithm`` : Shortest path finding algorithm
    (e.g. `dijkstra`, `floyd`) \n
    ``metric`` : Distance metric
    (e.g. `euclidean`, `cityblock`, `chebyshev`, `cosine`, `correlation`)
    
    """
    
    def __init__(self, 
                 n_components: int = None, 
                 epsilon: float = 5.0,
                 algorithm: Literal['dijkstra', 'floyd'] = 'dijkstra',
                 metric: Literal['euclidean', 'cityblock', 'chebyshev',
                                 'cosine', 'correlation'] = 'euclidean') -> None:
        self.n_components = n_components
        self.epsilon = epsilon
        self.algorithm = algorithm
        self.metric = metric
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'Isomap':
        self.adj = self._make_adjacency(X)
        if np.isinf(self.adj).any(): 
            print(f'[Isomap] Too narrow bin size with epsilon of {self.epsilon}!')
            return
        
        m, _ = self.adj.shape
        H = np.eye(m) - (1 / m) * np.ones((m, m))
        C = -0.5 / m * H.dot(self.adj ** 2).dot(H)
        
        eigvals, eigvecs = eigh(C)
        indices = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[indices], eigvecs[:, indices]
        self.eigvals = eigvals[:self.n_components]
        self.eigvecs = eigvecs[:, :self.n_components]
        self._fitted = True
        return self

    def _make_adjacency(self, X: np.ndarray) -> np.ndarray:
        m, _ = X.shape
        distance = cdist(X, X, metric=self.metric)
        adjacency = np.zeros((m, m)) + np.inf
        bin = distance < self.epsilon
        adjacency[bin] = distance[bin]
        
        if self.algorithm == 'dijkstra':
            path_mat = shortest_path(adjacency, directed=False, method='D')
        elif self.algorithm == 'floyd':
            path_mat = shortest_path(adjacency, directed=False, method='FW')
        else: raise UnsupportedParameterError(self.algorithm)
        return path_mat

    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return self.eigvecs.dot(np.diag(np.sqrt(self.eigvals)))
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()
    
    def set_params(self, 
                   n_components: int = None, 
                   epsilon: float = None,
                   algorithm: Literal = None,
                   metric: Literal = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if epsilon is not None: self.epsilon = float(epsilon)
        if algorithm is not None: self.algorithm = str(algorithm)
        if metric is not None: self.method = str(metric)


class ConformalIsomap(Transformer, Unsupervised):
    
    """
    Conformal Isomap (C-Isomap) is a variation of the Isomap algorithm that 
    prioritizes preserving infinitesimal angles (conformality) by adjusting 
    geodesic distances to reflect scale information. Unlike traditional Isomap, 
    which focuses on preserving full isometry, C-Isomap is well-suited for 
    datasets with varying local scales or non-uniform structures.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space \n
    ``epsilon`` : Boundary radius of the neighbor hypersphere \n
    ``algorithm`` : Shortest path finding algorithm
    (e.g. `dijkstra`, `floyd`)
    
    """
    
    def __init__(self,
                 n_components: int = None,
                 epsilon: float = 5.0,
                 algorithm: Literal['dijkstra', 'floyd'] = 'dijkstra') -> None:
        self.n_components = n_components
        self.epsilon = epsilon
        self.algorithm = algorithm
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'ConformalIsomap':
        self.adj = self._make_adjacency(X)
        if np.isinf(self.adj).any():
            print(f'[C-Isomap] Too narrow bin size with epsilon of {self.epsilon}!')
            return
        
        m, _ = self.adj.shape
        H = np.eye(m) - (1 / m) * np.ones((m, m))
        C = -0.5 / m * H.dot(self.adj ** 2).dot(H)
        
        eigvals, eigvecs = eigh(C)
        indices = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[indices], eigvecs[:, indices]
        self.eigvals = eigvals[:self.n_components]
        self.eigvecs = eigvecs[:, :self.n_components]
        self._fitted = True
        return self
    
    def _make_adjacency(self, X: np.ndarray) -> np.ndarray:
        m, _ = X.shape
        distance = cdist(X, X, metric='euclidean')
        adjacency = np.zeros((m, m)) + np.inf
        bin = distance < self.epsilon
        adjacency[bin] = distance[bin]
        
        masked = np.ma.masked_array(adjacency, mask=np.isinf(adjacency))
        masked_mean = np.ma.mean(masked, axis=0)
        for i in range(m):
            for j in range(m):
                adjacency[i, j] /= np.sqrt(masked_mean[i] * masked_mean[j])

        if self.algorithm == 'dijkstra':
            path_mat = shortest_path(adjacency, directed=False, method='D')
        elif self.algorithm == 'floyd':
            path_mat = shortest_path(adjacency, directed=False, method='FW')
        else: raise UnsupportedParameterError(self.algorithm)
        return path_mat
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return self.eigvecs.dot(np.diag(np.sqrt(self.eigvals)))
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()

    def set_params(self,
                   n_components: int = None,
                   epsilon: float = None,
                   algorithm: Literal = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if epsilon is not None: self.epsilon = float(epsilon)
        if algorithm is not None: self.algorithm = str(algorithm)


class LTSA(Transformer, Unsupervised):
    
    """
    Local Tangent Space Alignment (LTSA) is a dimensionality reduction technique 
    that aims to preserve the local geometry of data points in a manifold. 
    It works by approximating the tangent space at each data point and aligning 
    these local tangent spaces. This helps to capture the intrinsic structure 
    of the data, particularly in cases where the global geometry might be 
    complex or non-linear. 
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space \n
    ``n_neighbors`` : Number of neighbors
    
    """
    
    def __init__(self,
                 n_components: int = None,
                 n_neighbors: int = 10) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'LTSA':
        m, n = X.shape
        self.neighbors, self.neighbor_indices = self._compute_neighborhood(X)
        M = np.zeros((m, m))
        use_svd = self.n_neighbors > n
        
        for i in range(m):
            Xi = self.neighbors[i]
            Xi -= np.mean(self.neighbors[i], axis=0)
            if use_svd:
                v = svd(Xi, full_matrices=True)[0]
            else:
                Ci = Xi.dot(Xi.T)
                v = eigh(Ci)[1][:, ::-1]
            
            Gi = np.zeros((self.n_neighbors, self.n_components + 1))
            Gi[:, 1:] = v[:, :self.n_components]
            Gi[:, 0] = 1 / np.sqrt(self.n_neighbors)
            
            neighbors_x, neighbors_y = np.meshgrid(self.neighbor_indices[i], 
                                                   self.neighbor_indices[i])
            M[neighbors_x, neighbors_y] -= Gi.dot(Gi.T)
            M[self.neighbor_indices[i], self.neighbor_indices[i]] += 1
        
        self._M = M
        self._fitted = True
        return self
    
    def _compute_neighborhood(self, X: np.ndarray) -> np.ndarray:
        distances = cdist(X, X, metric='euclidean')
        indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        return X[indices], indices
    
    def _find_null_space(self, M: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = eigh(M, subset_by_index=(1, self.n_components), 
                                overwrite_a=True)
        indices = np.argsort(np.abs(eigvals))
        return eigvecs[:, indices]
    
    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return self._find_null_space(self._M)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()

    def set_params(self,
                   n_components: int = None,
                   n_neighbors: int = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if n_neighbors is not None: self.n_neighbors = int(n_neighbors)

