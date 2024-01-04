from typing import Any, Literal
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np

from luma.clustering.kmeans import KMeansClusteringPlus
from luma.interface.util import Matrix
from luma.interface.super import Estimator, Evaluator, Unsupervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.clustering import SilhouetteCoefficient


__all__ = ['SpectralClustering', 'NormalizedSpectralClustering']


class SpectralClustering(Estimator, Unsupervised):
    
    """
    Spectral clustering transforms data into a new space using the 
    eigenvectors of its graph Laplacian, revealing cluster structure. 
    It constructs a similarity graph from the data, capturing complex 
    relationships between points. The algorithm then applies 
    dimensionality reduction followed by a traditional clustering 
    method, like K-means, in this transformed space. This approach 
    is effective for identifying clusters with irregular shapes 
    and non-linearly separable data.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters to estimate
    `gamma` : Scaling factor for Gaussian kernel
    
    Examples
    --------
    >>> sp = SpectralClustering()
    >>> sp.fit(X, y)
    >>> lables = sp.labels # Get assigned labels
    
    """
    
    def __init__(self, 
                 n_clusters: int, 
                 gamma: float = 1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'SpectralClustering':
        self._X = X
        W = self._similarity_matrix(X)
        L = self._laplacian(W)
        
        _, eigenvecs = eigh(L)
        V = eigenvecs[:, :self.n_clusters]
        kmeans = KMeansClusteringPlus(n_clusters=self.n_clusters)
        self.kmeans = kmeans.fit(V)
        
        self._fitted = True
        return self

    def _similarity_matrix(self, X: Matrix) -> Matrix:
        sq_dists = pdist(X, 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        return np.exp(-self.gamma * mat_sq_dists)

    def _laplacian(self, W: Matrix) -> Matrix:
        D = np.diag(np.sum(W, axis=1))
        return D - W

    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return self.kmeans.labels
    
    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")
    
    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.compute(self._X, self.labels)
    
    def set_params(self, 
                   n_clusters: int = None,
                   gamma: float = None) -> None:
        if n_clusters is not None: self.n_clusters = int(n_clusters)
        if gamma is not None: self.gamma = float(gamma)


class NormalizedSpectralClustering(Estimator, Unsupervised):
    
    """
    Normalized spectral clustering is a technique that begins by constructing 
    a similarity matrix from the data, often using a Gaussian kernel. The graph 
    Laplacian matrix, derived from this similarity matrix, is then normalized 
    to enhance the distinctiveness of clusters. Eigenvalue decomposition is 
    applied to this normalized Laplacian, and the resulting eigenvectors are 
    used to transform the data into a new space. Finally, a clustering algorithm, 
    such as k-means, is applied in this transformed space, effectively revealing 
    clusters that are not easily distinguishable in the original space.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters to estimate
    `gamma` : Scaling factor for Gaussian kernel
    `strategy` : Normalization strategy (e.g. `symmetric`, `random-walk`)
    
    """
    
    def __init__(self, 
                 n_clusters: int,
                 gamma: float = 1.0,
                 strategy: Literal['symmetric', 
                                   'random-walk'] = 'symmetric') -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.strategy = strategy
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'NormalizedSpectralClustering':
        self._X = X
        W = self._similarity_matrix(X)
        L_norm = self._normalize_laplacian(W)
        
        _, eigvecs = eigh(L_norm)
        V = eigvecs[:, :self.n_clusters]
        kmeans = KMeansClusteringPlus(n_clusters=self.n_clusters)
        self.kmeans = kmeans.fit(V)
        
        self._fitted = True
        return self
    
    def _similarity_matrix(self, X: Matrix) -> Matrix:
        sq_dists = pdist(X, 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        return np.exp(-self.gamma * mat_sq_dists)
    
    def _normalize_laplacian(self, W: Matrix) -> Matrix:
        D = np.diag(np.sum(W, axis=1))
        D_inv = np.diag(1 / np.diag(D))
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        
        if self.strategy == 'symmetric':
            L_norm = np.eye(*D.shape) - D_inv_sqrt.dot(W).dot(D_inv_sqrt)
        elif self.strategy == 'random-walk':
            L_norm = np.eye(*D.shape) - D_inv.dot(W)
        else:
            raise UnsupportedParameterError(self.strategy)

        return L_norm
    
    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return self.kmeans.labels

    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")
    
    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.compute(self._X, self.labels)
    
    def set_params(self, 
                   n_clusters: int = None,
                   gamma: float = None,
                   strategy: Literal = None) -> None:
        if n_clusters is not None: self.n_clusters = int(n_clusters)
        if gamma is not None: self.gamma = float(gamma)
        if strategy is not None: self.strategy = str(strategy)

