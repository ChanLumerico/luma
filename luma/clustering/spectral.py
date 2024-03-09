from typing import Literal, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np

from luma.clustering.kmeans import KMeansClusteringPlus
from luma.clustering.hierarchy import AgglomerativeClustering, DivisiveClustering
from luma.interface.util import Matrix
from luma.core.super import Estimator, Evaluator, Unsupervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.clustering import SilhouetteCoefficient


__all__ = (
    'SpectralClustering', 
    'NormalizedSpectralClustering',
    'HierarchicalSpectralClustering',
    'AdaptiveSpectralClustering'
)


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
        return metric.score(self._X, self.labels)


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
        return metric.score(self._X, self.labels)


class HierarchicalSpectralClustering(Estimator, Unsupervised):
    
    """
    Hierarchical spectral clustering starts by applying spectral clustering 
    to transform data into a space where clusters are more linearly separable. 
    This involves constructing a similarity matrix, computing the Laplacian, 
    and performing eigenvalue decomposition. The transformed data is then 
    clustered using hierarchical methods, either agglomerative (building 
    clusters by merging smaller ones) or divisive (splitting larger clusters 
    into smaller ones). This approach reveals multi-level structures in the 
    data, uncovering both broad and fine-grained clustering patterns.
    
    Parameters
    ----------
    `n_clusters` : Number of clusters to estimate
    `method` : Method for hierarchical clustering
    (e.g. `agglomerative`, `divisive`)
    `linkage` : Linkage method for `agglomerative` clustering
    (e.g. `single`, `complete`, `average`, `ward`)
    `gamma` : Scaling factor for Gaussian kernel
    
    """
    
    def __init__(self, 
                 n_clusters: int, 
                 method: Literal['agglomerative', 'divisive'],
                 linkage: Literal['single', 'complete', 'average'] = 'single',
                 gamma: float = 1.0,):
        self.n_clusters = n_clusters
        self.method = method
        self.linkage = linkage
        self.gamma = gamma
        self._hierarchy_model = None
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'HierarchicalSpectralClustering':
        self._X = X
        W = self._similarity_matrix(X)
        L = self._laplacian(W)
        
        _, eigenvecs = eigh(L)
        V = eigenvecs[:, :self.n_clusters]
        
        if self.method == 'agglomerative':
            self._hierarchy_model = AgglomerativeClustering()
        elif self.method == 'divisive':
            self._hierarchy_model = DivisiveClustering()
        else:
            raise UnsupportedParameterError(self.method)
        
        self._hierarchy_model.n_clusters = self.n_clusters
        if hasattr(self._hierarchy_model, 'linkage'):
            self._hierarchy_model.linkage = self.linkage
        
        self._hierarchy_model.fit(V)
        
        self._fitted = True
        return self

    def _similarity_matrix(self, X: Matrix) -> Matrix:
        sq_dists = pdist(X, 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        return np.exp(-self.gamma * mat_sq_dists)

    def _laplacian(self, W: Matrix) -> Matrix:
        D = np.diag(np.sum(W, axis=1))
        return D - W
    
    def plot_dendrogram(self,
                        ax: Optional[plt.Axes],
                        hide_indices: bool = True, 
                        show: bool = False) -> plt.Axes:
        if not self.method == 'agglomerative':
            raise UnsupportedParameterError(self.method)
        
        return self._hierarchy_model.plot_dendrogram(ax, hide_indices, show)

    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return self._hierarchy_model.labels
    
    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")
    
    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)


class AdaptiveSpectralClustering(Estimator, Unsupervised):
    
    """
    Adaptive Spectral Clustering automatically determines the number of clusters 
    and scale parameters directly from the data, reducing the need for manual 
    tuning. It constructs a similarity matrix, typically using a Gaussian kernel 
    with an adaptively chosen scale. Eigenvalue decomposition on the Laplacian of 
    this matrix reveals the data's structure, and the eigen-gap heuristic is 
    employed to select the optimal number of clusters.
    
    Parameters
    ----------
    `gamma` : Scaling factor for Gaussian kernel
    `max_clusters` : Upper-bound of the number of clusters to estimate
    
    """
    
    def __init__(self, 
                 gamma: float = 1.0,
                 max_clusters: int = 10) -> None:
        self.gamma = gamma
        self.max_clusters = max_clusters
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'AdaptiveSpectralClustering':
        self._X = X
        W = self._similarity_matrix(X)
        L = self._laplacian(W)
        
        eigvals, eigvecs = eigh(L)
        self.n_clusters = self._optimal_clusters(eigvals)
        
        V = eigvecs[:, :self.n_clusters]
        kmeans = KMeansClusteringPlus(n_clusters=self.n_clusters)
        self.kmeans = kmeans.fit(V)
        
        self.W = W
        self.L = L
        self._fitted = True
        return self

    def _similarity_matrix(self, X: Matrix) -> Matrix:
        sq_dists = pdist(X, metric='sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        return np.exp(-self.gamma * mat_sq_dists)
    
    def _laplacian(self, W: Matrix) -> Matrix:
        D = np.diag(np.sum(W, axis=0))
        return D - W
    
    def _optimal_clusters(self, eigvals: Matrix) -> Matrix:
        eigval_diff = np.diff(eigvals)
        return np.argmax(eigval_diff[:self.max_clusters]) + 1
    
    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return self.kmeans.labels
    
    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")

    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.score(self._X, self.labels)

