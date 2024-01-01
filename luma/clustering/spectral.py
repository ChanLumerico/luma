from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np

from luma.clustering.kmeans import KMeansClusteringPlus
from luma.interface.util import Matrix
from luma.interface.super import Estimator, Evaluator, Unsupervised
from luma.interface.exception import NotFittedError
from luma.metric.clustering import SilhouetteCoefficient


__all__ = ['SpectralClustering']


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
        
        _, eigenvects = eigh(L)
        V = eigenvects[:, :self.n_clusters]
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

