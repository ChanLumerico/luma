from typing import Literal
import numpy as np

from luma.interface.util import Matrix, Vector, Constant
from luma.interface.super import Estimator, Evaluator, Unsupervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.clustering import SilhouetteCoefficient


__all__ = (
    'AffinityPropagation'
)


class AffinityPropagation(Estimator, Unsupervised):
    
    """
    Affinity Propagation is a clustering algorithm that identifies exemplars 
    among data points, effectively determining cluster centers. It works by 
    exchanging messages between pairs of data points until a set of exemplars 
    and corresponding clusters emerges. The algorithm doesn't require the 
    number of clusters to be specified in advance. Affinity Propagation is 
    particularly useful for problems where the optimal number of clusters 
    is unknown or hard to estimate.
    
    Parameters
    ----------
    `max_iter` : Maximum iterations for message exchange
    `damping` : Balancing factor for ensuring numerical stability and convergence
    `tol` : Early-stopping threshold
    `preference` : Parameter which determines the selectivity of choosing exemplars
    
    * Self-Similarity Setting:
        The preference value represents the "self-similarity" of each data point. 
        It is the value on the diagonal of the similarity matrix, which influences 
        how likely a data point is to be chosen as an exemplar (a cluster center).
    
    * Influencing Cluster Count:
        A higher preference value suggests a greater likelihood for a data point to 
        become an exemplar, potentially leading to more clusters. Conversely, a lower 
        preference value tends to produce fewer clusters, as fewer points are strong 
        candidates for being exemplars.
    
    * Default Setting:
        If not explicitly set, the preference is often chosen automatically based on 
        the input data. Common practices include setting it to the median or mean of 
        the similarity values.
    
    """
    
    def __init__(self, 
                 max_iter: int = 100,
                 damping: float = 0.7,
                 preference: Constant | Vector | Literal['median', 'min'] = 'median',
                 tol: float = None,
                 verbose: bool = False) -> None:
        self.max_iter = max_iter
        self.damping = damping
        self.preference = preference
        self.tol = tol
        self.verbose = verbose
        self._X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'AffinityPropagation':
        self._X = X
        m, _ = X.shape
        
        S = self._compute_similarity(X)
        S = self._assign_preference(S)
        S += 1e-12 * np.random.normal(size=(m, m)) * (np.max(S) - np.min(S))
        
        R = np.zeros_like(S)
        A = np.zeros_like(S)
        
        for iter in range(self.max_iter):
            E_old = R + A         
            R = self._compute_responsibility(S, R, A)
            A = self._compute_availability(R, A)         
            E_new = R + A

            if self.tol is not None and np.allclose(E_old, E_new, atol=self.tol):
                if not self.verbose: break
                print(f"[AP] Early convergence at iteration {iter}/{self.max_iter}")
                break
            
            if self.verbose and iter % 10 == 0 and iter:
                print(f"[AP] Finished iteration {iter}/{self.max_iter}",
                      f"with delta-E: {np.linalg.norm(E_new - E_old)}")
        
        E = R + A
        self._clusters = np.argmax(E, axis=1)
        self.exemplars = np.unique(self._clusters)
        self.centers = X[self.exemplars]
        
        replace = dict(zip(self.exemplars, range(len(self.exemplars))))
        new_ = np.arange(0, max(self._clusters) + 1)
        new_[list(replace.keys())] = list(replace.values())
        
        self._clusters = new_[self._clusters]
        self._fitted = True
        return self
    
    def _compute_similarity(self, X: Matrix) -> Matrix:
        first_ = np.sum(X ** 2, axis=-1).reshape(-1, 1)
        second_ = np.sum(X ** 2, axis=-1)
        third_ = -2 * np.dot(X, X.T)
        
        return -1 * (first_ + second_ + third_)

    def _assign_preference(self, S: Matrix) -> Matrix:
        indices = np.where(~np.eye(S.shape[0], dtype=bool))
        
        if self.preference == 'median': pref = np.median(S[indices])
        elif self.preference == 'min': pref = np.min(S[indices])
        
        elif isinstance(self.preference, (np.ndarray, float, int)): 
            pref = self.preference
        else:
            raise UnsupportedParameterError(self.preference)

        np.fill_diagonal(S, pref)
        return S
    
    def _compute_responsibility(self, S: Matrix, R: Matrix, A: Matrix) -> Matrix:
        max_ = A + S
        np.fill_diagonal(max_, -np.inf)
        
        row_indices = np.arange(max_.shape[0])
        max_indices = np.argmax(max_, axis=1)
        
        row_max = max_[row_indices, max_indices]
        max_[row_indices, max_indices] = -np.inf
        
        row_max_ = max_[row_indices, np.argmax(max_, axis=1)]
        max_AS = np.zeros_like(S) + row_max.reshape(-1, 1)
        max_AS[row_indices, max_indices] = row_max_
        
        return (1 - self.damping) * (S - max_AS) + self.damping * R
    
    def _compute_availability(self, R: Matrix, A: Matrix) -> Matrix:
        R = R.copy()
        diag = np.diag(R).copy()
        
        np.fill_diagonal(R, 0)
        R = np.where(R < 0, 0, R)
        
        sum_vectors = np.sum(R, axis=0)
        A_new = np.minimum(0, diag + sum_vectors - R)
        np.fill_diagonal(A_new, np.sum(R, axis=0))
        
        return (1 - self.damping) * A_new + self.damping * A

    @property
    def labels(self) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return self._clusters
    
    def predict(self) -> None:
        raise Warning(f"{type(self).__name__} does not support prediction!")
    
    def score(self, metric: Evaluator = SilhouetteCoefficient) -> float:
        return metric.compute(self._X, self.labels)
    
    def set_params(self, 
                   max_iter: int = None,
                   damping: float = None,
                   preference: Constant | Vector | str = None,
                   tol: float = None) -> None:
        if max_iter is not None: self.max_iter = int(max_iter)
        if damping is not None: self.damping = float(damping)
        if tol is not None: self.tol = float(tol)
        if preference is not None:
            if isinstance(preference, (int, float)):
                self.preference = float(preference)
            elif isinstance(preference, np.ndarray):
                self.preference = preference
            else: 
                self.preference = str(preference)

