from typing import *
from scipy.linalg import eigh
import numpy as np

from luma.interface.super import Transformer, Unsupervised
from luma.interface.exception import NotFittedError


__all__ = ['KernelPCA']
    

class KernelPCA(Transformer, Unsupervised):
    
    """
    Kernel PCA is a dimensionality reduction technique that extends 
    Principal Component Analysis (PCA) to capture complex, nonlinear 
    relationships in data by using a kernel function to transform 
    the data into a higher-dimensional space where nonlinear patterns 
    can be better captured.
    
    Parameters
    ----------
    ``n_components`` : Number of principal components \n
    ``degree`` : Polynomial degree of ``poly`` kernel \n
    ``gamma`` : Shape parameter of ``rbf``, ``sigmoid``, ``laplacian`` \n
    ``coef`` : Additional coefficient of ``poly``, ``sigmoid`` \n
    ``kernel`` : Type of kernel functions
    (e.g. ``linear``, ``poly``, ``rbf``, ``sigmoid``, ``laplacian``)
    
    """
    
    def __init__(self, 
                 n_components: int = None,
                 degree: int = 3,
                 gamma: float = 15.0,
                 coef: float = 1.0,
                 kernel: Literal['linear', 'poly', 'rbf', 
                                 'sigmoid', 'laplacian'] = 'linear') -> None:
        self.n_components = n_components
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
        self.kernel = kernel
        self.X = None
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'KernelPCA':
        self.X = X
        if self.kernel == 'linear': self.kernel_func = self._linear
        elif self.kernel == 'poly': self.kernel_func = self._poly
        elif self.kernel == 'rbf': self.kernel_func = self._rbf
        elif self.kernel == 'sigmoid': self.kernel_func = self._sigmoid
        elif self.kernel == 'laplacian': self.kernel_func = self._laplacian
        else: raise ValueError('[KPCA] Unsupported Kernel!')

        N = X.shape[0]
        self.K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                kernel_value = self.kernel_func(X[i], X[j])
                self.K[i, j] = kernel_value
                self.K[j, i] = kernel_value
        
        one_n = np.ones((N, N)) / N
        self.K = self.K - one_n.dot(self.K) - self.K.dot(one_n)
        self.K += one_n.dot(self.K).dot(one_n)
        self.eigvals, self.eigvecs = eigh(self.K)
        self.eigvals, self.eigvecs = self.eigvals[::-1], self.eigvecs[:, ::-1]
        self._fitted =  True
        return self

    def transform(self) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return np.column_stack([self.eigvecs[:, i] for i in range(self.n_components)])
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()
    
    def _linear(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.dot(x, y)
    
    def _poly(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (np.dot(x, y) + self.coef) ** self.degree
    
    def _rbf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.gamma is None: self.gamma = 1 / self.X.shape[1]
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)
    
    def _sigmoid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.tanh(self.gamma * np.dot(x, y) + self.coef)
    
    def _laplacian(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-self.gamma * np.linalg.norm(x - y))
    
    def set_params(self,
                   n_components: int = None,
                   degree: int = None,
                   gamma: float = None,
                   coef: float = None,
                   kernel: Literal = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if degree is not None: self.degree = int(degree)
        if gamma is not None: self.gamma = float(gamma)
        if coef is not None: self.coef = float(coef)
        if kernel is not None: self.kernel = str(kernel)

