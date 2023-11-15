from typing import *
import numpy as np

from LUMA.Interface.Super import _Transformer, _Unsupervised, _Supervised
from LUMA.Interface.Exception import NotFittedError


class PCA(_Transformer, _Unsupervised):
    
    """
    PCA, or Principal Component Analysis, is a dimensionality 
    reduction technique. It's primarily used to simplify complex 
    data while retaining the most important information.
    
    Parameters
    ----------
    ``n_components`` : Number of principal components
    
    """
    
    def __init__(self, n_components: int=None) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if self.n_components is not None:
            self.eigenvalues = eigenvalues[:self.n_components]
            self.eigenvectors = eigenvectors[:, :self.n_components]
        else:
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def set_params(self, n_components: int=None) -> None:
        if n_components is not None: self.n_components = int(n_components)


class LDA(_Transformer, _Supervised):
    
    """
    Linear Discriminant Analysis (LDA) is a dimensionality reduction 
    and classification technique. It's primarily employed for dimensionality 
    reduction and feature extraction while also considering class 
    information for classification tasks.
    
    Parameters
    ----------
    ``n_components`` : Number of linear discriminants
    
    """
    
    def __init__(self, n_components: int=None) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.classes = np.unique(y)
        self.class_means = [np.mean(X[y == c], axis=0) for c in self.classes]

        within_class_scatter = np.zeros((X.shape[1], X.shape[1]))
        for c in self.classes:
            X_c = X[y == c]
            mean_diff = X_c - self.class_means[c]
            within_class_scatter += np.dot(mean_diff.T, mean_diff)
            
        between_class_scatter = np.zeros((X.shape[1], X.shape[1]))
        for c in self.classes:
            n = X[y == c].shape[0]
            mean_diff = self.class_means[c] - np.mean(X)
            between_class_scatter += n * np.outer(mean_diff, mean_diff)
            
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(within_class_scatter)
                                                   .dot(between_class_scatter))

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if self.n_components is not None:
            self.eigenvalues = eigenvalues[:self.n_components]
            self.eigenvectors = eigenvectors[:, :self.n_components]
        else:
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return np.dot(X, self.eigenvectors)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, n_components: int=None) -> None:
        if n_components is not None: self.n_components = int(n_components)


class TruncatedSVD(_Transformer, _Unsupervised):
    
    """
    runcated Singular Value Decomposition (TruncatedSVD) is a linear dimensionality 
    reduction method that simplifies large datasets by preserving the most relevant 
    information. It achieves this by decomposing a matrix into three components 
    using singular value decomposition (SVD) and retaining only a specified number 
    of important components.
    
    Parameters
    ----------
    ``n_components`` : Dimensionality of low-space
    
    """
    
    def __init__(self, n_components: int=None) -> None:
        self.n_components = n_components
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        
        self.U = U[:, :self.n_components]
        self.S = np.diag(S[:self.n_components])
        self.VT = VT[:self.n_components, :]
        self._fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        return np.dot(X_centered, self.VT.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def set_params(self, n_components: int=None) -> None:
        if n_components is not None: self.n_components = int(n_components)

