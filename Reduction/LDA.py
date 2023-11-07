from typing import *
from typing import Any, Tuple
import numpy as np

from LUMA.Interface.Super import _Transformer, _Supervised


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

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.eigenvectors)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, n_components: int=None) -> None:
        if n_components is not None: self.n_components = int(n_components)

