from math import log
from scipy.linalg import svd
import numpy as np

from luma.interface.super import Transformer, Unsupervised, Supervised
from luma.interface.exception import NotFittedError, NotConvergedError


__all__ = ['PCA', 'LDA', 'TruncatedSVD', 'FactorAnalysis']


class PCA(Transformer, Unsupervised):
    
    """
    PCA, or Principal Component Analysis, is a dimensionality 
    reduction technique. It's primarily used to simplify complex 
    data while retaining the most important information.
    
    Parameters
    ----------
    ``n_components`` : Number of principal components
    
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'PCA':
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
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def set_params(self, n_components: int = None) -> None:
        if n_components is not None: self.n_components = int(n_components)


class LDA(Transformer, Supervised):
    
    """
    Linear Discriminant Analysis (LDA) is a dimensionality reduction 
    and classification technique. It's primarily employed for dimensionality 
    reduction and feature extraction while also considering class 
    information for classification tasks.
    
    Parameters
    ----------
    ``n_components`` : Number of linear discriminants
    
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
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
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return np.dot(X, self.eigenvectors)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, n_components: int = None) -> None:
        if n_components is not None: self.n_components = int(n_components)


class TruncatedSVD(Transformer, Unsupervised):
    
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
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'TruncatedSVD':
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        
        self.U = U[:, :self.n_components]
        self.S = np.diag(S[:self.n_components])
        self.VT = VT[:self.n_components, :]
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        return np.dot(X_centered, self.VT.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def set_params(self, n_components: int = None) -> None:
        if n_components is not None: self.n_components = int(n_components)


class FactorAnalysis(Transformer, Unsupervised):
    
    """
    Factor Analysis is a statistical method used to uncover latent factors 
    influencing observed variables. It assumes that observed variables share 
    common variance due to these unobservable factors, expressed through 
    factor loadings. The technique helps simplify data interpretation by 
    revealing the underlying structure of the dataset.
    
    Parameters
    ----------
    ``n_components`` : Dimeensionality of low-space \n
    ``max_iter`` : Number of iterations \n
    ``tol`` : Threshold for convergence \n
    ``noise_variance`` : Initial variances for noise of each features
    
    """
    
    def __init__(self,
                 n_components: int = None,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 noise_variance: np.ndarray | list = None,
                 verbose: bool = False) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.noise_variance = noise_variance
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'FactorAnalysis':
        m, n = X.shape
        self.mean = X.mean(axis=0)
        X -= self.mean
        
        logL_const = n * log(2 * np.pi) * self.n_components
        variance = X.var(axis=0)
        psi = np.ones(n)
        if self.noise_variance:
            psi = np.array(self.noise_variance)
        
        logLikelihood = []
        prev_logL = -np.inf
        for i in range(self.max_iter):
            sqrt_psi = np.sqrt(psi) + 1e-12
            _, _S, _VT = svd(X / (sqrt_psi * m ** 0.5), full_matrices=False)
            S, VT = _S[:self.n_components], _VT[:self.n_components]
            unexp_var = np.linalg.norm(S) ** 2
            
            S **= 2
            W = np.sqrt(np.maximum(S - 1.0, 0.0))[:, np.newaxis] * VT
            W *= sqrt_psi
            
            logL = np.sum(np.log(S)) + np.sum(np.log(psi))
            logL += logL_const + unexp_var
            logL *= -0.5 * m
            logLikelihood.append(logL)
            
            abs_diff = abs(logL - prev_logL)
            if abs_diff < self.tol: 
                if self.verbose:
                    print(f'[FA] Ealry Convergence at iteration {i}/{self.max_iter}', end=' ')
                    print(f'with delta-logL of {abs_diff}')
                break
            prev_logL = logL            
            psi = np.maximum(variance - np.sum(W ** 2, axis=0), 1e-12)
            
            if self.verbose and i % 100 == 0 and i:
                print(f'[FA] Iteration {i}/{self.max_iter} finished', end=' ')
                print(f'with delta-logL of {abs_diff}')
                
        else: NotConvergedError(self)
        
        self.W = W
        self._fitted = True
        return self

    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        Ih = np.eye(len(self.W))
        W_psi = self.W / self.noise_variance if self.noise_variance else self.W
        cov = np.linalg.inv(Ih + W_psi.dot(self.W.T))
        return np.dot((X - self.mean).dot(W_psi.T), cov)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def set_params(self,
                   n_components: int = None,
                   max_iter: int = None,
                   tol: float = None,
                   noise_variance: np.ndarray | list = None) -> None:
        if n_components is not None: self.n_components = int(n_components)
        if max_iter is not None: self.max_iter = int(max_iter)
        if tol is not None: self.tol = float(tol)
        if noise_variance is not None: self.noise_variance = np.array(noise_variance)

    