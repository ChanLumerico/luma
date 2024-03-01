from typing import Tuple
from scipy.linalg import svd, eigh
import numpy as np

from luma.interface.util import Matrix, Vector, KernelUtil
from luma.core.super import Transformer, Unsupervised, Supervised
from luma.interface.exception import (NotFittedError,
                                      NotConvergedError,
                                      UnsupportedParameterError)


__all__ = (
    'PCA', 
    'LDA', 
    'KDA', 
    'CCA', 
    'KernelPCA', 
    'TruncatedSVD', 
    'FactorAnalysis'
)


class PCA(Transformer, Unsupervised):
    
    """
    PCA, or Principal Component Analysis, is a dimensionality 
    reduction technique. It's primarily used to simplify complex 
    data while retaining the most important information.
    
    Parameters
    ----------
    `n_components` : Number of principal components
    
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, X: Matrix) -> 'PCA':
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

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)

    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)


class KernelPCA(Transformer, Unsupervised):
    
    """
    Kernel PCA is a dimensionality reduction technique that extends 
    Principal Component Analysis (PCA) to capture complex, nonlinear 
    relationships in data by using a kernel function to transform 
    the data into a higher-dimensional space where nonlinear patterns 
    can be better captured.
    
    Parameters
    ----------
    `n_components` : Number of principal components
    `deg` : Polynomial degree of `poly` kernel
    `gamma` : Shape parameter of `rbf`, `sigmoid`, `laplacian`
    `coef` : Additional coefficient of `poly`, `sigmoid`
    `kernel` : Type of kernel functions
    (e.g. `linear`, `poly`, `rbf`, `sigmoid`, `laplacian`)
    
    """
    
    def __init__(self, 
                 n_components: int = None,
                 deg: int = 3,
                 gamma: float = 15.0,
                 coef: float = 1.0,
                 kernel: KernelUtil.func_type = 'rbf') -> None:
        self.n_components = n_components
        self.deg = deg
        self.gamma = gamma
        self.coef = coef
        self.kernel = kernel
        self.X = None
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'KernelPCA':
        self.X = X
        self._select_kernel_function()
        N = X.shape[0]
        self.K = self._compute_kernel_matrix(X, X)
        
        one_n = np.ones((N, N)) / N
        self.K_centered = self.K - one_n.dot(self.K) - self.K.dot(one_n)
        self.K_centered += one_n.dot(self.K).dot(one_n)
        
        self.eigvals, self.eigvecs = eigh(self.K_centered)
        self.eigvals, self.eigvecs = self.eigvals[::-1], self.eigvecs[:, ::-1]

        self._fitted = True
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        
        K_new = self._compute_kernel_matrix(X, self.X)
        N = self.K.shape[0]
        one_N = np.ones((N, N)) / N
        one_new = np.ones((K_new.shape[0], N)) / N
        
        K_new_centered = K_new - one_new.dot(self.K)
        K_new_centered += one_new.dot(one_N.dot(self.K))
        K_new_centered -= K_new.dot(one_N.mean(axis=0))
        
        proj = K_new_centered.dot(self.eigvecs[:, :self.n_components])
        return proj
    
    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)
    
    def _compute_kernel_matrix(self, X: Matrix, Y: Matrix) -> Matrix:
        N, M = X.shape[0], Y.shape[0]
        K = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                K[i, j] = self.kernel_func(X[i], Y[j])
        return K
    
    def _select_kernel_function(self):
        if self.kernel == 'linear': self.kernel_func = self._linear
        elif self.kernel == 'poly': self.kernel_func = self._poly
        elif self.kernel == 'rbf': self.kernel_func = self._rbf
        elif self.kernel == 'sigmoid': self.kernel_func = self._sigmoid
        elif self.kernel == 'laplacian': self.kernel_func = self._laplacian
        else: raise UnsupportedParameterError(self.kernel)
    
    def _linear(self, x: Vector, y: Vector) -> float:
        return np.dot(x, y)
    
    def _poly(self, x: Vector, y: Vector) -> float:
        return (np.dot(x, y) + self.coef) ** self.deg
    
    def _rbf(self, x: Vector, y: Vector) -> float:
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)
    
    def _sigmoid(self, x: Vector, y: Vector) -> float:
        return np.tanh(self.gamma * np.dot(x, y) + self.coef)
    
    def _laplacian(self, x: Vector, y: Vector) -> float:
        return np.exp(-self.gamma * np.linalg.norm(x - y))


class LDA(Transformer, Supervised):
    
    """
    Linear Discriminant Analysis (LDA) is a dimensionality reduction 
    and classification technique. It's primarily employed for dimensionality 
    reduction and feature extraction while also considering class 
    information for classification tasks.
    
    Parameters
    ----------
    `n_components` : Number of linear discriminants
    
    Notes
    -----
    * To use LDA for classification, refer to 
        `luma.classifier.discriminant.LDAClassifier`
    
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'LDA':
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

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return np.dot(X, self.eigenvectors)

    def fit_transform(self, X: Matrix, y: Matrix) -> Matrix:
        self.fit(X, y)
        return self.transform(X)


class TruncatedSVD(Transformer, Unsupervised):
    
    """
    runcated Singular Value Decomposition (TruncatedSVD) is a linear dimensionality 
    reduction method that simplifies large datasets by preserving the most relevant 
    information. It achieves this by decomposing a matrix into three components 
    using singular value decomposition (SVD) and retaining only a specified number 
    of important components.
    
    Parameters
    ----------
    `n_components` : Dimensionality of low-space
    
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self._fitted = False
    
    def fit(self, X: Matrix) -> 'TruncatedSVD':
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        
        self.U = U[:, :self.n_components]
        self.S = np.diag(S[:self.n_components])
        self.VT = VT[:self.n_components, :]
        self._fitted = True
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        return np.dot(X_centered, self.VT.T)
    
    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)


class FactorAnalysis(Transformer, Unsupervised):
    
    """
    Factor Analysis is a statistical method used to uncover latent factors 
    influencing observed variables. It assumes that observed variables share 
    common variance due to these unobservable factors, expressed through 
    factor loadings. The technique helps simplify data interpretation by 
    revealing the underlying structure of the dataset.
    
    Parameters
    ----------
    `n_components` : Dimensionality of low-space
    `max_iter` : Number of iterations
    `tol` : Threshold for convergence
    `noise_variance` : Initial variances for noise of each features
    
    """
    
    def __init__(self,
                 n_components: int = None,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 noise_variance: Matrix | list = None,
                 verbose: bool = False) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.noise_variance = noise_variance
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: Matrix) -> 'FactorAnalysis':
        m, n = X.shape
        self.mean = X.mean(axis=0)
        X -= self.mean
        
        logL_const = n * np.log(2 * np.pi) * self.n_components
        variance = X.var(axis=0)
        psi = np.ones(n)
        if self.noise_variance:
            psi = Matrix(self.noise_variance)
        
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
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        Ih = np.eye(len(self.W))
        W_psi = self.W / self.noise_variance if self.noise_variance else self.W
        cov = np.linalg.inv(Ih + W_psi.dot(self.W.T))
        return np.dot((X - self.mean).dot(W_psi.T), cov)
    
    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)


class KDA(Transformer, Supervised):
    
    """
    Kernel Discriminant Analysis (KDA) is a machine learning technique for 
    dimensionality reduction and classification that extends Linear 
    Discriminant Analysis (LDA) to nonlinear feature spaces using kernel 
    methods. It projects data into a lower-dimensional space where classes 
    are more separable by maximizing the ratio of between-class variance 
    to within-class variance. KDA utilizes kernel functions to implicitly 
    map input data to a high-dimensional space without explicitly computing 
    the coordinates in that space.
    
    Parameters
    ----------
    `n_components` : Dimensionality of low-space
    `deg` : Polynomial degree of `poly` kernel
    `gamma` : Shape parameter of `rbf`, `sigmoid`, `laplacian`
    `coef` : Additional coefficient of `poly`, `sigmoid`
    `kernel` : Type of kernel functions
    
    Notes
    -----
    * To use KDA for classification, refer to 
        `luma.classifier.discriminant.KDAClassifier`
    
    """
    
    def __init__(self, 
                 n_components: int = None, 
                 deg: int = 2,
                 alpha: float = 1.0,
                 gamma: float = 1.0,
                 coef: int = 0.0,
                 kernel: KernelUtil.func_type = 'rbf') -> None:
        self.n_components = n_components
        self.deg = deg
        self.alpha = alpha
        self.gamma = gamma
        self.coef = coef
        self.kernel = kernel
        self.X_ = None
        self._fitted = False
        
        self.kernel_params = {
            'deg': self.deg,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'coef': self.coef
        }
    
    def fit(self, X: Matrix, y: Vector) -> 'KDA':
        m, _ = X.shape
        self.X_ = X
        self.classes = np.unique(y)
        self.ku_ = KernelUtil(self.kernel, **self.kernel_params)
        
        K = self.ku_.kernel_func(X)
        M = np.mean(K, axis=0)
        Sw = np.zeros((m, m))
        Sb = np.zeros((m, m))
        
        for i in self.classes:
            Xi = K[y == i]
            Mi = np.mean(Xi, axis=0)
            Ni = Xi.shape[0]
            
            Sw += np.dot((Xi - Mi).T, Xi - Mi)
            Sb += Ni * np.outer(Mi - M, Mi - M)
        
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw).dot(Sb))
        indices = np.argsort(eigvals)[::-1]
        self.eigvecs = eigvecs[:, indices[:self.n_components]]
        
        self._fitted = True
        return self
    
    def transform(self, X: Matrix) -> Vector:
        K = self.ku_.kernel_func(X, self.X_)
        return K.dot(self.eigvecs)
    
    def fit_transform(self, X: Matrix, y: Vector) -> Vector:
        self.fit(X, y)
        return self.transform(X)


class CCA(Transformer, Unsupervised):
    
    """
    Canonical Correlation Analysis (CCA) is a multivariate statistical method 
    that finds linear combinations of two sets of variables with the highest 
    correlation. It aims to uncover the underlying correlation structure 
    between the two variable sets. CCA is an unsupervised technique, often 
    used for exploring the relationships in complex data. The result is pairs 
    of canonical variables (or components) that represent the maximal 
    correlations between the sets.
    
    Parameters
    ----------
    `n_components` : Dimensionality of low-space
    
    Notes
    -----
    * `CCA` requires two distinct datasets `X` and `Y`, 
        in which `Y` is not a target variable
    * Due to its uniqueness in its parameters, 
        `CCA` may not be compatible with several meta estimators
    * `transform()` and `fit_transform()` returns a 2-tuple of `Matrix`
        
        ```py
        def transform(self, X: Matrix, Y: Matrix) -> Tuple[Matrix, Matrix]
        ```
    """
    
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self.correlations_ = None
        self._fitted = False
    
    def fit(self, X: Matrix, Y: Matrix) -> 'CCA':
        _, n = X.shape
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)
        
        Cxx = np.cov(X.T)
        Cyy = np.cov(Y.T)
        Cxy = np.cov(X.T, Y.T)[:n, n:]
        
        inv_Cxx = np.linalg.pinv(Cxx)
        inv_Cyy = np.linalg.pinv(Cyy)
        eigvals, x_weights = np.linalg.eig(inv_Cxx @ Cxy @ inv_Cyy @ Cxy.T)
        
        indices = eigvals.argsort()[::-1]
        self.x_weights = x_weights[:, indices][:, :self.n_components]
        self.y_weights = inv_Cyy @ Cxy.T @ self.x_weights 
        self.y_weights /= eigvals[indices][:self.n_components]
        
        self.x_scores = X.dot(self.x_weights)
        self.y_scores = Y.dot(self.y_weights)
        
        self.correlations_ = [np.corrcoef(self.x_scores[:, i], 
                                          self.y_scores[:, i])[0, 1]
                              for i in range(self.n_components)]
        
        self.x_loadings = Cxy.dot(self.y_weights)
        self.y_loadings = Cxy.T.dot(self.x_weights)
        
        self._fitted = True
        return self
    
    def transform(self, X: Matrix, Y: Matrix) -> Tuple[Matrix, Matrix]:
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)
        X_trans = X.dot(self.x_weights)
        Y_trans = Y.dot(self.y_weights)
        
        return X_trans, Y_trans
    
    def fit_transform(self, X: Matrix, Y: Matrix) -> Tuple[Matrix, Matrix]:
        self.fit(X, Y)
        return self.transform(X, Y)

