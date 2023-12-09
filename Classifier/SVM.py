from typing import *
import numpy as np

from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.interface.super import Estimator, Evaluator, Supervised
from luma.metric.classification import Accuracy


__all__ = ['SVC', 'KernelSVC']


class SVC(Estimator, Supervised):
    
    """
    Support Vector Classifier (SVC) is a supervised machine learning 
    algorithm mainly used for classification tasks. It operates by 
    determining a hyperplane that best separates different classes in 
    the input data, aiming to maximize the margin between the hyperplane 
    and the nearest data points from each class.
    
    Parameters
    ----------
    ``C`` : Regularization parameter \n
    ``batch_size`` : Size of a single batch \n
    ``learning_rate`` : Step-size of gradient descent update \n
    ``max_iter`` : Number of iteration
    
    """
    
    def __init__(self, 
                 C: float = 1.0, 
                 batch_size: int = 100, 
                 learning_rate: float = 0.001, 
                 max_iter: int = 1000, 
                 verbose: bool = False) -> None:
        self.C = C
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVC':
        classes = np.unique(y)
        self.models = []
        for cl in classes:
            binary_y = np.where(y == cl, 1, -1)
            self._binary_fit(X, binary_y, cl)
            if self.verbose:
                print(f'[SVC] Finished OvR fit for class {cl}\n')
        
        self._fitted = True
        return self

    def _binary_fit(self, X: np.ndarray, y: np.ndarray, label: int) -> None:
        m, n = X.shape
        id = np.arange(m)
        np.random.shuffle(id)

        bias = 0.0
        weight = np.zeros(n)
        
        for i in range(self.max_iter):
            for batch_point in range(0, m, self.batch_size):
                gradient_w, gradient_b = 0, 0
                for j in range(batch_point, batch_point + self.batch_size):
                    if j >= m: continue
                    idx = id[j]
                    disc = y[idx] * (np.dot(weight, X[idx].T) + bias)
                    if disc <= 1:
                        gradient_w += self.C * y[idx] * X[idx]
                        gradient_b += self.C * y[idx]

                weight -= self.learning_rate * weight
                weight += self.learning_rate * gradient_w
                bias += self.learning_rate * gradient_b

            if self.verbose and i % 100 == 0 and i:
                print(f'[SVC] Finished iteration {i}/{self.max_iter}', end=' ')
                print(f'with weight-norm of {np.linalg.norm(weight)}')

        self.models.append((label, weight.copy(), bias))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        scores = np.zeros((X.shape[0], len(self.models)))

        for i, (_, weight, bias) in enumerate(self.models):
            pred = np.dot(X, weight) + bias
            scores[:, i] = pred

        return np.argmax(scores, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self,
                   C: float = None,
                   batch_size: int = None,
                   learning_rate: float = None,
                   max_iter: int = None) -> None:
        if C is not None: self.C = float(C)
        if batch_size is not None: self.batch_size = int(batch_size)
        if learning_rate is not None: self.learning_rate = float(learning_rate)
        if max_iter is not None: self.max_iter = int(max_iter)


class KernelSVC(Estimator, Supervised):
    
    """
    Kernel Support Vector Classification (SVC) is an extension of the 
    Support Vector Machine (SVM) algorithm that facilitates the classification 
    of non-linearly separable data by mapping it into a higher-dimensional space
    using a kernel function. The kernel function enables the algorithm to 
    establish a linear decision boundary in the transformed space, allowing 
    for more complex and flexible classification models. 
    
    Parameters
    ----------
    ``C`` : Regularization parameter \n
    ``deg`` : Polynomial Degree for `poly` kernel \n
    ``gamma`` : Shape parameter of Gaussian curve for `rbf` kernel \n
    ``coef`` : Coefficient for `poly`, `sigmoid` kernel \n
    ``learning_rate`` : Step-size for gradient descent update \n
    ``max_iter`` : Number of iteration \n
    ``kernel`` : Type of kernel (e.g. `linear`, `poly`, `rbf`, `sigmoid`)
    
    """
    
    def __init__(self,
                 C: float = 1.0,
                 deg: int = 3,
                 gamma: float = 1.0,
                 coef: float = 1.0,
                 learning_rate: float = 0.001,
                 max_iter: int = 1000,
                 kernel: Literal['linear', 'poly', 'rbf', 'sigmoid'] = 'rbf',
                 verbose: bool = False) -> None:
        self.C = C
        self.deg = deg
        self.gamma = gamma
        self.coef = coef
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.kernel = kernel
        self.verbose = verbose
        self._kernel_func = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelSVC':
        classes = np.unique(y)
        self.models = []
        self._set_kernel_func()
        
        for cl in classes:
            binary_y = np.where(y == cl, 1, -1)
            self._binary_fit(X, binary_y, cl)
            if self.verbose:
                print(f'[KernelSVC] Finished OvR fit for class {cl}\n')
        
        self._X = X
        self._y = y
        self._fitted = True
        return self

    def _binary_fit(self, X: np.ndarray, y: np.ndarray, label: int) -> None:
        m, _ = X.shape
        self.alpha = np.random.random(m)
        self.bias = 0
        
        y_mul_kernel = np.outer(y, y) * self._kernel_func(X, X)
        for i in range(self.max_iter):
            gradient = np.ones(m) - y_mul_kernel.dot(self.alpha)
            self.alpha += self.learning_rate * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)
            
            if self.verbose and i % 100 == 0 and i:
                print(f'[KernelSVC] Finished iteration {i}/{self.max_iter}', end=' ')
                print(f'with alpha-norm: {np.linalg.norm(self.alpha)}')
        
        alpha_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        bias_list = []
        for idx in alpha_idx:
            _append = y[idx] - (self.alpha * y).dot(self._kernel_func(X, X[idx]))
            bias_list.append(_append)
        self.bias = np.mean(bias_list)
        
        self.models.append((label, X, y, self.alpha.copy(), self.bias))
    
    def _linear_kernel(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return xi.dot(xj.T)

    def _poly_kernel(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return (self.coef + xi.dot(xj.T)) ** self.deg

    def _rbf_kernel(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(xi[:, np.newaxis] - xj[np.newaxis, :], axis=2)
        return np.exp(-self.gamma * norm ** 2)

    def _sigmoid_kernel(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return np.tanh(2 * xi.dot(xj.T) + self.coef)

    def _set_kernel_func(self) -> None:
        if self.kernel == 'linear': self._kernel_func = self._linear_kernel
        elif self.kernel == 'poly': self._kernel_func = self._poly_kernel
        elif self.kernel == 'rbf': self._kernel_func = self._rbf_kernel
        elif self.kernel == 'sigmoid': self._kernel_func = self._sigmoid_kernel
        else: raise UnsupportedParameterError(self.kernel)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        scores = np.zeros((X.shape[0], len(self.models)))

        for i, (_, _X, _y, alpha, bias) in enumerate(self.models):
            pred = (alpha * _y).dot(self._kernel_func(_X, X)) + bias
            scores[:, i] = pred

        return np.argmax(scores, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self,
                   C: float = None,
                   deg: int = None,
                   gamma: float = None,
                   coef: float = None,
                   learning_rate: float = None,
                   max_iter: int = None,
                   kernel: Literal = None) -> None:
        if C is not None: self.C = float(C)
        if deg is not None: self.deg = int(deg)
        if gamma is not None: self.gamma = float(gamma)
        if coef is not None: self.coef = float(coef)
        if learning_rate is not None: self.learning_rate = float(learning_rate)
        if max_iter is not None: self.max_iter = int(max_iter)
        if kernel is not None: self.kernel = str(kernel)

