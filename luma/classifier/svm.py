import numpy as np

from luma.interface.util import Matrix, KernelUtil
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.metric.classification import Accuracy


__all__ = (
    'SVC', 
    'KernelSVC'
)


class SVC(Estimator, Supervised):
    
    """
    Support Vector Classifier (SVC) is a supervised machine learning 
    algorithm mainly used for classification tasks. It operates by 
    determining a hyperplane that best separates different classes in 
    the input data, aiming to maximize the margin between the hyperplane 
    and the nearest data points from each class.
    
    Parameters
    ----------
    `C` : Regularization parameter
    `batch_size` : Size of a single batch
    `learning_rate` : Step-size of gradient descent update
    `max_iter` : Number of iteration
    
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
        
    def fit(self, X: Matrix, y: Matrix) -> 'SVC':
        classes = np.unique(y)
        self.models = []
        for cl in classes:
            binary_y = np.where(y == cl, 1, -1)
            self._binary_fit(X, binary_y, cl)
            if self.verbose:
                print(f'[SVC] Finished OvR fit for class {cl}\n')
        
        self._fitted = True
        return self

    def _binary_fit(self, X: Matrix, y: Matrix, label: int) -> None:
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
    
    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        scores = np.zeros((X.shape[0], len(self.models)))

        for i, (_, weight, bias) in enumerate(self.models):
            pred = np.dot(X, weight) + bias
            scores[:, i] = pred
        
        return np.argmax(scores, axis=1)

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class KernelSVC(Estimator, Supervised):
    
    """
    Kernel Support Vector Classification (SVC) with batch-based learning.
    This approach processes the training data in batches, making it suitable
    for large datasets that cannot fit into memory at once.
    
    Parameters
    ----------
    `C` : Regularization parameter
    `deg` : Polynomial Degree for `poly` kernel
    `gamma` : Shape parameter of Gaussian curve for `rbf` kernel
    `coef` : Coefficient for `poly`, `sigmoid` kernel
    `learning_rate` : Step-size for gradient descent update
    `max_iter` : Number of iteration
    `batch_size`: Size of the batch for batch-based learning
    `kernel` : Type of kernel (e.g., `linear`, `poly`, `rbf`, `sigmoid`)
    
    """
    
    def __init__(self,
                 C: float = 1.0,
                 deg: int = 3,
                 gamma: float = 1.0,
                 coef: float = 1.0,
                 learning_rate: float = 0.001,
                 max_iter: int = 1000,
                 batch_size: int = 100, 
                 kernel: KernelUtil.func_type = 'rbf',
                 verbose: bool = False) -> None:
        self.C = C
        self.deg = deg
        self.gamma = gamma
        self.coef = coef
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.kernel = kernel
        self.verbose = verbose
        self._kernel_func = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Matrix) -> 'KernelSVC':
        classes = np.unique(y)
        self.models = []
        self._set_kernel_func()
        
        for cl in classes:
            binary_y = np.where(y == cl, 1, -1)
            self._binary_fit(X, binary_y, cl)
            if self.verbose:
                print(f'[KernelSVC] Finished OvR fit for class {cl}')
        
        self._X = X
        self._y = y
        self._fitted = True
        return self

    def _binary_fit(self, X: Matrix, y: Matrix, label: int) -> None:
        m, _ = X.shape
        self.alpha = np.zeros(m)
        self.bias = 0
        
        batch_size = min(self.batch_size, m)
        for i in range(self.max_iter):
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                y_mul_kernel = np.outer(y_batch, y_batch) \
                    * self._kernel_func(X_batch, X_batch)
                
                gradient = np.ones(y_batch.shape[0])
                gradient -= y_mul_kernel.dot(self.alpha[start:end])
                
                self.alpha[start:end] += self.learning_rate * gradient
                self.alpha[start:end] = np.clip(self.alpha[start:end], 0, self.C)
            
            if self.verbose and i % 100 == 0:
                print(f'[KernelSVC] Finished iteration {i}/{self.max_iter}',
                      f'with alpha-norm: {np.linalg.norm(self.alpha)}')
        
        self._update_bias(X, y)
        self.models.append((label, X, y, self.alpha.copy(), self.bias))

    def _update_bias(self, X: Matrix, y: Matrix) -> None:
        alpha_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        bias_list = []
        for idx in alpha_idx:
            _append = y[idx] - (self.alpha * y).dot(self._kernel_func(X, X[idx]))
            bias_list.append(_append)
        
        self.bias = np.mean(bias_list) if bias_list else 0
    
    def _linear_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        return xi.dot(xj.T)

    def _poly_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        return (self.coef + xi.dot(xj.T)) ** self.deg

    def _rbf_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        norm = np.linalg.norm(xi[:, np.newaxis] - xj[np.newaxis, :], axis=2)
        return np.exp(-self.gamma * norm ** 2)

    def _sigmoid_kernel(self, xi: Matrix, xj: Matrix) -> Matrix:
        return np.tanh(2 * xi.dot(xj.T) + self.coef)

    def _set_kernel_func(self) -> None:
        if self.kernel in ('linear', 'lin'): 
            self._kernel_func = self._linear_kernel
        elif self.kernel in ('poly', 'polynomial'): 
            self._kernel_func = self._poly_kernel
        elif self.kernel in ('rbf', 'gaussian', 'Gaussian'): 
            self._kernel_func = self._rbf_kernel
        elif self.kernel in ('sigmoid', 'tanh'): 
            self._kernel_func = self._sigmoid_kernel

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        scores = np.zeros((X.shape[0], len(self.models)))

        for i, (_, _X, _y, alpha, bias) in enumerate(self.models):
            pred = (alpha * _y).dot(self._kernel_func(_X, X)) + bias
            scores[:, i] = pred

        return np.argmax(scores, axis=1)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

