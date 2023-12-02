from typing_extensions import Self
import numpy as np

from LUMA.Interface.Exception import NotFittedError
from LUMA.Interface.Super import _Estimator
from LUMA.Interface.Type import Evaluator
from LUMA.Metric.Classification import Accuracy


__all__ = ['LinearSVC']


class LinearSVC(_Estimator):
    
    """
    Linear Support Vector Classifier (Linear SVC) is a supervised machine 
    learning algorithm mainly used for classification tasks. It operates 
    by determining a hyperplane that best separates different classes in 
    the input data, aiming to maximize the margin between the hyperplane 
    and the nearest data points from each class.
    
    Parameters
    ----------
    ``learning_rate`` : Step-size of gradient descent update \n
    ``lambda_param`` : Strength of regularization \n
    ``max_iter`` : Number of iteration
    
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 lambda_param: float = 0.01,
                 max_iter: int = 1000,
                 verbose: bool = False) -> None:
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        _, n = X.shape
        self.n_classes = len(np.unique(y))
        self.weights = np.zeros((self.n_classes, n))
        self.biases = np.zeros(self.n_classes)
        self._sub_fitted = [False] * self.n_classes
        
        for cl in range(self.n_classes):
            y_binary = np.where(y == cl, 1, -1)
            self._binary_fit(X, y_binary, cl)
            if self.verbose:
                print(f'[LinearSVC] Finished OvR fit for class {cl}\n')
        
        self._fitted = True
        if False in self._sub_fitted: self._fitted = False
        
        return self
    
    def _binary_fit(self, X: np.ndarray, y: np.ndarray, label: int) -> None:
        _, n = X.shape
        weight = np.zeros(n)
        bias = 0
        
        for i in range(self.max_iter):
            for idx, xi in enumerate(X):
                condition = y[idx] * (xi.dot(weight) - bias) >= 1
                gradient = 2 * self.lambda_param * weight
                if not condition:
                    gradient -= xi * y[idx]
                    bias -= self.learning_rate * y[idx]
                
                weight -= self.learning_rate * gradient
            
            if self.verbose and i % 100 == 0 and i:
                print(f'[LinearSVC] Finished iteration {i}/{self.max_iter}', end=' ')
                print(f'for class {label} with weight-norm: {np.linalg.norm(weight)}')
        
        self.weights[label] = weight
        self.biases[label] = bias
        self._sub_fitted[label] = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        m, _ = X.shape
        if not self._fitted: raise NotFittedError(self)
        
        pred = np.zeros((m, self.n_classes))
        for cl in range(self.n_classes):
            pred[:, cl] = X.dot(self.weights[cl]) - self.biases[cl]
        
        return np.argmax(pred, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self,
                   learning_rate: float = None,
                   lambda_param: float = None,
                   max_iter: int = None) -> None:
        if learning_rate is not None: self.learning_rate = float(learning_rate)
        if lambda_param is not None: self.lambda_param = float(lambda_param)
        if max_iter is not None: self.max_iter = int(max_iter)

