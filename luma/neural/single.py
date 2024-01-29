from typing import Literal
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.interface.util import Matrix, Vector
from luma.preprocessing.encoder import LabelBinarizer
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError


__all__ = (
    'PerceptronClassifier',
    'PerceptronRegressor'
)


class PerceptronClassifier(Estimator, Supervised):
    
    """
    A perceptron classifier is a simple linear binary classifier used in 
    machine learning. It makes predictions based on a linear combination of 
    input features and weights, followed by an activation function. The 
    perceptron updates its weights during training, aiming to correctly 
    classify training examples. It iteratively adjusts weights based on the 
    difference between predicted and actual outcomes. The perceptron's 
    simplicity makes it foundational in understanding more complex neural 
    networks.
    
    Parameters
    ----------
    `learning_rate` : Step size for updating weights
    `max_iter` : Maximum iteration
    `regularization` : Regularizing methods (e.g. `l1`, `l2`, `elastic-net`)
    `alpha` : Regularization strength
    `l1_ratio` : Ratio of `l1` (Only use when `elastic-net` regularization)
    `random_state` : Seed for random shuffling during SGD
    
    """    
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 regularization: Literal['l1', 'l2', 'elastic-net'] = None,
                 alpha: float = 0.0001,
                 l1_ratio: float = 0.5,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.verbose = verbose
        self.weights_ = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'PerceptronClassifier':
        m, n = X.shape
        np.random.seed(self.random_state)
        n_classes = len(np.unique(y))
        
        X = self._add_bias(X)
        y_binary = LabelBinarizer().fit_transform(y)
        self.weights_ = np.random.uniform(-0.01, 0.01, (n_classes, n + 1))
        
        for i in range(self.max_iter):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X, y_binary = X[indices], y_binary[indices]
            
            for xi, yi in zip(X, y_binary):
                linear_output = np.dot(self.weights_, xi)
                y_pred = self._softmax(linear_output)
                
                self.weights_ += self.learning_rate * np.outer(yi - y_pred, xi)
                if self.regularization:
                    self._regularize_weights()

            if self.verbose and i % 10 == 0:
                print(f'[Perceptron] Iteration {i}/{self.max_iter}',
                      f'with weight-norm: {np.linalg.norm(self.weights_)}')
        
        self._fitted = True
        return self
    
    def _add_bias(self, X: Matrix) -> Matrix:
        return np.insert(X, 0, 1, axis=1)
    
    def _softmax(self, X: Matrix) -> Matrix:
        X = X.reshape(-1, X.shape[-1])
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def _regularize_weights(self) -> None:
        if self.regularization == 'l1':
            self.weights_ -= self.alpha * np.sign(self.weights_)
        elif self.regularization == 'l2':
            self.weights_ -= self.alpha * self.weights_
        elif self.regularization == 'elastic-net':
            l1_term = self.l1_ratio * np.sign(self.weights_)
            l2_term = (1 - self.l1_ratio) * self.weights_
            self.weights_ -= self.alpha * (l1_term + l2_term)
        else:
            UnsupportedParameterError(self.regularization)
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        X = self._add_bias(X)
        linear_output = np.dot(X, self.weights_.T)
        y_pred = self._softmax(linear_output)
        
        return np.argmax(y_pred, axis=1)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class PerceptronRegressor(Estimator, Supervised):

    """
    Perceptron Regressor is a simple linear model used for regression tasks.
    It makes predictions based on a linear combination of input features and
    weights. The Perceptron Regressor updates its weights during training to
    minimize the difference between predicted and actual continuous outcomes.
    This model provides a basic understanding of linear regression in the
    context of neural networks.
    
    Parameters
    ----------
    `learning_rate` : Step size for updating weights
    `max_iter` : Maximum iteration
    `random_state` : Seed for random shuffling during SGD
    
    """

    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 regularization: Literal['l1', 'l2', 'elastic-net'] = None,
                 alpha: float = 0.0001,
                 l1_ratio: float = 0.5,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.verbose = verbose
        self.weights_ = None
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> 'PerceptronRegressor':
        m, n = X.shape
        np.random.seed(self.random_state)

        X = self._add_bias(X)
        self.weights_ = np.random.uniform(-0.01, 0.01, n + 1)

        for i in range(self.max_iter):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for xi, yi in zip(X, y):
                y_pred = np.dot(self.weights_, xi)
                error = yi - y_pred
                
                self.weights_ += self.learning_rate * error * xi
                if self.regularization:
                    self._regularize_weights()

            if self.verbose and i % 10 == 0:
                print(f'[Perceptron] Iteration {i}/{self.max_iter}',
                      f'with weight-norm: {np.linalg.norm(self.weights_)}')

        self._fitted = True
        return self

    def _add_bias(self, X: Matrix) -> Matrix:
        return np.insert(X, 0, 1, axis=1)
    
    def _regularize_weights(self) -> None:
        if self.regularization == 'l1':
            self.weights_ -= self.alpha * np.sign(self.weights_)
        elif self.regularization == 'l2':
            self.weights_ -= self.alpha * self.weights_
        elif self.regularization == 'elastic-net':
            l1_term = self.l1_ratio * np.sign(self.weights_)
            l2_term = (1 - self.l1_ratio) * self.weights_
            self.weights_ -= self.alpha * (l1_term + l2_term)
        else:
            UnsupportedParameterError(self.regularization)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        X = self._add_bias(X)
        return np.dot(X, self.weights_)

    def score(self, X: Matrix, y: Vector, 
              metric: Evaluator = MeanSquaredError) -> float:
        y_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=y_pred)

