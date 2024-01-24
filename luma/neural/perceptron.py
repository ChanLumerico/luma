import numpy as np

from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError
from luma.interface.util import Matrix, Vector
from luma.preprocessing.encoder import LabelBinarizer
from luma.metric.classification import Accuracy


__all__ = (
    'PerceptronClassifier',
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
    `random_state` : Seed for random shuffling during SGD
    
    """    
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iter: int = 100,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
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
    
    def set_params(self, 
                   learning_rate: float = None,
                   max_iter: int = None,
                   random_state: int = None) -> None:
        if learning_rate is not None: self.learning_rate = float(learning_rate)
        if max_iter is not None: self.max_iter = int(max_iter)
        if random_state is not None: self.random_state = int(random_state)

