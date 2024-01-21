import numpy as np

from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError
from luma.interface.util import Matrix, Vector
from luma.metric.classification import Accuracy


__all__ = (
    'PerceptronClassifier'
)


class PerceptronClassifier(Estimator, Supervised):
    
    """
    The perceptron classifier is a simple linear binary classifier that 
    uses a weighted combination of features to make predictions. It learns 
    the weights through an iterative process, adjusting them based on 
    misclassified samples. The algorithm updates the weights based on the 
    difference between predicted and actual class labels. The perceptron 
    uses a step function to classify inputs into one of two classes. It's 
    a foundational algorithm in neural networks and machine learning, 
    suitable for linearly separable datasets.
    
    Parameters
    ----------
    `learning_rate` : Step-size of the updating procedure
    `max_iter` : Number of iteration
    `random_state` : Seed for random initialization of weights and biases
    
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 max_iter: int = 100,
                 random_state: int = None,
                 verbose: bool = False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.weights_ = []
        self.biases_ = []
        self.unique_classes_ = None
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> 'PerceptronClassifier':
        _, n = X.shape
        np.random.seed(self.random_state)
        self.unique_classes_ = np.unique(y)
        
        for cl in self.unique_classes_:
            weights_ = np.random.normal(0, 0.01, n)
            bias_ = np.random.normal(0, 0.01, 1)
            y_binary = np.where(y == cl, 1, 0)

            for i in range(self.max_iter):
                for idx, xi in enumerate(X):
                    linear_output = np.dot(xi, weights_) + bias_
                    y_pred = self._step_function(linear_output)
                    
                    update = self.learning_rate * (y_binary[idx] - y_pred)
                    weights_ += update * xi
                    bias_ += update

            self.weights_.append(weights_)
            self.biases_.append(bias_)
            
            if self.verbose and i % 10 == 0:
                print(f'[Perceptron] Iteration {i}/{self.max_iter}',
                      f'with weight-norm: {np.linalg.norm(self.weights_)}')
        
        self._fitted = True
        return self
    
    def _step_function(self, x: Vector) -> Vector:
        return np.where(x >= 0, 1, 0)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        preds = [np.dot(X, w) + b for w, b in zip(self.weights_, self.biases_)]
        return np.argmax(np.array(preds).T, axis=1)

    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
    
    def set_params(self, 
                   learning_rate: float = None,
                   max_iter: int = None,
                   random_state: int = None) -> None:
        if learning_rate is not None: self.learning_rate = float(learning_rate)
        if max_iter is not None: self.max_iter = int(max_iter) 
        if random_state is not None: self.random_state = int(random_state)

