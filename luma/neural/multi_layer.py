from typing import List
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector, ActivationUtil
from luma.interface.exception import NotFittedError
from luma.metric.classification import Accuracy
from luma.neural.activation import Softmax


__all__ = (
    'MLPClassifier'
)


class MLPClassifier(Estimator, Supervised):
    
    """
    An MLP (Multilayer Perceptron) is a type of neural network composed 
    of one input layer, one or more hidden layers, and one output layer, 
    with fully connected neurons. Each neuron uses a nonlinear activation 
    function to allow the network to capture complex patterns in the data. 
    MLPs are trained using backpropagation, where the network learns by 
    adjusting weights to minimize the difference between its predictions 
    and the actual data. They are widely used for classification, regression, 
    and other predictive tasks in machine learning.
    
    Parameters
    ----------
    `input_size` : Number of neurons in the input layer
    `hidden_sizes` : List of numbers of neurons in the hidden layers
    (an `int` for a single hidden layer)
    `output_size` : Number of neurons in the output layer
    `max_epoch` :  Maximum number of epochs
    `learning_rate` : Step size during the optimization process
    `lambda_` : L2 regularization strength
    `dropout_rate` : Dropout rate for each layer
    `activation` : Activation function for hidden layers
    (default `ReLU`)
    
    Properties
    ----------
    For getting weights and biases:
    ```py
        (property) weights: List[Matrix]
        (property) biases: List[Vector]
    ```
    For getting losses of each epoch:
    ```py
        (property) losses_: List[float]
    ```
    Methods
    -------
    For printing the configuration of an MLP:
        ```py
        def dump(self, padding: int = 4) -> None
        ```
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] | int,
                 output_size: int,
                 max_epoch: int = 1000,
                 learning_rate: float = 0.001,
                 lambda_: float = 0.01,
                 dropout_rate: float = 0.1,
                 activation: ActivationUtil.func_type = 'relu',
                 verbose: bool = False) -> None:
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.verbose = verbose
        self.losses_ = []
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'MLPClassifier':
        if isinstance(self.hidden_sizes, int):
            self.hidden_sizes = [self.hidden_sizes]
        
        self.layer_sizes = [
            self.input_size, 
            *self.hidden_sizes, 
            self.output_size
        ]
        self.n_layers = len(self.layer_sizes) - 1
        self.weights = []
        self.biases = []
        
        act = ActivationUtil(self.activation)
        self.act_ = act.activation_type()
        self.softmax = Softmax()
        
        for i in range(self.n_layers):
            weight_mat = np.random.randn(self.layer_sizes[i], 
                                         self.layer_sizes[i + 1])
            self.weights.append(weight_mat * 0.01)
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        
        y = self._one_hot_encode(y)
        for epoch in range(self.max_epoch):
            y_pred = self._forward_pass(X, is_train=True)
            self._backpropagation(X, y)
            
            loss = self._compute_loss(y, y_pred)
            self.losses_.append(loss)
            if self.verbose and epoch % 100 == 0:
                print(f"[MLPClassifier] Epoch {epoch}, Loss: {loss}")
        
        self._fitted = True
        return self
    
    def _one_hot_encode(self, y: Vector) -> Matrix:
        m = y.shape[0]
        n_classes = len(np.unique(y))
        one_hot = np.zeros((m, n_classes))
        one_hot[np.arange(m), y] = 1
        return one_hot
    
    def _compute_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        m, _ = y_true.shape
        log_L = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-8)
        cross_entropy = np.sum(log_L) / m
        
        l2_reg = (self.lambda_ / (2 * m))
        l2_reg *= np.sum([np.square(w).sum() for w in self.weights])
        return cross_entropy + l2_reg
    
    def _forward_pass(self, X: Matrix, is_train: bool = False) -> Matrix:
        a = X
        for i in range(self.n_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.act_.func(z)
            if is_train:
                mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape)
                mask = mask.astype(float) / (1 - self.dropout_rate)
                a *= mask
        
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = self.softmax.func(z)
        return a
    
    def _backpropagation(self, X: Matrix, y: Matrix) -> None:
        m, _ = X.shape
        as_, zs_ = [X], []
        a = X
        for i in range(self.n_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.act_.func(z)
            as_.append(a)
            zs_.append(z)
        
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = self.softmax.func(z)
        as_.append(a)
        zs_.append(z)
        
        delta = a - y
        dW = np.dot(as_[-2].T, delta)
        dW += (self.lambda_ / m) * self.weights[-1]
        db = np.sum(delta, axis=0, keepdims=True) / m
        
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db
        
        for i in range(self.n_layers - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T)
            delta *= self.act_.derivative(as_[i + 1])
            
            dW = np.dot(as_[i].T, delta)
            dW += (self.lambda_ / m) * self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        y_pred = self._forward_pass(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        return self._forward_pass(X)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
    
    def dump(self, padding: int = 4) -> None:
        lines = []
        lines.append("MLP Configuration")
        lines.append(f"Input Size : {self.input_size}")
        
        total_params = 0
        layers = self.layer_sizes
        for i, (in_, out_) in enumerate(zip(layers[:-1], layers[1:])):
            params_W = in_ * out_
            params_b = out_
            params_ = params_W + params_b
            total_params += params_
            
            type_ = "Hidden" if i < len(self.layer_sizes) - 2 else "Output"
            lines.append(f"Layer {i + 1} ({type_}) : {in_} -> {out_}, " + 
                         f"Parameters: {params_W} + {params_b} = {params_}")
        
        lines.append(f"Total Parameters: {total_params}")
        lines.append(f"Activation Function: {type(self.act_).__name__}")
        box_width = max(len(line) for line in lines) + 2 * padding
        print("+" + "-" * (box_width - 2) + "+")
        
        for line in lines: print("|" + line.center(box_width - 2) + "|")
        print("+" + "-" * (box_width - 2) + "+")

