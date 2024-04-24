from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from luma.interface.typing import Matrix, Tensor, TensorLike
from luma.core.base import ModelBase
from luma.interface.util import InitUtil


__all__ = ("Layer", "Loss", "Initializer")


class Layer(ModelBase):
    """
    An internal class for layers in neural networks.

    Neural network layers are composed of interconnected nodes,
    each performing computations on input data. Common types include
    fully connected, convolutional, and recurrent layers, each
    serving distinct roles in learning from data.

    Attributes
    ----------
    - `weights_` : Weight tensor
    - `biases_` : Bias tensor
    - `dX` : Gradient w.r.t. the input
    - `dW` : Gradient w.r.t. the weights
    - `dB` : Gradient w.r.t. the biases
    - `optimizer` : Optimizer for certain layer
    - `out_shape` : Shape of the output when forwarding

    Properties
    ----------
    To get its parameter size (weights, biases):
    ```py
    (property) param_size: Tuple[int, int]
    ```
    """

    def __init__(self) -> None:
        self.input_: TensorLike = None
        self.output_: TensorLike = None

        self.weights_: TensorLike = None
        self.biases_: TensorLike = None

        self.dX: TensorLike = None
        self.dW: TensorLike = None
        self.dB: TensorLike = None

        self.optimizer: object = None
        self.out_shape: tuple = None

    def forward(self) -> TensorLike: ...

    def backward(self) -> TensorLike: ...

    def update(self) -> None:
        if self.optimizer is None:
            return
        weights_, biases_ = self.optimizer.update(
            self.weights_, self.biases_, self.dW, self.dB
        )
        self.weights_ = Tensor(weights_)
        self.biases_ = Tensor(biases_)

    def init_params(self, w_shape: tuple, b_shape: tuple) -> None:
        init_type_: type = InitUtil(self.initializer).initializer_type

        if init_type_ is None:
            self.weights_ = 0.01 * self.rs_.randn(*w_shape)
        else:
            if len(w_shape) == 2:
                self.weights_ = init_type_(self.random_state).init_2d(*w_shape)
            elif len(w_shape) == 4:
                self.weights_ = init_type_(self.random_state).init_4d(*w_shape)
            else:
                NotImplemented

        self.biases_: TensorLike = np.zeros(b_shape)

    @property
    def param_size(self) -> Tuple[int, int]:
        w_size, b_size = 0, 0
        if self.weights_ is not None:
            w_size += len(self.weights_.flatten())
        if self.biases_ is not None:
            b_size += len(self.biases_.flatten())

        return w_size, b_size

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        w_size, b_size = self.param_size
        return (
            f"{type(self).__name__}: "
            + f"({w_size:,} weights, {b_size:,} biases)"
            + f" -> {w_size + b_size:,} params"
        )


class Loss(ABC):
    """
    An internal class for loss functions used in neural networks.

    Loss functions, integral to the training process of machine
    learning models, serve as crucial metrics assessing the disparity
    between predicted outcomes and ground truth labels. They play a
    pivotal role in optimization algorithms, guiding parameter updates
    towards minimizing the discrepancy between predictions and true values.
    """

    def __init__(self) -> None:
        self.epsilon = 1e-12

    @abstractmethod
    def loss(self) -> float: ...

    @abstractmethod
    def grad(self) -> Matrix: ...

    def _clip(self, y: Matrix) -> Matrix:
        return np.clip(y, self.epsilon, 1 - self.epsilon)


class Initializer(ABC):
    """
    Abstract base class for initializing neural network weights.

    This class provides a structured way to implement weight
    initialization methods for different types of layers in a
    neural network.
    The class must be inherited by specific initializer implementations
    that define methods for 2D and 4D weight tensors.
    """

    def __init__(self, random_state: int) -> None:
        self.rs_ = np.random.RandomState(random_state)

    @classmethod
    def __class_alias__(cls) -> None: ...

    @abstractmethod
    def init_2d(self) -> Matrix: ...

    @abstractmethod
    def init_4d(self) -> Tensor: ...
