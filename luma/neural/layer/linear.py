from typing import Tuple
import numpy as np

from interface.typing import TensorLike
from luma.core.super import Optimizer
from luma.interface.typing import Tensor, Matrix
from luma.interface.util import InitUtil
from luma.neural.base import Layer


__all__ = ("_Flatten", "_Dense")


class _Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: Tensor, is_train: bool = False) -> Matrix:
        _ = is_train
        self.input_ = X
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out: Matrix) -> Tensor:
        dX = d_out.reshape(self.input_.shape)
        return dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, *shape = in_shape
        return (batch_size, int(np.prod(shape)))


class _Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(self.in_features, self.out_features),
            b_shape=(1, self.out_features),
        )
        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Matrix, is_train: bool = False) -> Matrix:
        _ = is_train
        self.input_ = X
        return np.dot(X, self.weights_) + self.biases_

    def backward(self, d_out: Matrix) -> Matrix:
        X = self.input_

        self.dX = np.dot(d_out, self.weights_.T)
        self.dW = np.dot(X.T, d_out)
        self.dW += 2 * self.lambda_ * self.weights_
        self.dB = np.sum(d_out, axis=0, keepdims=True)

        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _ = in_shape
        return (batch_size, self.out_features)
