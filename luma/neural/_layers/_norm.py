from typing import Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.neural.base import Layer


__all__ = (
    "_BatchNorm1D",
    "_BatchNorm2D",
    "_BatchNorm3D",
    "_LRN_1D",
    "_LRN_2D",
    "_LRN_3D",
)


class _BatchNorm1D(Layer):
    def __init__(
        self,
        in_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, in_features, 1))
        self.beta = np.zeros((1, in_features, 1))
        self.weights_ = [self.gamma, self.beta]

        self.running_mean = np.zeros((1, in_features, 1))
        self.running_var = np.ones((1, in_features, 1))

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            batch_mean = np.mean(X, axis=(0, 2), keepdims=True)
            batch_var = np.var(X, axis=(0, 2), keepdims=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            self.X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            self.X_norm = (X - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )

        out = self.weights_[0] * self.X_norm + self.weights_[1]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        batch_size, _, width = d_out.shape

        dX_norm = d_out * self.weights_[0]
        dgamma = np.sum(d_out * self.X_norm, axis=(0, 2), keepdims=True)
        dbeta = np.sum(d_out, axis=(0, 2), keepdims=True)
        self.dW = [dgamma, dbeta]

        dX = (
            1.0
            / (batch_size * width)
            * np.reciprocal(np.sqrt(self.running_var + self.epsilon))
            * (
                batch_size * width * dX_norm
                - np.sum(dX_norm, axis=(0, 2), keepdims=True)
                - self.X_norm
                * np.sum(dX_norm * self.X_norm, axis=(0, 2), keepdims=True)
            )
        )
        self.dX = dX
        return dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _BatchNorm2D(Layer):
    def __init__(
        self,
        in_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.momentum = momentum
        self.epsilon = epsilon

        gamma_ = np.ones((1, in_features, 1, 1))
        beta_ = np.zeros((1, in_features, 1, 1))
        self.weights_ = [gamma_, beta_]

        self.running_mean = np.zeros((1, in_features, 1, 1))
        self.running_var = np.ones((1, in_features, 1, 1))

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            self.X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            self.X_norm = (X - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )

        out = self.weights_[0] * self.X_norm + self.weights_[1]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        batch_size, _, height, width = d_out.shape
        dX_norm = d_out * self.weights_[0]

        dgamma = np.sum(d_out * self.X_norm, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(d_out, axis=(0, 2, 3), keepdims=True)
        self.dW = [dgamma, dbeta]

        dX = (
            (1.0 / (batch_size * height * width))
            * np.reciprocal(np.sqrt(self.running_var + self.epsilon))
            * (
                (batch_size * height * width) * dX_norm
                - np.sum(dX_norm, axis=(0, 2, 3), keepdims=True)
                - self.X_norm
                * np.sum(dX_norm * self.X_norm, axis=(0, 2, 3), keepdims=True)
            )
        )
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _BatchNorm3D(Layer):
    def __init__(
        self,
        in_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, in_features, 1, 1, 1))
        self.beta = np.zeros((1, in_features, 1, 1, 1))
        self.weights_ = [self.gamma, self.beta]

        self.running_mean = np.zeros((1, in_features, 1, 1, 1))
        self.running_var = np.ones((1, in_features, 1, 1, 1))

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            batch_mean = np.mean(X, axis=(0, 2, 3, 4), keepdims=True)
            batch_var = np.var(X, axis=(0, 2, 3, 4), keepdims=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            self.X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            self.X_norm = (X - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )

        out = self.weights_[0] * self.X_norm + self.weights_[1]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        batch_size, _, depth, height, width = d_out.shape

        dX_norm = d_out * self.weights_[0]
        dgamma = np.sum(d_out * self.X_norm, axis=(0, 2, 3, 4), keepdims=True)
        dbeta = np.sum(d_out, axis=(0, 2, 3, 4), keepdims=True)
        self.dW = [dgamma, dbeta]

        dX = (
            1.0
            / (batch_size * depth * height * width)
            * np.reciprocal(np.sqrt(self.running_var + self.epsilon))
            * (
                batch_size * depth * height * width * dX_norm
                - np.sum(dX_norm, axis=(0, 2, 3, 4), keepdims=True)
                - self.X_norm
                * np.sum(dX_norm * self.X_norm, axis=(0, 2, 3, 4), keepdims=True)
            )
        )
        self.dX = dX
        return dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _LRN_1D(Layer): ...


class _LRN_2D(Layer): ...


class _LRN_3D(Layer): ...
