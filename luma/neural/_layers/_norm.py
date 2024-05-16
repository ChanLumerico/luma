from typing import Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.neural.base import Layer


__all__ = (
    "_BatchNorm1D",
    "_BatchNorm2D",
    "_BatchNorm3D",
    "_LocalResponseNorm",
    "_LayerNorm",
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

    @Tensor.force_dim(3)
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

    @Tensor.force_dim(3)
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

    @Tensor.force_dim(4)
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

    @Tensor.force_dim(4)
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

    @Tensor.force_dim(5)
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

    @Tensor.force_dim(5)
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


class _LocalResponseNorm(Layer):
    def __init__(
        self,
        depth: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 2.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X

        channels = X.shape[1]
        sq_input = np.square(X)
        extra_dims = (slice(None),) * (X.ndim - 2)

        scale = self.k + self.alpha * np.array(
            [
                np.sum(
                    sq_input[
                        :,
                        max(0, j - self.depth // 2) : min(
                            channels, j + self.depth // 2 + 1
                        ),
                        *extra_dims,
                    ],
                    axis=1,
                )
                for j in range(channels)
            ]
        ).transpose((1, 0) + tuple(range(2, X.ndim)))

        self.scale = np.power(scale, self.beta)
        return X / self.scale

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        channels = X.shape[1]

        dX = np.zeros_like(X)
        extra_dims = (slice(None),) * (X.ndim - 2)

        dtmp = -self.beta * np.power(self.scale, -1) * d_out / self.scale
        for i in range(channels):
            dsum = (
                np.sum(
                    dtmp[
                        :,
                        max(0, i - self.depth // 2) : min(
                            channels, i + self.depth // 2 + 1
                        ),
                        *extra_dims,
                    ],
                    axis=1,
                    keepdims=True,
                )
                * 2
                * self.alpha
                * X[:, i, *extra_dims]
            )
            dX[:, i, *extra_dims] = (
                d_out[:, i, *extra_dims] / self.scale[:, i, *extra_dims] + dsum
            )
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _LayerNorm(Layer):
    def __init__(self, in_shape: tuple[int] | int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.in_shape = in_shape
        if isinstance(self.in_shape, int):
            self.in_shape = (self.in_shape,)
        self.epsilon = epsilon

        gamma = np.ones(in_shape[1:])
        beta = np.zeros(in_shape[1:])
        self.weights_ = [gamma, beta]

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        X_axis = tuple(range(1, X.ndim))

        mean = X.mean(axis=X_axis, keepdims=True)
        var = X.var(axis=X_axis, keepdims=True)

        self.X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        out = self.weights_[0] * self.X_norm + self.weights_[1]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        X_axis = tuple(range(1, X.ndim))
        n_features = np.prod(X.shape[1:])

        X_mu = X - X.mean(axis=X_axis, keepdims=True)
        std_inv = 1.0 / np.sqrt(X.var(axis=X_axis, keepdims=True) + self.epsilon)

        dgamma = np.sum(d_out * self.X_norm, axis=0)
        dbeta = np.sum(d_out, axis=0)
        self.dW = [dgamma, dbeta]

        dX_norm = d_out * self.weights_[0]
        dvar = np.sum(-0.5 * dX_norm * X_mu * std_inv**3, axis=X_axis, keepdims=True)
        dmu = np.sum(dX_norm * -std_inv, axis=X_axis, keepdims=True) + dvar * np.mean(
            -2 * X_mu, axis=X_axis, keepdims=True
        )

        dX = dX_norm * std_inv + (dvar * 2 * X_mu + dmu) / n_features
        self.dX = dX
        return self.dX
