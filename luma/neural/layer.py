from typing import Any, List, Literal, Self, Tuple, Type, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Matrix, Tensor, TensorLike, ClassType
from luma.interface.util import InitUtil, Clone
from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Layer


__all__ = (
    "Convolution1D",
    "Convolution2D",
    "Convolution3D",
    "Pooling1D",
    "Pooling2D",
    "Pooling3D",
    "Dense",
    "Dropout",
    "Flatten",
    "Activation",
    "BatchNorm1D",
    "BatchNorm2D",
    "BatchNorm3D",
    "Sequential",
)


class Convolution1D(Layer):
    """
    Convolutional layer for 1-dimensional data.

    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : Number of input channels
    `out_channels` : Number of output channels (filters)
    `filter_size`: Length of each filter
    `stride` : Step size for filters during convolution
    `padding` : Padding strategies
    (`valid` for no padding, `same` for zero-padding)
    `initializer` : Type of weight initializer (default `None`)
    `optimizer` : Optimizer for weight update (default `SGDOptimizer`)
    `lambda_` : L2-regularization strength
    `random_state` : Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, width = X.shape

        if self.in_channels != channels:
            raise ValueError(
                f"channels of 'X' does not match with 'in_channels'! "
                + f"({self.in_channels}!={channels})"
            )

        pad_w, padded_w = self._get_padding_dim(width)

        out_width = ((padded_w - self.filter_size) // self.stride) + 1
        out: Tensor = np.zeros((batch_size, self.out_channels, out_width))

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")
        X_fft = np.fft.rfft(X_padded, n=padded_w, axis=2)
        filter_fft = np.fft.rfft(
            self.weights_,
            n=padded_w,
            axis=2,
        )

        for i in range(batch_size):
            for f in range(self.out_channels):
                result_fft = np.sum(X_fft[i] * filter_fft[f], axis=0)
                result = np.fft.irfft(result_fft, n=padded_w)

                out[i, f] = result[pad_w : padded_w - pad_w : self.stride][:out_width]

        out += self.biases_[:, :, np.newaxis]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        batch_size, channels, width = X.shape
        pad_w, padded_w = self._get_padding_dim(width)

        dX_padded = np.zeros((batch_size, channels, padded_w))
        self.dW = np.zeros_like(self.weights_)
        self.dB = np.zeros_like(self.biases_)

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")
        X_fft = np.fft.rfft(X_padded, n=padded_w, axis=2)
        d_out_fft = np.fft.rfft(d_out, n=padded_w, axis=2)

        for f in range(self.out_channels):
            self.dB[:, f] = np.sum(d_out[:, f, :])

        for f in range(self.out_channels):
            for c in range(channels):
                filter_d_out_fft = np.sum(
                    X_fft[:, c] * d_out_fft[:, f].conj(),
                    axis=0,
                )
                self.dW[f, c] = np.fft.irfft(filter_d_out_fft, n=padded_w)[
                    : self.filter_size
                ]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                temp = np.zeros(padded_w // 2 + 1, dtype=np.complex128)
                for f in range(self.out_channels):
                    filter_fft = np.fft.rfft(self.weights_[f, c], n=padded_w)
                    temp += filter_fft * d_out_fft[i, f]
                dX_padded[i, c] = np.fft.irfft(temp, n=padded_w)

        self.dX = dX_padded[:, :, pad_w:-pad_w] if pad_w > 0 else dX_padded
        return self.dX

    def _get_padding_dim(self, width: int) -> Tuple[int, ...]:
        if self.padding == "same":
            pad_w = (self.filter_size - 1) // 2
            padded_w = width + 2 * pad_w

        elif self.padding == "valid":
            pad_w = 0
            padded_w = width
        else:
            raise UnsupportedParameterError(self.padding)

        return pad_w, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, width = in_shape
        _, padded_w = self._get_padding_dim(width)
        out_width = ((padded_w - self.filter_size) // self.stride) + 1

        return (batch_size, self.out_channels, out_width)


class Convolution2D(Layer):
    """
    Convolutional layer for 2-dimensional data.

    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : Number of input channels
    `out_channels` : Number of output channels(filters)
    `filter_size`: Size of each filter
    `stride` : Step size for filters during convolution
    `padding` : Padding stratagies
    (`valid` for no padding, `same` for typical 0-padding)
    `initializer` : Type of weight initializer (default `None`)
    `optimizer` : Optimizer for weight update (default `SGDOptimizer`)
    `lambda_` : L2-regularization strength
    `random_state` : Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                self.filter_size,
                self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, height, width = X.shape

        if self.in_channels != channels:
            raise ValueError(
                f"channels of 'X' does not match with 'in_channels'! "
                + f"({self.in_channels}!={channels})"
            )

        pad_h, pad_w, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = ((padded_h - self.filter_size) // self.stride) + 1
        out_width = ((padded_w - self.filter_size) // self.stride) + 1
        out: Tensor = np.zeros(
            (
                batch_size,
                self.out_channels,
                out_height,
                out_width,
            )
        )

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_h, padded_w), axes=[2, 3])
        filter_fft = np.fft.rfftn(
            self.weights_,
            s=(padded_h, padded_w),
            axes=[2, 3],
        )
        for i in range(batch_size):
            for f in range(self.out_channels):
                result_fft = np.sum(X_fft[i] * filter_fft[f], axis=0)
                result = np.fft.irfftn(result_fft, s=(padded_h, padded_w))

                sampled_result = result[
                    pad_h : padded_h - pad_h : self.stride,
                    pad_w : padded_w - pad_w : self.stride,
                ]
                out[i, f] = sampled_result[:out_height, :out_width]

        out += self.biases_[:, :, np.newaxis, np.newaxis]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        batch_size, channels, height, width = X.shape
        pad_h, pad_w, padded_h, padded_w = self._get_padding_dim(height, width)

        dX_padded = np.zeros((batch_size, channels, padded_h, padded_w))
        self.dW = np.zeros_like(self.weights_)
        self.dB = np.zeros_like(self.biases_)

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_h, padded_w), axes=[2, 3])
        d_out_fft = np.fft.rfftn(d_out, s=(padded_h, padded_w), axes=[2, 3])

        for f in range(self.out_channels):
            self.dB[:, f] = np.sum(d_out[:, f, :, :])

        for f in range(self.out_channels):
            for c in range(channels):
                filter_d_out_fft = np.sum(
                    X_fft[:, c] * d_out_fft[:, f].conj(),
                    axis=0,
                )
                self.dW[f, c] = np.fft.irfftn(
                    filter_d_out_fft,
                    s=(padded_h, padded_w),
                )[pad_h : pad_h + self.filter_size, pad_w : pad_w + self.filter_size]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                temp = np.zeros((padded_h, padded_w // 2 + 1), dtype=np.complex128)
                for f in range(self.out_channels):
                    filter_fft = np.fft.rfftn(
                        self.weights_[f, c], s=(padded_h, padded_w)
                    )
                    temp += filter_fft * d_out_fft[i, f]
                dX_padded[i, c] = np.fft.irfftn(temp, s=(padded_h, padded_w))

        self.dX = (
            dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
            if pad_h > 0 or pad_w > 0
            else dX_padded
        )
        return self.dX

    def _get_padding_dim(self, height: int, width: int) -> Tuple[int, ...]:
        if self.padding == "same":
            pad_h = pad_w = (self.filter_size - 1) // 2
            padded_h = height + 2 * pad_h
            padded_w = width + 2 * pad_w

        elif self.padding == "valid":
            pad_h = pad_w = 0
            padded_h = height
            padded_w = width
        else:
            raise UnsupportedParameterError(self.padding)

        return pad_h, pad_w, padded_h, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        _, _, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = ((padded_h - self.filter_size) // self.stride) + 1
        out_width = ((padded_w - self.filter_size) // self.stride) + 1

        return (batch_size, self.out_channels, out_height, out_width)


class Convolution3D(Layer):
    """
    Convolutional layer for 3-dimensional data.

    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : Number of input channels
    `out_channels` : Number of output channels(filters)
    `filter_size`: Size of each filter
    `stride` : Step size for filters during convolution
    `padding` : Padding stratagies
    (`valid` for no padding, `same` for typical 0-padding)
    `initializer` : Type of weight initializer (default `None`)
    `optimizer` : Optimizer for weight update (default `SGDOptimizer`)
    `lambda_` : L2-regularization strength
    `random_state` : Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                self.filter_size,
                self.filter_size,
                self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, depth, height, width = X.shape

        if self.in_channels != channels:
            raise ValueError(
                f"channels of 'X' does not match with 'in_channels'! "
                + f"({self.in_channels}!={channels})"
            )

        pad_d, pad_h, pad_w, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )
        out_depth = ((padded_d - self.filter_size) // self.stride) + 1
        out_height = ((padded_h - self.filter_size) // self.stride) + 1
        out_width = ((padded_w - self.filter_size) // self.stride) + 1
        out: Tensor = np.zeros(
            (batch_size, self.out_channels, out_depth, out_height, out_width)
        )

        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        X_fft = np.fft.rfftn(
            X_padded,
            s=(padded_d, padded_h, padded_w),
            axes=[2, 3, 4],
        )
        filter_fft = np.fft.rfftn(
            self.weights_,
            s=(padded_d, padded_h, padded_w),
            axes=[2, 3, 4],
        )

        for i in range(batch_size):
            for f in range(self.out_channels):
                result_fft = np.sum(X_fft[i] * filter_fft[f], axis=0)
                result = np.fft.irfftn(result_fft, s=(padded_d, padded_h, padded_w))

                sampled_result = result[
                    pad_d : padded_d - pad_d : self.stride,
                    pad_h : padded_h - pad_h : self.stride,
                    pad_w : padded_w - pad_w : self.stride,
                ]
                out[i, f] = sampled_result[:out_depth, :out_height, :out_width]

        out += self.biases_[:, :, np.newaxis, np.newaxis, np.newaxis]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        batch_size, channels, depth, height, width = X.shape
        pad_d, pad_h, pad_w, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )

        dX_padded = np.zeros((batch_size, channels, padded_d, padded_h, padded_w))
        self.dW = np.zeros_like(self.weights_)
        self.dB = np.zeros_like(self.biases_)

        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        X_fft = np.fft.rfftn(
            X_padded,
            s=(padded_d, padded_h, padded_w),
            axes=[2, 3, 4],
        )
        d_out_fft = np.fft.rfftn(
            d_out,
            s=(padded_d, padded_h, padded_w),
            axes=[2, 3, 4],
        )

        for f in range(self.out_channels):
            self.dB[:, f] = np.sum(d_out[:, f, :, :, :])

        for f in range(self.out_channels):
            for c in range(channels):
                filter_d_out_fft = np.sum(
                    X_fft[:, c] * d_out_fft[:, f].conj(),
                    axis=0,
                )
                self.dW[f, c] = np.fft.irfftn(
                    filter_d_out_fft,
                    s=(padded_d, padded_h, padded_w),
                )[
                    pad_d : pad_d + self.filter_size,
                    pad_h : pad_h + self.filter_size,
                    pad_w : pad_w + self.filter_size,
                ]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                temp = np.zeros(
                    (padded_d, padded_h, padded_w // 2 + 1), dtype=np.complex128
                )
                for f in range(self.out_channels):
                    filter_fft = np.fft.rfftn(
                        self.weights_[f, c],
                        s=(padded_d, padded_h, padded_w),
                        axes=[2, 3, 4],
                    )
                    temp += filter_fft * d_out_fft[i, f]
                dX_padded[i, c] = np.fft.irfftn(
                    temp,
                    s=(padded_d, padded_h, padded_w),
                )

        self.dX = (
            dX_padded[:, :, pad_d:-pad_d, pad_h:-pad_h, pad_w:-pad_w]
            if (pad_d > 0 or pad_h > 0 or pad_w > 0)
            else dX_padded
        )
        return self.dX

    def _get_padding_dim(
        self,
        depth: int,
        height: int,
        width: int,
    ) -> Tuple[int, ...]:
        if self.padding == "same":
            pad_d = pad_h = pad_w = (self.filter_size - 1) // 2
            padded_d = depth + 2 * pad_d
            padded_h = height + 2 * pad_h
            padded_w = width + 2 * pad_w

        elif self.padding == "valid":
            pad_d = pad_h = pad_w = 0
            padded_d = depth
            padded_h = height
            padded_w = width
        else:
            raise UnsupportedParameterError(self.padding)

        return pad_d, pad_h, pad_w, padded_d, padded_h, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, depth, height, width = in_shape
        _, _, _, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )
        out_depth = ((padded_d - self.filter_size) // self.stride) + 1
        out_height = ((padded_h - self.filter_size) // self.stride) + 1
        out_width = ((padded_w - self.filter_size) // self.stride) + 1

        return (batch_size, self.out_channels, out_depth, out_height, out_width)


class Pooling1D(Layer):
    """
    Pooling layer for 1-dimensional data.

    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `filter_size` : Size of the pooling filter
    `stride` : Step size of the filter during pooling
    `mode` : Pooling strategy (i.e., 'max' or 'avg')

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, width = X.shape

        out_width = 1 + (width - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_width))

        for i in range(out_width):
            w_start, w_end = self._get_pooling_bounds(i)
            window = X[:, :, w_start:w_end]

            if self.mode == "max":
                out[:, :, i] = np.max(window, axis=2)
            elif self.mode == "avg":
                out[:, :, i] = np.mean(window, axis=2)
            else:
                raise UnsupportedParameterError(self.mode)

        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, out_width = d_out.shape
        self.dX = np.zeros_like(X)

        for i in range(out_width):
            w_start, w_end = self._get_pooling_bounds(i)
            window = X[:, :, w_start:w_end]

            if self.mode == "max":
                max_vals = np.max(window, axis=2, keepdims=True)
                mask_ = window == max_vals
                self.dX[:, :, w_start:w_end] += mask_ * d_out[:, :, i : i + 1]
            elif self.mode == "avg":
                self.dX[:, :, w_start:w_end] += (
                    d_out[:, :, i : i + 1] / self.filter_size
                )

        return self.dX

    def _get_pooling_bounds(self, cur_w: int) -> Tuple[int, int]:
        w_start = cur_w * self.stride
        w_end = w_start + self.filter_size

        return w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, width = in_shape
        out_width = 1 + (width - self.filter_size) // self.stride

        return (batch_size, channels, out_width)


class Pooling2D(Layer):
    """
    Pooling layer for 2-dimensional data.

    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `size` : Size of pooling filter
    `stride` : Step size of filter during pooling
    `mode` : Pooling strategy (i.e. max, average)

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, height, width = X.shape

        out_height = 1 + (height - self.filter_size) // self.stride
        out_width = 1 + (width - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_pooling_bounds(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    out[:, :, i, j] = np.max(window, axis=(2, 3))
                elif self.mode == "avg":
                    out[:, :, i, j] = np.mean(window, axis=(2, 3))
                else:
                    raise UnsupportedParameterError(self.mode)

        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, out_height, out_width = d_out.shape
        self.dX = np.zeros_like(X)

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_pooling_bounds(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    max_vals = np.max(window, axis=(2, 3), keepdims=True)
                    mask_ = window == max_vals
                    self.dX[:, :, h_start:h_end, w_start:w_end] += (
                        mask_ * d_out[:, :, i : i + 1, j : j + 1]
                    )
                elif self.mode == "avg":
                    self.dX[:, :, h_start:h_end, w_start:w_end] += d_out[
                        :, :, i : i + 1, j : j + 1
                    ] / (self.filter_size**2)

        return self.dX

    def _get_pooling_bounds(self, cur_h: int, cur_w: int) -> Tuple[int, ...]:
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride

        h_end = h_start + self.filter_size
        w_end = w_start + self.filter_size

        return h_start, h_end, w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, height, width = in_shape
        out_height = 1 + (height - self.filter_size) // self.stride
        out_width = 1 + (width - self.filter_size) // self.stride

        return (batch_size, channels, out_height, out_width)


class Pooling3D(Layer):
    """
    Pooling layer for 3-dimensional data.

    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `filter_size` : Size of the pooling filter (cubic)
    `stride` : Step size of the filter during pooling
    `mode` : Pooling strategy (i.e., 'max' or 'avg')

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, depth, height, width = X.shape

        out_depth = 1 + (depth - self.filter_size) // self.stride
        out_height = 1 + (height - self.filter_size) // self.stride
        out_width = 1 + (width - self.filter_size) // self.stride
        out: Tensor = np.zeros(
            (
                batch_size,
                channels,
                out_depth,
                out_height,
                out_width,
            )
        )

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start, d_end, h_start, h_end, w_start, w_end = (
                        self._get_pooling_bounds(i, j, k)
                    )
                    window = X[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                    if self.mode == "max":
                        out[:, :, i, j, k] = np.max(window, axis=(2, 3, 4))
                    elif self.mode == "avg":
                        out[:, :, i, j, k] = np.mean(window, axis=(2, 3, 4))
                    else:
                        raise UnsupportedParameterError(self.mode)

        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, out_depth, out_height, out_width = d_out.shape
        self.dX = np.zeros_like(X)

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start, d_end, h_start, h_end, w_start, w_end = (
                        self._get_pooling_bounds(i, j, k)
                    )
                    window = X[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                    if self.mode == "max":
                        max_vals = np.max(window, axis=(2, 3, 4), keepdims=True)
                        mask_ = window == max_vals
                        self.dX[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += (
                            mask_ * d_out[:, :, i : i + 1, j : j + 1, k : k + 1]
                        )
                    elif self.mode == "avg":
                        self.dX[
                            :, :, d_start:d_end, h_start:h_end, w_start:w_end
                        ] += d_out[:, :, i : i + 1, j : j + 1, k : k + 1] / (
                            self.filter_size**3
                        )

        return self.dX

    def _get_pooling_bounds(
        self, cur_d: int, cur_h: int, cur_w: int
    ) -> Tuple[int, ...]:
        d_start = cur_d * self.stride
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride

        d_end = d_start + self.filter_size
        h_end = h_start + self.filter_size
        w_end = w_start + self.filter_size

        return d_start, d_end, h_start, h_end, w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, depth, height, width = in_shape

        out_depth = 1 + (depth - self.filter_size) // self.stride
        out_height = 1 + (height - self.filter_size) // self.stride
        out_width = 1 + (width - self.filter_size) // self.stride

        return (batch_size, channels, out_depth, out_height, out_width)


class Dense(Layer):
    """
    A dense layer, also known as a fully connected layer, connects each
    neuron in one layer to every neuron in the next layer. It performs a
    linear transformation followed by a nonlinear activation function,
    enabling complex relationships between input and output. Dense layers
    are fundamental in deep learning models for learning representations from
    data. They play a crucial role in capturing intricate patterns and
    features during the training process.

    Parameters
    ----------
    `in_features` : Number of input features
    `out_features`:  Number of output features
    `initializer` : Type of weight initializer (default `None`)
    `optimizer` : Optimizer for weight update
    `lambda_` : L2-regularization strength
    `random_state` : Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 2D-array(`Matrix`).

        ```py
        X.shape = (batch_size, n_features)
        ```
    """

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


class Dropout(Layer):
    """
    Dropout is a regularization technique used during training to prevent
    overfitting by randomly setting a fraction of input units to zero during
    the forward pass. This helps in reducing co-adaptation of neurons and
    encourages the network to learn more robust features.

    Parameters
    ----------
    `dropout_rate` : The fraction of input units to drop during training
    `random_state` : Seed for various random sampling processes

    Notes
    -----
    - During inference, dropout is typically turned off, and the layer behaves
      as the identity function.

    """

    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        else:
            return X

    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class Flatten(Layer):
    """
    A flatten layer reshapes the input tensor into a 2D array(`Matrix`),
    collapsing all dimensions except the batch dimension.

    Notes
    -----
    - Use this class when using `Dense` layer.
        Flatten the tensor into matrix in order to feed-forward dense layer(s).
    """

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
        return (batch_size, np.prod(shape))


@ClassType.non_instantiable()
class Activation:
    """
    An Activation Layer in a neural network applies a specific activation
    function to the input it receives, transforming the input to activate
    or deactivate neurons within the network. This function can be linear
    or non-linear, such as Sigmoid, ReLU, or Tanh, which helps to introduce
    non-linearity into the model, allowing it to learn complex patterns.

    Notes
    -----
    - This class is not instantiable, meaning that a solitary use of it
        is not available.

    - All the activation functions are included inside `Activation`.

    Examples
    --------
    >>> # Activation() <- Impossible
    >>> Activation.Linear()
    >>> Activation.ReLU()

    """

    type FuncType = Type

    @classmethod
    def _out_shape(cls, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape

    class Linear(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            return X

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class ReLU(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            return np.maximum(0, X)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out.copy()
            self.dX[self.input_ <= 0] = 0
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class Sigmoid(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.output_ = 1 / (1 + np.exp(-X))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * self.output_ * (1 - self.output_)
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class Tanh(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.output_ = np.tanh(X)
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (1 - np.square(self.output_))
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class LeakyReLU(Layer):
        def __init__(self, alpha: float = 0.01) -> None:
            super().__init__()
            self.alpha = alpha

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            return np.where(X > 0, X, X * self.alpha)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * np.where(self.input_ > 0, 1, self.alpha)
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class Softmax(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            e_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
            return e_X / np.sum(e_X, axis=-1, keepdims=True)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = np.empty_like(d_out)
            for i, (y, dy) in enumerate(zip(self.output_, d_out)):
                y = y.reshape(-1, 1)
                jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)

                self.dX[i] = np.dot(jacobian_matrix, dy)

            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class ELU(Layer):
        def __init__(self, alpha: float = 1.0) -> None:
            super().__init__()
            self.alpha = alpha

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            self.output_ = np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * np.where(
                self.input_ > 0,
                1,
                self.output_ + self.alpha,
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class SELU(Layer):
        def __init__(
            self,
            lambda_: float = 1.0507,
            alpha: float = 1.67326,
        ) -> None:
            super().__init__()
            self.lambda_ = lambda_
            self.alpha = alpha

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            self.output_ = self.lambda_ * np.where(
                X > 0, X, self.alpha * (np.exp(X) - 1)
            )
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = (
                d_out
                * self.lambda_
                * np.where(
                    self.input_ > 0,
                    1,
                    self.alpha * np.exp(self.input_),
                )
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class Softplus(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.output_ = np.log1p(np.exp(X))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (1 - 1 / (1 + np.exp(self.output_)))
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)

    class Swish(Layer):
        def __init__(self, beta: float = 1.0) -> None:
            super().__init__()
            self.beta = beta

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            self.sigmoid = 1 / (1 + np.exp(-self.beta * X))

            self.output_ = X * self.sigmoid
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (
                self.sigmoid
                + self.beta * self.input_ * self.sigmoid * (1 - self.sigmoid)
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return Activation._out_shape(in_shape)


class BatchNorm1D(Layer):
    """
    Batch normalization layer for 1-dimensional data.

    Batch normalization standardizes layer inputs across mini-batches to stabilize
    learning, accelerate convergence, and reduce sensitivity to initialization.
    It adjusts normalized outputs using learnable parameters, mitigating internal
    covariate shift in deep networks.

    Parameters
    ----------
    `in_features` : Number of input features
    `momentum` : Momentum for updating the running averages

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

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


class BatchNorm2D(Layer):
    """
    Batch normalization layer for 2-dimensional data.

    Batch normalization standardizes layer inputs across mini-batches to stabilize
    learning, accelerate convergence, and reduce sensitivity to initialization.
    It adjusts normalized outputs using learnable parameters, mitigating internal
    covariate shift in deep networks.

    Parameters
    ----------
    `in_features` : Number of input features
    `momentum` : Momentum for updating the running averages

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

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


class BatchNorm3D(Layer):
    """
    Batch normalization layer for 3-dimensional data.

    Batch normalization standardizes layer inputs across mini-batches to stabilize
    learning, accelerate convergence, and reduce sensitivity to initialization.
    It adjusts normalized outputs using learnable parameters, mitigating internal
    covariate shift in deep networks.

    Parameters
    ----------
    `in_features` : Number of input features
    `momentum` : Momentum for updating the running averages

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

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


class Sequential(Layer):
    """
    Sequential represents a linear arrangement of layers in a neural network
    model. Each layer is added sequentially, with data flowing from one layer
    to the next in the order they are added. This organization simplifies the
    construction of neural networks, especially for straightforward architectures,
    by mirroring the logical flow of data from input through hidden layers to
    output.

    Parameters
    ----------
    `*layers` : Layers or layers with its name assigned
    (class name of the layer assigned by default)

    Methods
    -------
    For setting an optimizer of each layer:
    ```py
    def set_optimizer(self, optimizer: Optimizer) -> None
    ```
    To add additional layer:
    ```py
    def add(self, layer: Layer) -> None
    ```
    Specials
    --------
    - You can use `+` operator to add a layer or another instance
        of `Sequential`.

    - Calling its instance performs a single forwarding.

    Notes
    -----
    - Before any execution, an optimizer must be assigned.

    - For multi-class classification, the target variable `y`
        must be one-hot encoded.

    Examples
    --------
    ```py
    model = Sequential(
        ("conv_1", Convolution(3, 6, 3, activation="relu")),
        ("pool_1", Pooling(2, 2, mode="max")),
        ...,
        ("drop", Dropout(0.1)),
        ("flat", Flatten()),
        ("dense_1", Dense(384, 32, activation="relu")),
        ("dense_2", Dense(32, 10, activation="softmax")),
    )
    model.set_optimizer(AnyOptimizer())

    out = model(X, is_train=True) # model.forward(X, is_train=True)
    model.backward(d_out) # assume d_out is the gradient w.r.t. loss
    model.update()
    ```
    """

    def __init__(self, *layers: Layer | tuple[str, Layer]) -> None:
        super().__init__()
        self.layers: List[tuple[str, Layer]] = list()
        for layer in layers:
            self.add(layer)

    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        self.input_ = X
        out = X
        for _, layer in self.layers:
            out = layer(out, is_train=is_train)

        return out

    def backward(self, d_out: TensorLike) -> TensorLike:
        for _, layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def update(self) -> None:
        self._check_no_optimizer()
        for _, layer in reversed(self.layers):
            layer.update()

    def set_optimizer(self, optimizer: Optimizer, **params: Any) -> None:
        self.optimizer = optimizer
        self.optimizer.set_params(**params, ignore_missing=True)

        for _, layer in self.layers:
            cloned_opt = Clone(self.optimizer).get
            if hasattr(layer, "set_optimizer"):
                layer.set_optimizer(cloned_opt)
            else:
                layer.optimizer = cloned_opt

    def _check_no_optimizer(self) -> None:
        if self.optimizer is None:
            raise RuntimeError(
                f"'{self}' has no optimizer! "
                + f"Call '{self}().set_optimizer' to assign an optimizer."
            )

    def add(self, layer: Layer | Tuple[str, Layer]) -> None:
        if not isinstance(layer, tuple):
            layer = (str(layer), layer)
        self.layers.append(layer)

        if self.optimizer is not None:
            self.set_optimizer(self.optimizer)

    @override
    @property
    def param_size(self) -> Tuple[int, int]:
        w_size, b_size = 0, 0
        for _, layer in self.layers:
            w_, b_ = layer.param_size
            w_size += w_
            b_size += b_

        return w_size, b_size

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        for _, layer in self.layers:
            in_shape = layer.out_shape(in_shape)
        return in_shape

    def __add__(self, other: Layer | Self) -> Self:
        if isinstance(other, Layer):
            self.add(other)
        elif isinstance(other, Self):
            for layer in other.layers:
                self.add(layer)
        else:
            raise TypeError(
                "Unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self).__name__, type(other).__name__
                )
            )

        return self

    def __getitem__(self, index: int) -> Tuple[str, Layer]:
        return self.layers[index]
