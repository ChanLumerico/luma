from typing import Literal, Tuple
import numpy as np

from luma.core.super import Optimizer
from luma.interface.util import Layer, Vector, Matrix, Tensor, ActivationUtil
from luma.interface.exception import UnsupportedParameterError
from luma.neural.optimizer import SGDOptimizer


__all__ = ("Convolution", "Pooling", "Dense")


class Convolution(Layer):
    """
    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `n_filters` : Number of filters(kernels) to use
    `size`: Size of each filter
    `stride` : Step size for filters during convolution
    `padding` : Padding stratagies
    (`valid` for no padding, `same` for typical 0-padding)
    `activation` : Type of activation function
    `optimizer` : Optimizer for weight update (default `SGDOptimizer`)

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self,
        n_filters: int,
        size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        activation: ActivationUtil.FuncType = "relu",
        optimizer: Optimizer = SGDOptimizer(),
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.size = size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.optimizer = optimizer

        act = ActivationUtil(self.activation)
        self.act_ = act.activation_type()
        self.rs_ = np.random.RandomState(random_state)

        self.biases_: Vector = np.zeros(self.n_filters)

    def forward(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 4, "X must have the form of 4D-array!"
        batch_size, channels, height, width = X.shape

        if self.weights_ is None:
            self.weights_ = 0.01 * self.rs_.randn(
                self.n_filters, channels, self.size, self.size
            )

        pad_h, pad_w, padded_height, padded_width = self._get_padding_dim(height, width)

        out_height = ((padded_height - self.size) // self.stride) + 1
        out_width = ((padded_width - self.size) // self.stride) + 1
        out: Tensor = np.zeros((batch_size, self.n_filters, out_height, out_width))

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_height, padded_width), axes=[2, 3])
        filter_fft = np.fft.rfftn(
            self.weights_, s=(padded_height, padded_width), axes=[2, 3]
        )

        for i in range(batch_size):
            for f in range(self.n_filters):
                result_fft = np.sum(X_fft[i] * filter_fft[f], axis=0)
                result = np.fft.irfftn(result_fft, s=(padded_height, padded_width))

                sampled_result = result[
                    pad_h : padded_height - pad_h : self.stride,
                    pad_w : padded_width - pad_w : self.stride,
                ]
                out[i, f] = sampled_result[:out_height, :out_width]

        out += self.biases_[:, np.newaxis, np.newaxis]
        out = self.act_.func(out)
        return out

    def backward(self, X: Tensor, d_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, channels, height, width = X.shape
        pad_h, pad_w, padded_height, padded_width = self._get_padding_dim(height, width)

        dX_padded = np.zeros((batch_size, channels, padded_height, padded_width))
        self.dW = np.zeros_like(self.weights_)
        self.dB = np.zeros_like(self.biases_)

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_height, padded_width), axes=[2, 3])
        d_out_fft = np.fft.rfftn(d_out, s=(padded_height, padded_width), axes=[2, 3])

        for f in range(self.n_filters):
            self.dB[f] = np.sum(d_out[:, f, :, :])

        for f in range(self.n_filters):
            for c in range(channels):
                filter_d_out_fft = np.sum(X_fft[:, c] * d_out_fft[:, f].conj(), axis=0)
                self.dW[f, c] = np.fft.irfftn(
                    filter_d_out_fft, s=(padded_height, padded_width)
                )[pad_h : pad_h + self.size, pad_w : pad_w + self.size]

        for i in range(batch_size):
            for c in range(channels):
                temp = np.zeros(
                    (padded_height, padded_width // 2 + 1), dtype=np.complex128
                )
                for f in range(self.n_filters):
                    filter_fft = np.fft.rfftn(
                        self.weights_[f, c], s=(padded_height, padded_width)
                    )
                    temp += filter_fft * d_out_fft[i, f]
                dX_padded[i, c] = np.fft.irfftn(temp, s=(padded_height, padded_width))

        self.dX = (
            dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
            if pad_h > 0 or pad_w > 0
            else dX_padded
        )
        self.dX = self.act_.derivative(self.dX)
        return self.dX

    def _get_padding_dim(self, height: int, width: int) -> Tuple[int, int, int, int]:
        if self.padding == "same":
            pad_h = pad_w = (self.size - 1) // 2
            padded_height = height + 2 * pad_h
            padded_width = width + 2 * pad_w
        elif self.padding == "valid":
            pad_h = pad_w = 0
            padded_height = height
            padded_width = width
        else:
            raise UnsupportedParameterError(self.padding)

        return pad_h, pad_w, padded_height, padded_width


class Pooling(Layer):
    """
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
        self, size: int = 2, stride: int = 2, mode: Literal["max", "avg"] = "max"
    ) -> None:
        super().__init__()
        self.size = size
        self.stride = stride
        self.mode = mode

    def forward(self, X: Tensor) -> Tensor:
        batch_size, channels, height, width = X.shape
        out_height = 1 + (height - self.size) // self.stride
        out_width = 1 + (width - self.size) // self.stride

        out: Tensor = np.zeros((batch_size, channels, out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_height_width(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    out[:, :, i, j] = np.max(window, axis=(2, 3))
                elif self.mode == "avg":
                    out[:, :, i, j] = np.mean(window, axis=(2, 3))
                else:
                    raise UnsupportedParameterError(self.mode)

        return out

    def backward(self, X: Tensor, d_out: Tensor) -> Tensor:
        _, _, out_height, out_width = d_out.shape
        self.dX = np.zeros_like(X)

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_height_width(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    max_vals = np.max(window, axis=(2, 3), keepdims=True)
                    mask = window == max_vals
                    self.dX[:, :, h_start:h_end, w_start:w_end] += (
                        mask * d_out[:, :, i : i + 1, j : j + 1]
                    )
                elif self.mode == "avg":
                    avg_grad = d_out[:, :, i, j] / (self.size**2)
                    self.dX[:, :, h_start:h_end, w_start:w_end] += (
                        np.ones((1, 1, self.size, self.size)) * avg_grad
                    )

        return self.dX

    def _get_height_width(self, cur_h: int, cur_w: int) -> Tuple[int, int, int, int]:
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride
        h_end = h_start + self.size
        w_end = w_start + self.size

        return h_start, h_end, w_start, w_end


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
    - `input_size` : Number of input neurons
    - `output_size`:  Number of output neurons
    - `optimizer` : Optimizer for weight update (default `SGDOptimizer`)

    Notes
    -----
    - The input `X` must have the form of 2D-array(`Matrix`).

        ```py
        X.shape = (batch_size, n_features)
        ```
    """

    def __init__(
        self, input_size: int, output_size: int, optimizer: Optimizer = SGDOptimizer()
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer

        self.weights_: Matrix = 0.01 * np.random.randn(
            self.input_size, self.output_size
        )
        self.biases_: Vector = np.zeros(self.output_size)

    def forward(self, X: Tensor | Matrix) -> Tensor:
        X = self._flatten(X)
        out = np.dot(X, self.weights_) + self.biases_
        return out

    def backward(self, X: Tensor, d_out: Tensor) -> Tensor:
        X = self._flatten(X)
        self.dX = np.dot(d_out, self.weights_.T)
        self.dW = np.dot(X.T, d_out)
        self.dB = np.sum(d_out, axis=0, keepdims=True)

        return self.dX

    def _flatten(self, X: Tensor) -> Matrix:
        return X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
