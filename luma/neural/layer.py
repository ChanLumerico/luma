from typing import Literal, Tuple
import numpy as np

from luma.interface.util import Tensor, ActivationUtil
from luma.interface.exception import UnsupportedParameterError


__all__ = ("Layer", "Convolution", "Pooling")


class Layer:
    """
    An internal class for layers in neural networks.

    Neural network layers are composed of interconnected nodes,
    each performing computations on input data. Common types include
    fully connected, convolutional, and recurrent layers, each
    serving distinct roles in learning from data.
    """

    def forward(self) -> Tensor: ...

    def backward(self) -> Tensor | Tuple[Tensor, ...]: ...


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

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    - `backward` returns gradients w.r.t. the input, the weights(filters),
        and the biases respectively.

        ```py
        def backward(self, ...) -> Tuple[Tensor, Tensor, Tensor]
        ```
    """

    def __init__(
        self,
        n_filters: int,
        size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        activation: ActivationUtil.FuncType = "relu",
        random_state: int = None,
    ) -> None:
        self.n_filters = n_filters
        self.size = size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        act = ActivationUtil(self.activation)
        self.act_ = act.activation_type()
        self.rs_ = np.random.RandomState(random_state)

        self.filters_ = None
        self.biases_ = np.zeros(self.n_filters)

    def forward(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 4, "X must have the form of 4D-array!"
        batch_size, channels, height, width = X.shape

        if self.filters_ is None:
            self.filters_ = 0.1 * self.rs_.randn(
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
            self.filters_, s=(padded_height, padded_width), axes=[2, 3]
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
        dW = np.zeros_like(self.filters_)
        dB = np.zeros(self.n_filters)

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_height, padded_width), axes=[2, 3])
        d_out_fft = np.fft.rfftn(d_out, s=(padded_height, padded_width), axes=[2, 3])

        for f in range(self.n_filters):
            dB[f] = np.sum(d_out[:, f, :, :])

        for f in range(self.n_filters):
            for c in range(channels):
                filter_d_out_fft = np.sum(X_fft[:, c] * d_out_fft[:, f].conj(), axis=0)
                dW[f, c] = np.fft.irfftn(
                    filter_d_out_fft, s=(padded_height, padded_width)
                )[pad_h : pad_h + self.size, pad_w : pad_w + self.size]

        for i in range(batch_size):
            for c in range(channels):
                temp = np.zeros(
                    (padded_height, padded_width // 2 + 1), dtype=np.complex128
                )
                for f in range(self.n_filters):
                    filter_fft = np.fft.rfftn(
                        self.filters_[f, c], s=(padded_height, padded_width)
                    )
                    temp += filter_fft * d_out_fft[i, f]
                dX_padded[i, c] = np.fft.irfftn(temp, s=(padded_height, padded_width))

        dX = (
            dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
            if pad_h > 0 or pad_w > 0
            else dX_padded
        )

        dX = self.act_.derivative(dX)
        return dX, dW, dB

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

    """

    def __init__(
        self, size: int = 2, stride: int = 2, mode: Literal["max", "avg"] = "max"
    ) -> None:
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
        dX = np.zeros_like(X)

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_height_width(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    max_vals = np.max(window, axis=(2, 3), keepdims=True)
                    mask = window == max_vals
                    dX[:, :, h_start:h_end, w_start:w_end] += (
                        mask * d_out[:, :, i : i + 1, j : j + 1]
                    )
                elif self.mode == "avg":
                    avg_grad = d_out[:, :, i, j] / (self.size**2)
                    dX[:, :, h_start:h_end, w_start:w_end] += (
                        np.ones((1, 1, self.size, self.size)) * avg_grad
                    )

        return dX

    def _get_height_width(self, cur_h: int, cur_w: int) -> Tuple[int, int, int, int]:
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride
        h_end = h_start + self.size
        w_end = w_start + self.size

        return h_start, h_end, w_start, w_end
