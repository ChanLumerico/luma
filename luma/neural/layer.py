from typing import Literal, Tuple
import numpy as np

from luma.interface.util import Tensor, ActivationUtil
from luma.interface.exception import UnsupportedParameterError


__all__ = "Convolution"


class Layer:
    def forward(self, **kwargs): ...
    def backward(self, **kwargs): ...


class Convolution(Layer):
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

    def forward(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 4, "X must have the form of 4D-array!"
        batch_size, channels, height, width = X.shape

        if self.filters_ is None:
            self.filters_ = 0.1 * self.rs_.randn(
                self.n_filters, channels, self.size, self.size
            )

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

        out_height = ((padded_height - self.size) // self.stride) + 1
        out_width = ((padded_width - self.size) // self.stride) + 1
        out = np.zeros((batch_size, self.n_filters, out_height, out_width))

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

        out = self.act_.func(out)
        return out

    def backward(self, X: Tensor, d_out: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, channels, height, width = X.shape

        if self.padding == "same":
            pad_h = pad_w = (self.size - 1) // 2
        elif self.padding == "valid":
            pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_height = height + 2 * pad_h
        padded_width = width + 2 * pad_w

        dX_padded = np.zeros((batch_size, channels, padded_height, padded_width))
        dW = np.zeros_like(self.filters_)

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_height, padded_width), axes=[2, 3])
        d_out_fft = np.fft.rfftn(d_out, s=(padded_height, padded_width), axes=[2, 3])

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
        return dX, dW
