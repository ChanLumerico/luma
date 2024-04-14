from typing import Literal
import numpy as np

from luma.interface.util import Tensor, ActivationUtil
from luma.interface.exception import UnsupportedParameterError


__all__ = "Convolution"


class Layer: ...


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
        self.rs_ = np.random.RandomState(random_state)

    def forward(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 4, "X must have the form of 4D-array!"
        self.batch_size, channels, height, width = X.shape

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
        out = np.zeros((self.batch_size, self.n_filters, out_height, out_width))

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        input_fft = np.fft.rfftn(X_padded, s=(padded_height, padded_width), axes=[2, 3])
        filter_fft = np.fft.rfftn(
            self.filters_, s=(padded_height, padded_width), axes=[2, 3]
        )

        for i in range(self.batch_size):
            for f in range(self.n_filters):
                result_fft = np.sum(input_fft[i] * filter_fft[f], axis=0)
                result = np.fft.irfftn(result_fft, s=(padded_height, padded_width))

                sampled_result = result[
                    pad_h : padded_height - pad_h : self.stride,
                    pad_w : padded_width - pad_w : self.stride,
                ]
                out[i, f] = sampled_result[:out_height, :out_width]

        print(out.shape)

    def backward(self) -> None: ...
