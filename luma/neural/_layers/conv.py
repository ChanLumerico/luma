from typing import Literal, Tuple
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil
from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Layer


__all__ = (
    "_Conv1D",
    "_Conv2D",
    "_Conv3D",
    "_DepthConv1D",
    "_DepthConv2D",
    "_DepthConv3D",
)


class _Conv1D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int] | int,
        stride: int = 1,
        padding: Tuple[int] | int | Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        if isinstance(filter_size, int):
            self.filter_size = (filter_size,)
        else:
            self.filter_size = filter_size

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                *self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(3)
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

        out_width = ((padded_w - self.filter_size[0]) // self.stride) + 1
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

    @Tensor.force_dim(3)
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
                self.dW[f, c] = np.fft.irfftn(
                    filter_d_out_fft,
                    s=(padded_w,),
                )[pad_w : pad_w + self.filter_size[0]]

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
        if isinstance(self.padding, tuple):
            if len(self.padding) != 1:
                raise ValueError("Padding tuple must have exactly one value.")
            pad_w = self.padding[0]

        elif isinstance(self.padding, int):
            pad_w = self.padding
        elif self.padding == "same":
            pad_w = (self.filter_size[0] - 1) // 2
        elif self.padding == "valid":
            pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_w = width + 2 * pad_w
        return pad_w, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, width = in_shape
        _, padded_w = self._get_padding_dim(width)
        out_width = ((padded_w - self.filter_size[0]) // self.stride) + 1

        return (batch_size, self.out_channels, out_width)


class _Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int,
        stride: int = 1,
        padding: Tuple[int, int] | int | Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)
        else:
            self.filter_size = filter_size

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                *self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(4)
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

        out_height = ((padded_h - self.filter_size[0]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[1]) // self.stride) + 1
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

    @Tensor.force_dim(4)
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
                )[
                    pad_h : pad_h + self.filter_size[0],
                    pad_w : pad_w + self.filter_size[1],
                ]

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
        if isinstance(self.padding, tuple):
            if len(self.padding) != 2:
                raise ValueError("Padding tuple must have exactly two values.")
            pad_h, pad_w = self.padding
        elif isinstance(self.padding, int):
            pad_h = pad_w = self.padding

        elif self.padding == "same":
            pad_h = (self.filter_size[0] - 1) // 2
            pad_w = (self.filter_size[1] - 1) // 2
        elif self.padding == "valid":
            pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w

        return pad_h, pad_w, padded_h, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        _, _, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = ((padded_h - self.filter_size[0]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[1]) // self.stride) + 1

        return (batch_size, self.out_channels, out_height, out_width)


class _Conv3D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int, int] | int,
        stride: int = 1,
        padding: Tuple[int, int, int] | int | Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size, filter_size)
        else:
            self.filter_size = filter_size

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                *self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(5)
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
        out_depth = ((padded_d - self.filter_size[0]) // self.stride) + 1
        out_height = ((padded_h - self.filter_size[1]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[2]) // self.stride) + 1
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

    @Tensor.force_dim(5)
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
                    pad_d : pad_d + self.filter_size[0],
                    pad_h : pad_h + self.filter_size[1],
                    pad_w : pad_w + self.filter_size[2],
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
        if isinstance(self.padding, tuple):
            if len(self.padding) != 3:
                raise ValueError("Padding tuple must have exactly three values.")
            pad_d, pad_h, pad_w = self.padding

        elif isinstance(self.padding, int):
            pad_d = pad_h = pad_w = self.padding
        elif self.padding == "same":
            pad_d = (self.filter_size[0] - 1) // 2
            pad_h = (self.filter_size[1] - 1) // 2
            pad_w = (self.filter_size[2] - 1) // 2
        elif self.padding == "valid":
            pad_d = pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_d = depth + 2 * pad_d
        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w

        return pad_d, pad_h, pad_w, padded_d, padded_h, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, depth, height, width = in_shape
        _, _, _, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )
        out_depth = ((padded_d - self.filter_size[0]) // self.stride) + 1
        out_height = ((padded_h - self.filter_size[1]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[2]) // self.stride) + 1

        return (batch_size, self.out_channels, out_depth, out_height, out_width)


class _DepthConv1D(Layer):
    def __init__(
        self,
        in_channels: int,
        filter_size: Tuple[int] | int,
        stride: int = 1,
        padding: Tuple[int] | int | Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        if isinstance(filter_size, int):
            self.filter_size = (filter_size,)
        else:
            self.filter_size = filter_size

        self.init_params(
            w_shape=(self.in_channels, 1, *self.filter_size),
            b_shape=(1, self.in_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(3)
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

        out_width = ((padded_w - self.filter_size[0]) // self.stride) + 1
        out: Tensor = np.zeros((batch_size, self.in_channels, out_width))

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")
        X_fft = np.fft.rfft(X_padded, n=padded_w, axis=2)
        filter_fft = np.fft.rfft(
            self.weights_,
            n=padded_w,
            axis=2,
        )

        for i in range(batch_size):
            for c in range(self.in_channels):
                result_fft = X_fft[i, c] * filter_fft[c, 0]
                result = np.fft.irfft(result_fft, n=padded_w)

                out[i, c] = result[pad_w : padded_w - pad_w : self.stride][:out_width]

        out += self.biases_[:, :, np.newaxis]
        return out

    @Tensor.force_dim(3)
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

        for c in range(self.in_channels):
            self.dB[:, c] = np.sum(d_out[:, c, :])

        for c in range(self.in_channels):
            filter_d_out_fft = np.sum(
                X_fft[:, c] * d_out_fft[:, c].conj(),
                axis=0,
            )
            self.dW[c, 0] = np.fft.irfftn(
                filter_d_out_fft,
                s=(padded_w,),
            )[pad_w : pad_w + self.filter_size[0]]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                filter_fft = np.fft.rfft(self.weights_[c, 0], n=padded_w)
                temp = filter_fft * d_out_fft[i, c]
                dX_padded[i, c] = np.fft.irfft(temp, n=padded_w)

        self.dX = dX_padded[:, :, pad_w:-pad_w] if pad_w > 0 else dX_padded
        return self.dX

    def _get_padding_dim(self, width: int) -> Tuple[int, ...]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 1:
                raise ValueError("Padding tuple must have exactly one value.")
            pad_w = self.padding[0]

        elif isinstance(self.padding, int):
            pad_w = self.padding
        elif self.padding == "same":
            pad_w = (self.filter_size[0] - 1) // 2
        elif self.padding == "valid":
            pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_w = width + 2 * pad_w
        return pad_w, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, width = in_shape
        _, padded_w = self._get_padding_dim(width)
        out_width = ((padded_w - self.filter_size[0]) // self.stride) + 1

        return (batch_size, self.in_channels, out_width)


class _DepthConv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        filter_size: Tuple[int, int] | int,
        stride: int = 1,
        padding: Tuple[int, int] | int | Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)
        else:
            self.filter_size = filter_size

        self.init_params(
            w_shape=(self.in_channels, 1, *self.filter_size),
            b_shape=(1, self.in_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, height, width = X.shape

        if self.in_channels != channels:
            raise ValueError(
                f"channels of 'X' does not match with 'in_channels'! "
                + f"({self.in_channels}!={channels})"
            )

        pad_h, padded_h, pad_w, padded_w = self._get_padding_dims(height, width)

        out_height = ((padded_h - self.filter_size[0]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[1]) // self.stride) + 1
        out: Tensor = np.zeros((batch_size, self.in_channels, out_height, out_width))

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_h, padded_w), axes=(2, 3))
        filter_fft = np.fft.rfftn(
            self.weights_,
            s=(padded_h, padded_w),
            axes=(2, 3),
        )

        for i in range(batch_size):
            for c in range(self.in_channels):
                result_fft = X_fft[i, c] * filter_fft[c, 0]
                result = np.fft.irfftn(result_fft, s=(padded_h, padded_w))

                out[i, c] = result[
                    pad_h : padded_h - pad_h : self.stride,
                    pad_w : padded_w - pad_w : self.stride,
                ][:out_height, :out_width]

        out += self.biases_[:, :, np.newaxis, np.newaxis]
        return out

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        batch_size, channels, height, width = X.shape
        pad_h, padded_h, pad_w, padded_w = self._get_padding_dims(height, width)

        dX_padded = np.zeros((batch_size, channels, padded_h, padded_w))
        self.dW = np.zeros_like(self.weights_)
        self.dB = np.zeros_like(self.biases_)

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_h, padded_w), axes=(2, 3))
        d_out_fft = np.fft.rfftn(d_out, s=(padded_h, padded_w), axes=(2, 3))

        for c in range(self.in_channels):
            self.dB[:, c] = np.sum(d_out[:, c, :, :])

        for c in range(self.in_channels):
            filter_d_out_fft = np.sum(
                X_fft[:, c] * d_out_fft[:, c].conj(),
                axis=0,
            )
            self.dW[c, 0] = np.fft.irfftn(
                filter_d_out_fft,
                s=(padded_h, padded_w),
            )[
                pad_h : pad_h + self.filter_size[0],
                pad_w : pad_w + self.filter_size[1],
            ]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                filter_fft = np.fft.rfftn(self.weights_[c, 0], s=(padded_h, padded_w))
                temp = filter_fft * d_out_fft[i, c]
                dX_padded[i, c] = np.fft.irfftn(temp, s=(padded_h, padded_w))

        self.dX = (
            dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
            if pad_h > 0 and pad_w > 0
            else dX_padded
        )
        return self.dX

    def _get_padding_dims(self, height: int, width: int) -> Tuple[int, ...]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 2:
                raise ValueError("Padding tuple must have exactly two values.")
            pad_h, pad_w = self.padding

        elif isinstance(self.padding, int):
            pad_h = pad_w = self.padding
        elif self.padding == "same":
            pad_h = (self.filter_size[0] - 1) // 2
            pad_w = (self.filter_size[1] - 1) // 2
        elif self.padding == "valid":
            pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w
        return pad_h, padded_h, pad_w, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        _, padded_h, _, padded_w = self._get_padding_dims(height, width)

        out_height = ((padded_h - self.filter_size[0]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[1]) // self.stride) + 1

        return (batch_size, self.in_channels, out_height, out_width)


class _DepthConv3D(Layer):
    def __init__(
        self,
        in_channels: int,
        filter_size: Tuple[int, int, int] | int,
        stride: int = 1,
        padding: Tuple[int, int, int] | int | Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size, filter_size)
        else:
            self.filter_size = filter_size

        self.init_params(
            w_shape=(self.in_channels, 1, *self.filter_size),
            b_shape=(1, self.in_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, depth, height, width = X.shape

        if self.in_channels != channels:
            raise ValueError(
                f"channels of 'X' does not match with 'in_channels'! "
                + f"({self.in_channels}!={channels})"
            )

        pad_d, padded_d, pad_h, padded_h, pad_w, padded_w = self._get_padding_dims(
            depth, height, width
        )

        out_depth = ((padded_d - self.filter_size[0]) // self.stride) + 1
        out_height = ((padded_h - self.filter_size[1]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[2]) // self.stride) + 1
        out: Tensor = np.zeros(
            (batch_size, self.in_channels, out_depth, out_height, out_width)
        )

        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_d, padded_h, padded_w), axes=(2, 3, 4))
        filter_fft = np.fft.rfftn(
            self.weights_,
            s=(padded_d, padded_h, padded_w),
            axes=(2, 3, 4),
        )

        for i in range(batch_size):
            for c in range(self.in_channels):
                result_fft = X_fft[i, c] * filter_fft[c, 0]
                result = np.fft.irfftn(result_fft, s=(padded_d, padded_h, padded_w))

                out[i, c] = result[
                    pad_d : padded_d - pad_d : self.stride,
                    pad_h : padded_h - pad_h : self.stride,
                    pad_w : padded_w - pad_w : self.stride,
                ][:out_depth, :out_height, :out_width]

        out += self.biases_[:, :, np.newaxis, np.newaxis, np.newaxis]
        return out

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        batch_size, channels, depth, height, width = X.shape
        pad_d, padded_d, pad_h, padded_h, pad_w, padded_w = self._get_padding_dims(
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
        X_fft = np.fft.rfftn(X_padded, s=(padded_d, padded_h, padded_w), axes=(2, 3, 4))
        d_out_fft = np.fft.rfftn(
            d_out, s=(padded_d, padded_h, padded_w), axes=(2, 3, 4)
        )

        for c in range(self.in_channels):
            self.dB[:, c] = np.sum(d_out[:, c, :, :, :])

        for c in range(self.in_channels):
            filter_d_out_fft = np.sum(
                X_fft[:, c] * d_out_fft[:, c].conj(),
                axis=0,
            )
            self.dW[c, 0] = np.fft.irfftn(
                filter_d_out_fft,
                s=(padded_d, padded_h, padded_w),
            )[
                pad_d : pad_d + self.filter_size[0],
                pad_h : pad_h + self.filter_size[1],
                pad_w : pad_w + self.filter_size[2],
            ]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                filter_fft = np.fft.rfftn(
                    self.weights_[c, 0], s=(padded_d, padded_h, padded_w)
                )
                temp = filter_fft * d_out_fft[i, c]
                dX_padded[i, c] = np.fft.irfftn(temp, s=(padded_d, padded_h, padded_w))

        self.dX = (
            dX_padded[:, :, pad_d:-pad_d, pad_h:-pad_h, pad_w:-pad_w]
            if pad_d > 0 and pad_h > 0 and pad_w > 0
            else dX_padded
        )
        return self.dX

    def _get_padding_dims(self, depth: int, height: int, width: int) -> Tuple[int, ...]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 3:
                raise ValueError("Padding tuple must have exactly three values.")
            pad_d, pad_h, pad_w = self.padding

        elif isinstance(self.padding, int):
            pad_d = pad_h = pad_w = self.padding
        elif self.padding == "same":
            pad_d = (self.filter_size[0] - 1) // 2
            pad_h = (self.filter_size[1] - 1) // 2
            pad_w = (self.filter_size[2] - 1) // 2
        elif self.padding == "valid":
            pad_d = pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_d = depth + 2 * pad_d
        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w
        return pad_d, padded_d, pad_h, padded_h, pad_w, padded_w

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, depth, height, width = in_shape
        _, padded_d, _, padded_h, _, padded_w = self._get_padding_dims(
            depth, height, width
        )

        out_depth = ((padded_d - self.filter_size[0]) // self.stride) + 1
        out_height = ((padded_h - self.filter_size[1]) // self.stride) + 1
        out_width = ((padded_w - self.filter_size[2]) // self.stride) + 1

        return (batch_size, self.in_channels, out_depth, out_height, out_width)
