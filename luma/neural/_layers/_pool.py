from typing import Literal, Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Layer


__all__ = ("_Pool1D", "_Pool2D", "_Pool3D")


class _Pool1D(Layer):
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

    @Tensor.force_dim(3)
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

    @Tensor.force_dim(3)
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


class _Pool2D(Layer):
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

    @Tensor.force_dim(4)
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

    @Tensor.force_dim(4)
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


class _Pool3D(Layer):
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

    @Tensor.force_dim(5)
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

    @Tensor.force_dim(5)
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
