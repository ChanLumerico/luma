from typing import Literal, Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Layer


__all__ = (
    "_Pool1D",
    "_Pool2D",
    "_Pool3D",
    "_GlobalAvgPool1D",
    "_GlobalAvgPool2D",
    "_GlobalAvgPool3D",
    "_AdaptiveAvgPool1D",
    "_AdaptiveAvgPool2D",
    "_AdaptiveAvgPool3D",
    "_LpPool1D",
    "_LpPool2D",
    "_LpPool3D",
)


class _Pool1D(Layer):
    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
        padding: Tuple[int] | int | Literal["same", "valid"] = "valid",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode
        self.padding = padding

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

        pad_w, padded_w = self._get_padding_dim(width)

        out_width = 1 + (padded_w - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_width))

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")

        for i in range(out_width):
            w_start, w_end = self._get_pooling_bounds(i)
            window = X_padded[:, :, w_start:w_end]

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
        pad_w, _ = self._get_padding_dim(X.shape[2])
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")
        dX_padded = np.zeros_like(X_padded)

        for i in range(out_width):
            w_start, w_end = self._get_pooling_bounds(i)
            window = X_padded[:, :, w_start:w_end]

            if self.mode == "max":
                max_vals = np.max(window, axis=2, keepdims=True)
                mask_ = window == max_vals
                dX_padded[:, :, w_start:w_end] += mask_ * d_out[:, :, i : i + 1]
            elif self.mode == "avg":
                dX_padded[:, :, w_start:w_end] += (
                    d_out[:, :, i : i + 1] / self.filter_size
                )

        if pad_w > 0:
            self.dX = dX_padded[:, :, pad_w:-pad_w]
        else:
            self.dX = dX_padded

        return self.dX

    def _get_padding_dim(self, width: int) -> Tuple[int, ...]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 1:
                raise ValueError("Padding tuple must have exactly one value.")
            pad_w = self.padding[0]

        elif isinstance(self.padding, int):
            pad_w = self.padding
        elif self.padding == "same":
            pad_w = (self.filter_size - 1) // 2
        elif self.padding == "valid":
            pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_w = width + 2 * pad_w
        return pad_w, padded_w

    def _get_pooling_bounds(self, cur_w: int) -> Tuple[int, int]:
        w_start = cur_w * self.stride
        w_end = w_start + self.filter_size

        return w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, width = in_shape
        _, padded_w = self._get_padding_dim(width)

        out_width = 1 + (padded_w - self.filter_size) // self.stride
        return (batch_size, channels, out_width)


class _Pool2D(Layer):
    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
        padding: Tuple[int, int] | int | Literal["same", "valid"] = "valid",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode
        self.padding = padding

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

        pad_h, pad_w, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_height, out_width))

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_pooling_bounds(i, j)
                window = X_padded[:, :, h_start:h_end, w_start:w_end]

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
        pad_h, pad_w, _, _ = self._get_padding_dim(X.shape[2], X.shape[3])
        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        dX_padded = np.zeros_like(X_padded)

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_pooling_bounds(i, j)
                window = X_padded[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    max_vals = np.max(window, axis=(2, 3), keepdims=True)
                    mask_ = window == max_vals
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += (
                        mask_ * d_out[:, :, i : i + 1, j : j + 1]
                    )
                elif self.mode == "avg":
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += d_out[
                        :, :, i : i + 1, j : j + 1
                    ] / (self.filter_size**2)

        if pad_h > 0 or pad_w > 0:
            self.dX = dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            self.dX = dX_padded

        return self.dX

    def _get_padding_dim(self, height: int, width: int) -> Tuple[int, ...]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 2:
                raise ValueError("Padding tuple must have exactly two values.")
            pad_h, pad_w = self.padding
        elif isinstance(self.padding, int):
            pad_h = pad_w = self.padding

        elif self.padding == "same":
            pad_h = pad_w = (self.filter_size - 1) // 2
        elif self.padding == "valid":
            pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w

        return pad_h, pad_w, padded_h, padded_w

    def _get_pooling_bounds(self, cur_h: int, cur_w: int) -> Tuple[int, ...]:
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride

        h_end = h_start + self.filter_size
        w_end = w_start + self.filter_size

        return h_start, h_end, w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, height, width = in_shape
        _, _, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride

        return (batch_size, channels, out_height, out_width)


class _Pool3D(Layer):
    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
        padding: Tuple[int, int, int] | int | Literal["same", "valid"] = "valid",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode
        self.padding = padding

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

        pad_d, pad_h, pad_w, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )

        out_depth = 1 + (padded_d - self.filter_size) // self.stride
        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride
        out: Tensor = np.zeros(
            (
                batch_size,
                channels,
                out_depth,
                out_height,
                out_width,
            )
        )
        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start, d_end, h_start, h_end, w_start, w_end = (
                        self._get_pooling_bounds(i, j, k)
                    )
                    window = X_padded[
                        :,
                        :,
                        d_start:d_end,
                        h_start:h_end,
                        w_start:w_end,
                    ]
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
        pad_d, pad_h, pad_w, _, _, _ = self._get_padding_dim(
            X.shape[2], X.shape[3], X.shape[4]
        )
        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        dX_padded = np.zeros_like(X_padded)

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start, d_end, h_start, h_end, w_start, w_end = (
                        self._get_pooling_bounds(i, j, k)
                    )
                    window = X_padded[
                        :,
                        :,
                        d_start:d_end,
                        h_start:h_end,
                        w_start:w_end,
                    ]
                    if self.mode == "max":
                        max_vals = np.max(window, axis=(2, 3, 4), keepdims=True)
                        mask_ = window == max_vals
                        dX_padded[
                            :, :, d_start:d_end, h_start:h_end, w_start:w_end
                        ] += (mask_ * d_out[:, :, i : i + 1, j : j + 1, k : k + 1])
                    elif self.mode == "avg":
                        dX_padded[
                            :, :, d_start:d_end, h_start:h_end, w_start:w_end
                        ] += d_out[:, :, i : i + 1, j : j + 1, k : k + 1] / (
                            self.filter_size**3
                        )

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            self.dX = dX_padded[:, :, pad_d:-pad_d, pad_h:-pad_h, pad_w:-pad_w]
        else:
            self.dX = dX_padded

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
            pad_d = pad_h = pad_w = (self.filter_size - 1) // 2
        elif self.padding == "valid":
            pad_d = pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_d = depth + 2 * pad_d
        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w

        return pad_d, pad_h, pad_w, padded_d, padded_h, padded_w

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
        _, _, _, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )
        out_depth = 1 + (padded_d - self.filter_size) // self.stride
        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride

        return (batch_size, channels, out_depth, out_height, out_width)


class _GlobalAvgPool1D(Layer):
    def __init__(self) -> None:
        super().__init__()

    @Tensor.force_dim(3)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        out = np.mean(X, axis=2, keepdims=True)

        return out

    @Tensor.force_dim(3)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, width = X.shape

        dX = np.zeros_like(X)
        dX += d_out / width
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _ = in_shape
        return (batch_size, channels, 1)


class _GlobalAvgPool2D(Layer):
    def __init__(self) -> None:
        super().__init__()

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        out = np.mean(X, axis=(2, 3), keepdims=True)

        return out

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, height, width = X.shape

        dX = np.zeros_like(X)
        dX += d_out / (height * width)
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _, _ = in_shape
        return (batch_size, channels, 1, 1)


class _GlobalAvgPool3D(Layer):
    def __init__(self) -> None:
        super().__init__()

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        out = np.mean(X, axis=(2, 3, 4), keepdims=True)

        return out

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, depth, height, width = X.shape

        dX = np.zeros_like(X)
        dX += d_out / (depth * height * width)
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _, _, _ = in_shape
        return (batch_size, channels, 1, 1, 1)


class _AdaptiveAvgPool1D(Layer):
    def __init__(self, out_size: int | Tuple[int]) -> None:
        super().__init__()
        self.out_size = out_size

    @Tensor.force_dim(3)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, width = X.shape
        target_width = self.out_size

        out = np.zeros((batch_size, channels, target_width))

        for i in range(target_width):
            start = int(np.floor(i * width / target_width))
            end = int(np.ceil((i + 1) * width / target_width))

            out[:, :, i] = np.mean(X[:, :, start:end], axis=2)

        return out

    @Tensor.force_dim(3)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, width = X.shape
        target_width = self.out_size

        dX = np.zeros_like(X)
        for i in range(target_width):
            start = int(np.floor(i * width / target_width))
            end = int(np.ceil((i + 1) * width / target_width))

            dX[:, :, start:end] += d_out[:, :, i][:, :, None] / (end - start)

        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _ = in_shape
        return (batch_size, channels, self.out_size)


class _AdaptiveAvgPool2D(Layer):
    def __init__(self, out_size: Tuple[int, int]) -> None:
        super().__init__()
        self.out_size = out_size

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, height, width = X.shape
        target_height, target_width = self.out_size

        out = np.zeros((batch_size, channels, target_height, target_width))

        for i in range(target_height):
            for j in range(target_width):
                h_start = int(np.floor(i * height / target_height))
                h_end = int(np.ceil((i + 1) * height / target_height))
                w_start = int(np.floor(j * width / target_width))
                w_end = int(np.ceil((j + 1) * width / target_width))

                out[:, :, i, j] = np.mean(
                    X[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)
                )

        return out

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, height, width = X.shape
        target_height, target_width = self.out_size

        dX = np.zeros_like(X)
        for i in range(target_height):
            for j in range(target_width):
                h_start = int(np.floor(i * height / target_height))
                h_end = int(np.ceil((i + 1) * height / target_height))
                w_start = int(np.floor(j * width / target_width))
                w_end = int(np.ceil((j + 1) * width / target_width))

                dX[:, :, h_start:h_end, w_start:w_end] += d_out[:, :, i, j][
                    :, :, None, None
                ] / ((h_end - h_start) * (w_end - w_start))

        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _, _ = in_shape
        return (batch_size, channels, *self.out_size)


class _AdaptiveAvgPool3D(Layer):
    def __init__(self, out_size: Tuple[int, int, int]) -> None:
        super().__init__()
        self.out_size = out_size

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, depth, height, width = X.shape
        target_depth, target_height, target_width = self.out_size

        out = np.zeros(
            (
                batch_size,
                channels,
                target_depth,
                target_height,
                target_width,
            )
        )
        for d in range(target_depth):
            d_start = int(np.floor(d * depth / target_depth))
            d_end = int(np.ceil((d + 1) * depth / target_depth))

            for i in range(target_height):
                h_start = int(np.floor(i * height / target_height))
                h_end = int(np.ceil((i + 1) * height / target_height))

                for j in range(target_width):
                    w_start = int(np.floor(j * width / target_width))
                    w_end = int(np.ceil((j + 1) * width / target_width))

                    out[:, :, d, i, j] = np.mean(
                        X[:, :, d_start:d_end, h_start:h_end, w_start:w_end],
                        axis=(2, 3, 4),
                    )

        return out

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, depth, height, width = X.shape
        target_depth, target_height, target_width = self.out_size

        dX = np.zeros_like(X)

        for d in range(target_depth):
            d_start = int(np.floor(d * depth / target_depth))
            d_end = int(np.ceil((d + 1) * depth / target_depth))

            for i in range(target_height):
                h_start = int(np.floor(i * height / target_height))
                h_end = int(np.ceil((i + 1) * height / target_height))

                for j in range(target_width):
                    w_start = int(np.floor(j * width / target_width))
                    w_end = int(np.ceil((j + 1) * width / target_width))

                    dX[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += d_out[
                        :, :, d, i, j
                    ][:, :, None, None, None] / (
                        (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
                    )

        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _, _, _ = in_shape
        return (batch_size, channels, *self.out_size)


class _LpPool1D(Layer):
    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        p: float = 2.0,
        padding: Tuple[int] | int | Literal["same", "valid"] = "valid",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.p = p
        self.padding = padding

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "p": ("0<,+inf", float),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(3)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, width = X.shape

        pad_w, padded_w = self._get_padding_dim(width)

        out_width = 1 + (padded_w - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_width))

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")

        for i in range(out_width):
            w_start, w_end = self._get_pooling_bounds(i)
            window = X_padded[:, :, w_start:w_end]

            out[:, :, i] = np.power(
                np.sum(np.power(np.abs(window), self.p), axis=2), 1 / self.p
            )

        return out

    @Tensor.force_dim(3)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, width = X.shape

        pad_w, _ = self._get_padding_dim(width)
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_w, pad_w)), mode="constant")
        dX_padded = np.zeros_like(X_padded)

        out_width = d_out.shape[2]

        for i in range(out_width):
            w_start, w_end = self._get_pooling_bounds(i)
            window = X_padded[:, :, w_start:w_end]

            window_p_norm = np.power(
                np.sum(np.power(np.abs(window), self.p), axis=2), 1 / self.p
            )

            gradient = (np.power(np.abs(window), self.p - 1) * np.sign(window)) / (
                np.power(window_p_norm, self.p - 1, keepdims=True) + 1e-12
            )

            dX_padded[:, :, w_start:w_end] += gradient * d_out[:, :, i : i + 1]

        if pad_w > 0:
            self.dX = dX_padded[:, :, pad_w:-pad_w]
        else:
            self.dX = dX_padded

        return self.dX

    def _get_padding_dim(self, width: int) -> Tuple[int, int]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 1:
                raise ValueError("Padding tuple must have exactly one value.")
            pad_w = self.padding[0]

        elif isinstance(self.padding, int):
            pad_w = self.padding
        elif self.padding == "same":
            pad_w = (self.filter_size - 1) // 2
        elif self.padding == "valid":
            pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_w = width + 2 * pad_w
        return pad_w, padded_w

    def _get_pooling_bounds(self, cur_w: int) -> Tuple[int, int]:
        w_start = cur_w * self.stride
        w_end = w_start + self.filter_size

        return w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, width = in_shape
        _, padded_w = self._get_padding_dim(width)

        out_width = 1 + (padded_w - self.filter_size) // self.stride
        return (batch_size, channels, out_width)


class _LpPool2D(Layer):
    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        p: float = 2.0,
        padding: Tuple[int, int] | int | Literal["same", "valid"] = "valid",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.p = p
        self.padding = padding

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "p": ("0<,+inf", float),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, height, width = X.shape

        pad_h, pad_w, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_height, out_width))

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_pooling_bounds(i, j)
                window = X_padded[:, :, h_start:h_end, w_start:w_end]

                out[:, :, i, j] = np.power(
                    np.sum(np.power(np.abs(window), self.p), axis=(2, 3)), 1 / self.p
                )

        return out

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, height, width = X.shape

        pad_h, pad_w, _, _ = self._get_padding_dim(height, width)
        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        dX_padded = np.zeros_like(X_padded)

        out_height, out_width = d_out.shape[2], d_out.shape[3]

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_pooling_bounds(i, j)
                window = X_padded[:, :, h_start:h_end, w_start:w_end]

                window_p_norm = np.power(
                    np.sum(np.power(np.abs(window), self.p), axis=(2, 3)), 1 / self.p
                )

                gradient = (np.power(np.abs(window), self.p - 1) * np.sign(window)) / (
                    np.power(window_p_norm, self.p - 1) + 1e-12
                )

                dX_padded[:, :, h_start:h_end, w_start:w_end] += (
                    gradient * d_out[:, :, i : i + 1, j : j + 1]
                )

        if pad_h > 0 or pad_w > 0:
            self.dX = dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            self.dX = dX_padded

        return self.dX

    def _get_padding_dim(self, height: int, width: int) -> Tuple[int, ...]:
        if isinstance(self.padding, tuple):
            if len(self.padding) != 2:
                raise ValueError("Padding tuple must have exactly two values.")
            pad_h, pad_w = self.padding
        elif isinstance(self.padding, int):
            pad_h = pad_w = self.padding

        elif self.padding == "same":
            pad_h = pad_w = (self.filter_size - 1) // 2
        elif self.padding == "valid":
            pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w

        return pad_h, pad_w, padded_h, padded_w

    def _get_pooling_bounds(self, cur_h: int, cur_w: int) -> Tuple[int, ...]:
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride

        h_end = h_start + self.filter_size
        w_end = w_start + self.filter_size

        return h_start, h_end, w_start, w_end

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, height, width = in_shape
        _, _, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride

        return (batch_size, channels, out_height, out_width)


class _LpPool3D(Layer):
    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        p: float = 2.0,
        padding: Tuple[int, int, int] | int | Literal["same", "valid"] = "valid",
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.p = p
        self.padding = padding

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "p": ("0<,+inf", float),
            }
        )
        self.check_param_ranges()

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        self.input_ = X
        batch_size, channels, depth, height, width = X.shape

        pad_d, pad_h, pad_w, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )

        out_depth = 1 + (padded_d - self.filter_size) // self.stride
        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride
        out: Tensor = np.zeros((batch_size, channels, out_depth, out_height, out_width))

        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start, d_end, h_start, h_end, w_start, w_end = (
                        self._get_pooling_bounds(i, j, k)
                    )
                    window = X_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                    out[:, :, i, j, k] = np.power(
                        np.sum(np.power(np.abs(window), self.p), axis=(2, 3, 4)),
                        1 / self.p,
                    )

        return out

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, depth, height, width = X.shape

        pad_d, pad_h, pad_w, _, _, _ = self._get_padding_dim(depth, height, width)
        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        dX_padded = np.zeros_like(X_padded)

        out_depth, out_height, out_width = (
            d_out.shape[2],
            d_out.shape[3],
            d_out.shape[4],
        )
        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start, d_end, h_start, h_end, w_start, w_end = (
                        self._get_pooling_bounds(i, j, k)
                    )
                    window = X_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                    window_p_norm = np.power(
                        np.sum(np.power(np.abs(window), self.p), axis=(2, 3, 4)),
                        1 / self.p,
                    )

                    gradient = (
                        np.power(np.abs(window), self.p - 1) * np.sign(window)
                    ) / (np.power(window_p_norm, self.p - 1, keepdims=True) + 1e-12)

                    dX_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += (
                        gradient * d_out[:, :, i : i + 1, j : j + 1, k : k + 1]
                    )

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            self.dX = dX_padded[:, :, pad_d:-pad_d, pad_h:-pad_h, pad_w:-pad_w]
        else:
            self.dX = dX_padded

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
            pad_d = pad_h = pad_w = (self.filter_size - 1) // 2
        elif self.padding == "valid":
            pad_d = pad_h = pad_w = 0
        else:
            raise UnsupportedParameterError(self.padding)

        padded_d = depth + 2 * pad_d
        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w

        return pad_d, pad_h, pad_w, padded_d, padded_h, padded_w

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
        _, _, _, padded_d, padded_h, padded_w = self._get_padding_dim(
            depth, height, width
        )

        out_depth = 1 + (padded_d - self.filter_size) // self.stride
        out_height = 1 + (padded_h - self.filter_size) // self.stride
        out_width = 1 + (padded_w - self.filter_size) // self.stride

        return (batch_size, channels, out_depth, out_height, out_width)
