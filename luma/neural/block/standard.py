from typing import Literal, Tuple, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.util import InitUtil
from luma.interface.typing import TensorLike

from luma.neural.layer import *


class _ConvBlock1D(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int] | int,
        activation: callable,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        padding: Tuple[int] | int | Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

        super(_ConvBlock1D, self).__init__(
            Conv1D(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(_ConvBlock1D, self).__add__(
                BatchNorm1D(out_channels, momentum),
            )
        super(_ConvBlock1D, self).__add__(activation())
        if do_pooling:
            super(_ConvBlock1D, self).__add__(
                Pool1D(pool_filter_size, pool_stride, pool_mode)
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _ConvBlock2D(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int,
        activation: callable,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        padding: Tuple[int, int] | int | Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

        super(_ConvBlock2D, self).__init__(
            Conv2D(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(_ConvBlock2D, self).__add__(
                BatchNorm2D(out_channels, momentum),
            )
        super(_ConvBlock2D, self).__add__(activation())
        if do_pooling:
            super(_ConvBlock2D, self).__add__(
                Pool2D(pool_filter_size, pool_stride, pool_mode)
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _ConvBlock3D(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int, int] | int,
        activation: callable,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        padding: Tuple[int, int, int] | int | Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

        super(_ConvBlock3D, self).__init__(
            Conv3D(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(_ConvBlock3D, self).__add__(
                BatchNorm3D(out_channels, momentum),
            )
        super(_ConvBlock3D, self).__add__(activation())
        if do_pooling:
            super(_ConvBlock3D, self).__add__(
                Pool3D(pool_filter_size, pool_stride, pool_mode)
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _SeparableConv1D(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int] | int,
        stride: int = 1,
        padding: Tuple[int] | int | Literal["same", "valid"] = "same",
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_SeparableConv1D, self).__init__(
            DepthConv1D(in_channels, filter_size, stride, padding, **basic_args),
            BatchNorm1D(in_channels, momentum) if do_batch_norm else None,
        )
        self.extend(
            Conv1D(in_channels, out_channels, 1, 1, "valid", **basic_args),
            BatchNorm1D(out_channels, momentum) if do_batch_norm else None,
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _SeparableConv2D(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int] | int,
        stride: int = 1,
        padding: Tuple[int] | int | Literal["same", "valid"] = "same",
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_SeparableConv2D, self).__init__(
            DepthConv2D(in_channels, filter_size, stride, padding, **basic_args),
            BatchNorm2D(in_channels, momentum) if do_batch_norm else None,
        )
        self.extend(
            Conv2D(in_channels, out_channels, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_channels, momentum) if do_batch_norm else None,
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _SeparableConv3D(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int] | int,
        stride: int = 1,
        padding: Tuple[int] | int | Literal["same", "valid"] = "same",
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_SeparableConv3D, self).__init__(
            DepthConv3D(in_channels, filter_size, stride, padding, **basic_args),
            BatchNorm3D(in_channels, momentum) if do_batch_norm else None,
        )
        self.extend(
            Conv3D(in_channels, out_channels, 1, 1, "valid", **basic_args),
            BatchNorm3D(out_channels, momentum) if do_batch_norm else None,
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _DenseBlock(Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: callable,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: float = True,
        momentum: float = 0.9,
        do_dropout: bool = True,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "dropout_rate": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_DenseBlock, self).__init__(
            Dense(
                in_features,
                out_features,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(_DenseBlock, self).__add__(
                BatchNorm1D(
                    1,
                    momentum,
                )
            )
        super(_DenseBlock, self).__add__(
            activation(),
        )
        if do_dropout:
            super(_DenseBlock, self).__add__(
                Dropout(
                    dropout_rate,
                    random_state,
                ),
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        self.input_ = X
        out = X
        for _, layer in self.layers:
            if isinstance(layer, BatchNorm1D):
                out = layer(out[:, np.newaxis, :], is_train=is_train).squeeze()
                continue
            out = layer(out, is_train=is_train)

        self.out_shape = out.shape
        return out

    @override
    def backward(self, d_out: TensorLike) -> TensorLike:
        for _, layer in reversed(self.layers):
            if isinstance(layer, BatchNorm1D):
                d_out = layer.backward(d_out[:, np.newaxis, :]).squeeze()
                continue
            d_out = layer.backward(d_out)
        return d_out
