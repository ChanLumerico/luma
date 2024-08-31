from typing import Tuple, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.util import InitUtil
from luma.interface.typing import TensorLike

from luma.neural.layer import *


class _SEBlock1D(Sequential):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        activation: callable = Activation.HardSwish,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        keep_shape: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.keep_shape = keep_shape
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "reduction": (f"0<,{in_channels}", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

        super(_SEBlock1D, self).__init__(
            GlobalAvgPool1D(),
            Flatten(),
            Dense(in_channels, in_channels // reduction, **basic_args),
            Activation.ReLU(),
            Dense(in_channels // reduction, in_channels, **basic_args),
            activation(),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        _, _, width = X.shape
        out = super().forward(X, is_train)
        if self.keep_shape:
            return np.broadcast_to(
                out[:, :, np.newaxis],
                (*out.shape, width),
            )
        else:
            return out

    @override
    def backward(self, d_out: TensorLike) -> TensorLike:
        d_out_sq = d_out[:, :, 0] if self.keep_shape else d_out
        return super().backward(d_out_sq)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape if self.keep_shape else in_shape[:2]


class _SEBlock2D(Sequential):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        activation: callable = Activation.HardSwish,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        keep_shape: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.keep_shape = keep_shape
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "reduction": (f"0<,{in_channels}", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

        super(_SEBlock2D, self).__init__(
            GlobalAvgPool2D(),
            Flatten(),
            Dense(in_channels, in_channels // reduction, **basic_args),
            Activation.ReLU(),
            Dense(in_channels // reduction, in_channels, **basic_args),
            activation(),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        _, _, height, width = X.shape
        out = super().forward(X, is_train)
        if self.keep_shape:
            return np.broadcast_to(
                out[:, :, np.newaxis, np.newaxis],
                (*out.shape, height, width),
            )
        else:
            return out

    @override
    def backward(self, d_out: TensorLike) -> TensorLike:
        d_out_sq = d_out[:, :, 0, 0] if self.keep_shape else d_out
        return super().backward(d_out_sq)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape if self.keep_shape else in_shape[:2]


class _SEBlock3D(Sequential):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        activation: callable = Activation.HardSwish,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        keep_shape: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.keep_shape = keep_shape
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "reduction": (f"0<,{in_channels}", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

        super(_SEBlock3D, self).__init__(
            GlobalAvgPool3D(),
            Flatten(),
            Dense(in_channels, in_channels // reduction, **basic_args),
            Activation.ReLU(),
            Dense(in_channels // reduction, in_channels, **basic_args),
            activation(),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        _, _, depth, height, width = X.shape
        out = super().forward(X, is_train)
        if self.keep_shape:
            return np.broadcast_to(
                out[:, :, np.newaxis, np.newaxis, np.newaxis],
                (*out.shape[:2], depth, height, width),
            )
        else:
            return out

    @override
    def backward(self, d_out: TensorLike) -> TensorLike:
        d_out_sq = d_out[:, :, 0, 0, 0] if self.keep_shape else d_out
        return super().backward(d_out_sq)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape if self.keep_shape else in_shape[:2]
