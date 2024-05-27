from typing import Tuple, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, Vector
from luma.interface.util import InitUtil

from luma.neural.layer import *


__all__ = ("_V2_TypeA", "_V2_TypeB", "_V2_TypeC")


class _V2_TypeA(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_3x3_db: int,
        out_3x3_db: Tuple[int, int],
        out_pool: int,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.out_1x1 = out_1x1
        self.out_3x3 = out_3x3
        self.out_3x3_db = out_3x3_db
        self.out_pool = out_pool

        basic_args = {
            "initializer": initializer,
            "optimizer": optimizer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_1x1": ("0<,+inf", int),
                "red_3x3": ("0<,+inf", int),
                "out_3x3": ("0<,+inf", int),
                "red_3x3_db": ("0<,+inf", int),
                "out_pool": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        self.branch_1x1 = Sequential(
            Convolution2D(in_channels, out_1x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_1x1, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_3x3 = Sequential(
            Convolution2D(in_channels, red_3x3, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_3x3, out_3x3, 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_3x3_db = Sequential(
            Convolution2D(in_channels, red_3x3_db, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3_db, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_3x3_db, out_3x3_db[0], 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3_db[0], momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_3x3_db.extend(
            Convolution2D(out_3x3_db[0], out_3x3_db[1], 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3_db[1], momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_pool = Sequential(
            Pooling2D(3, 1, "avg", "same"),
            Convolution2D(in_channels, out_pool, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_pool, momentum) if do_batch_norm else None,
            activation(),
        )

        super(_V2_TypeA, self).__init__()
        self.layers = [
            *self.branch_1x1.layers,
            *self.branch_3x3.layers,
            *self.branch_3x3_db.layers,
            *self.branch_pool.layers,
        ]

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        branch_1x1 = self.branch_1x1(X, is_train)
        branch_3x3 = self.branch_3x3(X, is_train)
        branch_3x3_db = self.branch_3x3_db(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out_channels = [
            out_.shape[1]
            for out_ in [
                branch_1x1,
                branch_3x3,
                branch_3x3_db,
                branch_pool,
            ]
        ]
        self.out_channels = Vector(out_channels)
        out = np.concatenate(
            (branch_1x1, branch_3x3, branch_3x3_db, branch_pool),
            axis=1,
        )
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_1x1 = d_out[:, : self.out_channels[0], ...]
        d_out_3x3 = d_out[:, self.out_channels[0] : self.out_channels[:2].sum(), ...]
        d_out_3x3_db = d_out[
            :, self.out_channels[:2].sum() : self.out_channels[:3].sum(), ...
        ]
        d_out_pool = d_out[:, -self.out_channels[3] :, ...]

        dX_1x1 = self.branch_1x1.backward(d_out_1x1)
        dX_3x3 = self.branch_3x3.backward(d_out_3x3)
        dX_3x3_db = self.branch_3x3_db.backward(d_out_3x3_db)
        dX_pool = self.branch_pool.backward(d_out_pool)

        self.dX = dX_1x1 + dX_3x3 + dX_3x3_db + dX_pool
        return self.dX

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        return (
            batch_size,
            self.out_1x1 + self.out_3x3 + self.out_3x3_db[1] + self.out_pool,
            height,
            width,
        )


class _V2_TypeB(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_7x7: int,
        out_7x7: int,
        red_7x7_db: int,
        out_7x7_db: Tuple[int, int],
        out_pool: int,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.out_1x1 = out_1x1
        self.out_7x7 = out_7x7
        self.out_7x7_db = out_7x7_db
        self.out_pool = out_pool

        basic_args = {
            "initializer": initializer,
            "optimizer": optimizer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_1x1": ("0<,+inf", int),
                "red_7x7": ("0<,+inf", int),
                "out_7x7": ("0<,+inf", int),
                "red_7x7_db": ("0<,+inf", int),
                "out_pool": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        self.branch_1x1 = Sequential(
            Convolution2D(in_channels, out_1x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_1x1, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_7x7 = Sequential(
            Convolution2D(in_channels, red_7x7, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_7x7, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_7x7, red_7x7, (1, 7), 1, (0, 3), **basic_args),
            BatchNorm2D(red_7x7, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_7x7, out_7x7, (7, 1), 1, (3, 0), **basic_args),
            BatchNorm2D(out_7x7, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_7x7_db = Sequential(
            Convolution2D(in_channels, red_7x7_db, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_7x7_db, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_7x7_db, red_7x7_db, (1, 7), 1, (0, 3), **basic_args),
            BatchNorm2D(red_7x7_db, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_7x7_db, out_7x7_db[0], (7, 1), 1, (3, 0), **basic_args),
            BatchNorm2D(out_7x7_db[0], momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_7x7_db.extend(
            Convolution2D(
                out_7x7_db[0], out_7x7_db[0], (1, 7), 1, (0, 3), **basic_args
            ),
            BatchNorm2D(out_7x7_db[0], momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(
                out_7x7_db[0], out_7x7_db[1], (7, 1), 1, (3, 0), **basic_args
            ),
            BatchNorm2D(out_7x7_db[1], momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_pool = Sequential(
            Pooling2D(3, 1, "max", "same"),
            Convolution2D(in_channels, out_pool, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_pool, momentum) if do_batch_norm else None,
            activation(),
        )

        super(_V2_TypeB, self).__init__()
        self.layers = [
            *self.branch_1x1.layers,
            *self.branch_7x7.layers,
            *self.branch_7x7_db.layers,
            *self.branch_pool.layers,
        ]

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        branch_1x1 = self.branch_1x1(X, is_train)
        branch_7x7 = self.branch_7x7(X, is_train)
        branch_7x7_db = self.branch_7x7_db(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out_channels = [
            out_.shape[1]
            for out_ in [
                branch_1x1,
                branch_7x7,
                branch_7x7_db,
                branch_pool,
            ]
        ]
        self.out_channels = Vector(out_channels)
        out = np.concatenate(
            (branch_1x1, branch_7x7, branch_7x7_db, branch_pool),
            axis=1,
        )
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_1x1 = d_out[:, : self.out_channels[0], ...]
        d_out_7x7 = d_out[:, self.out_channels[0] : self.out_channels[:2].sum(), ...]
        d_out_7x7_db = d_out[
            :, self.out_channels[:2].sum() : self.out_channels[:3].sum(), ...
        ]
        d_out_pool = d_out[:, -self.out_channels[3] :, ...]

        dX_1x1 = self.branch_1x1.backward(d_out_1x1)
        dX_7x7 = self.branch_7x7.backward(d_out_7x7)
        dX_7x7_db = self.branch_7x7_db.backward(d_out_7x7_db)
        dX_pool = self.branch_pool.backward(d_out_pool)

        self.dX = dX_1x1 + dX_7x7 + dX_7x7_db + dX_pool
        return self.dX

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        return (
            batch_size,
            self.out_1x1 + self.out_7x7 + self.out_7x7_db[1] + self.out_pool,
            height,
            width,
        )
