from typing import Tuple, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural.layer import *


class _Incep_V2_TypeA(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_3x3_db: int,
        out_3x3_db: Tuple[int, int],
        out_pool: int,
        activation: callable = Activation.ReLU,
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
            Conv2D(in_channels, out_1x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_1x1, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_3x3 = Sequential(
            Conv2D(in_channels, red_3x3, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_3x3, out_3x3, 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_3x3_db = Sequential(
            Conv2D(in_channels, red_3x3_db, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3_db, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_3x3_db, out_3x3_db[0], 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3_db[0], momentum) if do_batch_norm else None,
            activation(),
            Conv2D(out_3x3_db[0], out_3x3_db[1], 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3_db[1], momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_pool = Sequential(
            Pool2D(3, 1, "avg", "same"),
            Conv2D(in_channels, out_pool, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_pool, momentum) if do_batch_norm else None,
            activation(),
        )

        super(_Incep_V2_TypeA, self).__init__()
        self.layers = [
            *self.branch_1x1.layers,
            *self.branch_3x3.layers,
            *self.branch_3x3_db.layers,
            *self.branch_pool.layers,
        ]

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        branch_1x1 = self.branch_1x1(X, is_train)
        branch_3x3 = self.branch_3x3(X, is_train)
        branch_3x3_db = self.branch_3x3_db(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out = np.concatenate(
            (branch_1x1, branch_3x3, branch_3x3_db, branch_pool),
            axis=1,
        )
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_1x1 = d_out[:, : self.out_1x1, ...]
        d_out_3x3 = d_out[:, self.out_1x1 : self.out_1x1 + self.out_3x3, ...]
        d_out_3x3_db = d_out[
            :,
            self.out_1x1
            + self.out_3x3 : self.out_1x1
            + self.out_3x3
            + self.out_3x3_db[1],
            ...,
        ]
        d_out_pool = d_out[:, -self.out_pool :, ...]

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


class _Incep_V2_TypeB(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_7x7: int,
        out_7x7: int,
        red_7x7_db: int,
        out_7x7_db: Tuple[int, int],
        out_pool: int,
        activation: callable = Activation.ReLU,
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
            Conv2D(in_channels, out_1x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_1x1, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_7x7 = Sequential(
            Conv2D(in_channels, red_7x7, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_7x7, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_7x7, red_7x7, (1, 7), 1, (0, 3), **basic_args),
            BatchNorm2D(red_7x7, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_7x7, out_7x7, (7, 1), 1, (3, 0), **basic_args),
            BatchNorm2D(out_7x7, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_7x7_db = Sequential(
            Conv2D(in_channels, red_7x7_db, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_7x7_db, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_7x7_db, red_7x7_db, (1, 7), 1, (0, 3), **basic_args),
            BatchNorm2D(red_7x7_db, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_7x7_db, out_7x7_db[0], (7, 1), 1, (3, 0), **basic_args),
            BatchNorm2D(out_7x7_db[0], momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_7x7_db.extend(
            Conv2D(out_7x7_db[0], out_7x7_db[0], (1, 7), 1, (0, 3), **basic_args),
            BatchNorm2D(out_7x7_db[0], momentum) if do_batch_norm else None,
            activation(),
            Conv2D(out_7x7_db[0], out_7x7_db[1], (7, 1), 1, (3, 0), **basic_args),
            BatchNorm2D(out_7x7_db[1], momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_pool = Sequential(
            Pool2D(3, 1, "max", "same"),
            Conv2D(in_channels, out_pool, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_pool, momentum) if do_batch_norm else None,
            activation(),
        )

        super(_Incep_V2_TypeB, self).__init__()
        self.layers = [
            *self.branch_1x1.layers,
            *self.branch_7x7.layers,
            *self.branch_7x7_db.layers,
            *self.branch_pool.layers,
        ]

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        branch_1x1 = self.branch_1x1(X, is_train)
        branch_7x7 = self.branch_7x7(X, is_train)
        branch_7x7_db = self.branch_7x7_db(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out = np.concatenate(
            (branch_1x1, branch_7x7, branch_7x7_db, branch_pool),
            axis=1,
        )
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_1x1 = d_out[:, : self.out_1x1, ...]
        d_out_7x7 = d_out[:, self.out_1x1 : self.out_1x1 + self.out_7x7, ...]
        d_out_7x7_db = d_out[
            :,
            self.out_1x1
            + self.out_7x7 : self.out_1x1
            + self.out_7x7
            + self.out_7x7_db[1],
            ...,
        ]
        d_out_pool = d_out[:, -self.out_pool :, ...]

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


class _Incep_V2_TypeC(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_1x3_3x1: int,
        out_1x3_3x1: Tuple[int, int],
        red_3x3: int,
        out_3x3: int,
        out_1x3_3x1_after: Tuple[int, int],
        out_pool: int,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.out_1x1 = out_1x1
        self.out_1x3_3x1 = out_1x3_3x1
        self.out_1x3_3x1_after = out_1x3_3x1_after
        self.out_pool = out_pool

        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_1x1": ("0<,+inf", int),
                "red_1x3_3x1": ("0<,+inf", int),
                "red_3x3": ("0<,+inf", int),
                "out_3x3": ("0<,+inf", int),
                "out_pool": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        self.branch_1x1 = Sequential(
            Conv2D(in_channels, out_1x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_1x1, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_1x3_3x1 = Sequential(
            Conv2D(in_channels, red_1x3_3x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_1x3_3x1, momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_1x3_3x1_left = Sequential(
            Conv2D(red_1x3_3x1, out_1x3_3x1[0], (1, 3), 1, (0, 1), **basic_args),
            BatchNorm2D(out_1x3_3x1[0], momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_1x3_3x1_right = Sequential(
            Conv2D(red_1x3_3x1, out_1x3_3x1[1], (3, 1), 1, (1, 0), **basic_args),
            BatchNorm2D(out_1x3_3x1[1], momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_3x3 = Sequential(
            Conv2D(in_channels, red_3x3, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_3x3, out_3x3, 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3, momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_3x3_left = Sequential(
            Conv2D(out_3x3, out_1x3_3x1_after[0], (1, 3), 1, (0, 1), **basic_args),
            BatchNorm2D(out_1x3_3x1_after[0], momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_3x3_right = Sequential(
            Conv2D(out_3x3, out_1x3_3x1_after[1], (3, 1), 1, (1, 0), **basic_args),
            BatchNorm2D(out_1x3_3x1_after[1], momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_pool = Sequential(
            Pool2D(3, 1, "max", "same"),
            Conv2D(in_channels, out_pool, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_pool, momentum) if do_batch_norm else None,
            activation(),
        )

        super(_Incep_V2_TypeC, self).__init__()
        self.layers = [
            *self.branch_1x1.layers,
            *self.branch_1x3_3x1.layers,
            *self.branch_1x3_3x1_left.layers,
            *self.branch_1x3_3x1_right.layers,
            *self.branch_3x3.layers,
            *self.branch_3x3_left.layers,
            *self.branch_3x3_right.layers,
            *self.branch_pool.layers,
        ]

        if optimizer is not None:
            self.set_optimizer(optimizer)

        self.branch_1x3_3x1.override_method("forward", self._forward_1x3_3x1)
        self.branch_1x3_3x1.override_method("backward", self._backward_1x3_3x1)

        self.branch_3x3.override_method("forward", self._forward_3x3)
        self.branch_3x3.override_method("backward", self._backward_3x3)

    @Tensor.force_dim(4)
    def _forward_1x3_3x1(self, X: Tensor, is_train: bool = False) -> Tensor:
        x = self.branch_1x3_3x1(X, is_train)
        x_left = self.branch_1x3_3x1_left(x, is_train)
        x_right = self.branch_1x3_3x1_right(x, is_train)

        out = np.concatenate((x_left, x_right), axis=1)
        return out

    @Tensor.force_dim(4)
    def _forward_3x3(self, X: Tensor, is_train: bool = False) -> Tensor:
        x = self.branch_3x3(X, is_train)
        x_left = self.branch_3x3_left(x, is_train)
        x_right = self.branch_3x3_right(x, is_train)

        out = np.concatenate((x_left, x_right), axis=1)
        return out

    @Tensor.force_dim(4)
    def _backward_1x3_3x1(self, d_out: Tensor) -> Tensor:
        d_out_1x3 = d_out[:, : self.out_1x3_3x1[0], ...]
        d_out_3x1 = d_out[:, -self.out_1x3_3x1[1] :, ...]

        dX_1x3 = self.branch_1x3_3x1_left.backward(d_out_1x3)
        dX_3x1 = self.branch_1x3_3x1_right.backward(d_out_3x1)

        d_out_1x1 = dX_1x3 + dX_3x1
        dX = self.branch_1x3_3x1.backward(d_out_1x1)
        return dX

    @Tensor.force_dim(4)
    def _backward_3x3(self, d_out: Tensor) -> Tensor:
        d_out_1x3 = d_out[:, : self.out_1x3_3x1_after[0], ...]
        d_out_3x1 = d_out[:, -self.out_1x3_3x1_after[1] :, ...]

        dX_1x3 = self.branch_3x3_left.backward(d_out_1x3)
        dX_3x1 = self.branch_3x3_right.backward(d_out_3x1)

        d_out_3x3 = dX_1x3 + dX_3x1
        dX = self.branch_3x3.backward(d_out_3x3)
        return dX

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        branch_1x1 = self.branch_1x1(X, is_train)
        branch_1x3_3x1 = self.branch_1x3_3x1(X, is_train)
        branch_3x3 = self.branch_3x3(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out = np.concatenate(
            (branch_1x1, branch_1x3_3x1, branch_3x3, branch_pool), axis=1
        )
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_1x1 = d_out[:, : self.out_1x1, ...]
        d_out_1x3_3x1 = d_out[:, self.out_1x1 : sum(self.out_1x3_3x1), ...]
        d_out_3x3 = d_out[
            :,
            self.out_1x1
            + sum(self.out_1x3_3x1) : self.out_1x1
            + sum(self.out_1x3_3x1)
            + sum(self.out_1x3_3x1_after),
            ...,
        ]
        d_out_pool = d_out[:, -self.out_pool :, ...]

        dX_1x1 = self.branch_1x1.backward(d_out_1x1)
        dX_1x3_3x1 = self.branch_1x3_3x1.backward(d_out_1x3_3x1)
        dX_3x3 = self.branch_3x3.backward(d_out_3x3)
        dX_pool = self.branch_pool.backward(d_out_pool)

        self.dX = dX_1x1 + dX_1x3_3x1 + dX_3x3 + dX_pool
        return self.dX

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        return (
            batch_size,
            self.out_1x1
            + sum(self.out_1x3_3x1)
            + sum(self.out_1x3_3x1_after)
            + self.out_pool,
            height,
            width,
        )


class _Incep_V2_Redux(Sequential):
    def __init__(
        self,
        in_channels: int,
        red_3x3: int,
        out_3x3: int,
        red_3x3_db: int,
        out_3x3_db: Tuple[int, int],
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_3x3 = out_3x3
        self.out_3x3_db = out_3x3_db

        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "red_3x3": ("0<,+inf", int),
                "out_3x3": ("0<,+inf", int),
                "red_3x3_db": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        self.branch_3x3 = Sequential(
            Conv2D(in_channels, red_3x3, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_3x3, out_3x3, 3, 2, "valid", **basic_args),
            BatchNorm2D(out_3x3, momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_3x3_db = Sequential(
            Conv2D(in_channels, red_3x3_db, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3_db, momentum) if do_batch_norm else None,
            activation(),
            Conv2D(red_3x3_db, out_3x3_db[0], 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3_db[0], momentum) if do_batch_norm else None,
            activation(),
            Conv2D(out_3x3_db[0], out_3x3_db[1], 3, 2, "valid", **basic_args),
            BatchNorm2D(out_3x3_db[1], momentum) if do_batch_norm else None,
            activation(),
        )
        self.branch_pool = Sequential(
            Pool2D(3, 2, "max", "valid"),
        )

        super(_Incep_V2_Redux, self).__init__()
        self.extend(
            self.branch_3x3,
            self.branch_3x3_db,
            self.branch_pool,
            deep_add=True,
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        branch_3x3 = self.branch_3x3(X, is_train)
        branch_3x3_db = self.branch_3x3_db(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out = np.concatenate((branch_3x3, branch_3x3_db, branch_pool), axis=1)
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_3x3 = d_out[:, : self.out_3x3, ...]
        d_out_3x3_db = d_out[:, self.out_3x3 : self.out_3x3 + self.out_3x3_db[1], ...]
        d_out_pool = d_out[:, -self.in_channels :, ...]

        dX_3x3 = self.branch_3x3.backward(d_out_3x3)
        dX_3x3_db = self.branch_3x3_db.backward(d_out_3x3_db)
        dX_pool = self.branch_pool.backward(d_out_pool)

        self.dX = dX_3x3 + dX_3x3_db + dX_pool
        return self.dX

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _, _ = in_shape
        red_h, red_w = self.branch_pool.out_shape(in_shape)[2:]

        return (
            batch_size,
            self.out_3x3 + self.out_3x3_db[1] + channels,
            red_h,
            red_w,
        )
