from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _Entry(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_Entry, self).__init__(
            graph={
                self.rt_: [self.res_1, self.dsc_1],
                self.res_1: [self.sum_1],
                self.dsc_1: [self.sum_1],
                self.sum_1: [self.res_2, self.dsc_2],
                self.res_2: [self.sum_2],
                self.dsc_2: [self.sum_2],
                self.sum_2: [self.res_3, self.dsc_3],
                self.res_3: [self.sum_3],
                self.dsc_3: [self.sum_3],
            },
            root=self.rt_,
            term=self.sum_3,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(
            Sequential(
                Conv2D(3, 32, 3, 2, **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 64, 3, **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
            ),
            name="rt_",
        )

        self.res_1 = LayerNode(
            Sequential(
                Conv2D(64, 128, 1, 2, **self.basic_args),
                BatchNorm2D(128, self.momentum),
            ),
            name="res_1",
        )
        self.dsc_1 = LayerNode(
            Sequential(
                DepthConv2D(64, 3, **self.basic_args),
                Conv2D(64, 128, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
                DepthConv2D(128, 3, **self.basic_args),
                Conv2D(128, 128, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                Pool2D(3, 2, "max"),
            ),
            name="dsc_1",
        )
        self.sum_1 = LayerNode(Identity(), MergeMode.SUM, name="sum_1")

        self.res_2 = LayerNode(
            Sequential(
                Conv2D(128, 256, 1, 2, **self.basic_args),
                BatchNorm2D(256, self.momentum),
            ),
            name="res_2",
        )
        self.dsc_2 = LayerNode(
            Sequential(
                self.activation(),
                DepthConv2D(128, 3, **self.basic_args),
                Conv2D(128, 256, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                DepthConv2D(256, 3, **self.basic_args),
                Conv2D(256, 256, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                Pool2D(3, 2, "max"),
            ),
            name="dsc_2",
        )
        self.sum_2 = LayerNode(Identity(), MergeMode.SUM, name="sum_2")

        self.res_3 = LayerNode(
            Sequential(
                Conv2D(265, 728, 1, 2, **self.basic_args),
                BatchNorm2D(728, self.momentum),
            ),
            name="res_3",
        )
        self.dsc_3 = LayerNode(
            Sequential(
                self.activation(),
                DepthConv2D(256, 3, **self.basic_args),
                Conv2D(256, 728, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(728, self.momentum),
                self.activation(),
                DepthConv2D(728, 3, **self.basic_args),
                Conv2D(728, 728, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(728, self.momentum),
                Pool2D(3, 2, "max"),
            ),
            name="dsc_3",
        )
        self.sum_3 = LayerNode(Identity(), MergeMode.SUM, name="sum_3")

    @Tensor.force_shape((-1, 3, 299, 299))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 728, 19, 19))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 728, 19, 19


class _Middle(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_Middle, self).__init__(
            graph={
                self.rt_: [self.sum_, self.dsc_],
                self.dsc_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.dsc_ = LayerNode(
            Sequential(
                self.activation(),
                DepthConv2D(728, 3, **self.basic_args),
                Conv2D(728, 728, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(728, self.momentum),
                self.activation(),
                DepthConv2D(728, 3, **self.basic_args),
                Conv2D(728, 728, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(728, self.momentum),
                self.activation(),
                DepthConv2D(728, 3, **self.basic_args),
                Conv2D(728, 728, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(728, self.momentum),
            ),
            name="dsc_",
        )
        self.sum_ = LayerNode(Identity(), MergeMode.SUM, name="sum_")

    @Tensor.force_shape((-1, 728, 19, 19))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 728, 19, 19))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 728, 19, 19


class _Exit(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_Exit, self).__init__(
            graph={
                self.rt_: [self.sum_, self.dsc_],
                self.dsc_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.dsc_ = LayerNode(
            Sequential(
                self.activation(),
                DepthConv2D(728, 3, **self.basic_args),
                Conv2D(728, 728, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(728, self.momentum),
                self.activation(),
                DepthConv2D(728, 3, **self.basic_args),
                Conv2D(728, 1024, 1, 1, "valid", **self.basic_args),
                BatchNorm2D(1024, self.momentum),
                Pool2D(3, 2, "max"),
            ),
            name="dsc_",
        )
        self.sum_ = LayerNode(Identity(), MergeMode.SUM, name="sum_")

    @Tensor.force_shape((-1, 728, 19, 19))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1024, 9, 9))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1024, 9, 9
