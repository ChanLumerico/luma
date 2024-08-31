from typing import Tuple, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike, LayerLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _Basic(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_Basic, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 1

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.conv_ = LayerNode(
            Sequential(
                Conv2D(
                    self.in_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args,
                ),
                BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                Conv2D(
                    self.out_channels,
                    self.out_channels * _Basic.expansion,
                    3,
                    **self.basic_args,
                ),
                BatchNorm2D(
                    self.out_channels * _Basic.expansion,
                    self.momentum,
                ),
            ),
            name="conv_",
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else Identity(),
            name="down_",
        )
        self.sum_ = LayerNode(
            self.activation(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)


class _Bottleneck(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_Bottleneck, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.conv_ = LayerNode(
            Sequential(
                Conv2D(self.in_channels, self.out_channels, 1, **self.basic_args),
                BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                Conv2D(
                    self.out_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args,
                ),
                BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                Conv2D(
                    self.out_channels,
                    self.out_channels * _Bottleneck.expansion,
                    1,
                    **self.basic_args,
                ),
                BatchNorm2D(
                    self.out_channels * _Bottleneck.expansion,
                    self.momentum,
                ),
            ),
            name="conv_",
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else Identity(),
            name="down_",
        )
        self.sum_ = LayerNode(
            self.activation(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)


class _PreActBottleneck(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_PreActBottleneck, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.conv_ = LayerNode(
            Sequential(
                BatchNorm2D(self.in_channels, self.momentum),
                self.activation(),
                Conv2D(
                    self.in_channels,
                    self.out_channels,
                    1,
                    **self.basic_args,
                ),
                BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                Conv2D(
                    self.out_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args,
                ),
                BatchNorm2D(self.out_channels),
                self.activation(),
                Conv2D(
                    self.out_channels,
                    self.out_channels * _PreActBottleneck.expansion,
                    1,
                    **self.basic_args,
                ),
            )
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else Identity(),
            name="down_",
        )
        self.sum_ = LayerNode(
            Identity(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)
