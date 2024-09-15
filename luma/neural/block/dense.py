from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _Composite(Sequential):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        momentum: float = 0.9,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        inter_channels = bn_size * growth_rate
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_Composite, self).__init__(
            BatchNorm2D(in_channels, momentum),
            activation(),
            Conv2D(in_channels, inter_channels, 1, 1, "valid", **basic_args),
            BatchNorm2D(inter_channels, momentum),
            activation(),
            Conv2D(inter_channels, growth_rate, 3, 1, "same", **basic_args),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _DenseUnit(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        n_layers: int,
        growth_rate: int,
        bn_size: int = 4,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "momentum": momentum,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_DenseUnit, self).__init__(
            graph={
                self.comp_nodes[i]: self.comp_nodes[i + 1 :] for i in range(n_layers)
            },
            root=self.comp_nodes[0],
            term=self.comp_nodes[-1],
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.comp_nodes = [None for _ in range(self.n_layers)]

        for i in range(self.n_layers):
            self.comp_nodes[i] = LayerNode(
                _Composite(
                    self.in_channels + i * self.growth_rate,
                    self.growth_rate,
                    self.bn_size,
                    self.activation,
                    **self.basic_args,
                ),
                MergeMode.CHCAT,
                name=f"comp_{i + 1}",
            )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, in_channels, height, width = in_shape
        return (
            batch_size,
            in_channels + self.n_layers * self.growth_rate,
            height,
            width,
        )


class _Transition(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        compression: float = 1.0,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        momentum: float = 0.9,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        out_channels = int(out_channels * compression)
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_Transition, self).__init__(
            BatchNorm2D(in_channels, momentum),
            activation(),
            Conv2D(in_channels, out_channels, 1, 1, "valid", **basic_args),
            Pool2D(2, 2, "avg"),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)
