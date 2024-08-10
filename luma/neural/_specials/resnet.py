from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import LayerLike
from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph


class _Basic(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: Activation.FuncType = Activation.ReLU,
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
                self.rt_: [self.sum_, self.conv_],
                self.conv_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.conv_ = LayerNode(
            Sequential(
                Convolution2D(
                    self.in_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args
                ),
                BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                Convolution2D(
                    self.out_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args
                ),
                BatchNorm2D(self.out_channels, self.momentum),
            ),
            name="conv_",
        )
        self.sum_ = LayerNode(
            self.activation(),
            merge_mode="sum",
            name="sum_",
        )

        ...
