from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph


class _IncepRes_V2_TypeA(LayerGraph):
    def __init__(
        self,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
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
        super(_IncepRes_V2_TypeA, self).__init__(
            graph={
                self.rt_: [self.res_sum, self.br_a, self.br_b, self.br_c],
                self.br_a: [self.br_cat],
                self.br_b: [self.br_cat],
                self.br_c: [self.br_cat],
                self.br_cat: [self.res_sum],
            },
            root=self.rt_,
            term=self.res_sum,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.res_sum = LayerNode(
            Sequential(Identity(), self.activation()),
            merge_mode="sum",
            name="res_sum",
        )

        self.br_a = LayerNode(
            Sequential(
                Convolution2D(384, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Convolution2D(384, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Convolution2D(32, 32, 3, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Convolution2D(384, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Convolution2D(32, 48, 3, 1, "same", **self.basic_args),
                BatchNorm2D(48, self.momentum),
                self.activation(),
                Convolution2D(48, 64, 3, 1, "same", **self.basic_args),
                BatchNorm2D(48, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )

        self.br_cat = LayerNode(
            Sequential(
                Convolution2D(128, 384, 1, 1, "same", **self.basic_args),
                BatchNorm2D(384, self.momentum),
            ),
            merge_mode="chcat",
            name="br_cat",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 384, 35, 35


class _IncepRes_V2_TypeB(LayerGraph):
    def __init__(
        self,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
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
        super(_IncepRes_V2_TypeB, self).__init__(
            graph={
                self.rt_: [self.res_sum, self.br_a, self.br_b],
                self.br_a: [self.br_cat],
                self.br_b: [self.br_cat],
                self.br_cat: [self.res_sum],
            },
            root=self.rt_,
            term=self.res_sum,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.res_sum = LayerNode(
            Sequential(Identity(), self.activation()),
            merge_mode="sum",
            name="res_sum",
        )

        self.br_a = LayerNode(
            Sequential(
                Convolution2D(1154, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Convolution2D(1154, 128, 1, 1, "same", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
                Convolution2D(128, 160, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(160, self.momentum),
                self.activation(),
                Convolution2D(160, 192, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_cat = LayerNode(
            Sequential(
                Convolution2D(384, 1154, 1, 1, "same", **self.basic_args),
                BatchNorm2D(1154, self.momentum),
            ),
            merge_mode="chcat",
            name="br_cat",
        )
    
    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1154, 17, 17


class _IncepRes_V2_TypeC(LayerGraph):
    def __init__(
        self,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
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
        super(_IncepRes_V2_TypeC, self).__init__(
            graph={
                self.rt_: [self.res_sum, self.br_a, self.br_b],
                self.br_a: [self.br_cat],
                self.br_b: [self.br_cat],
                self.br_cat: [self.res_sum],
            },
            root=self.rt_,
            term=self.res_sum,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)
    
    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.res_sum = LayerNode(
            Sequential(Identity(), self.activation()),
            merge_mode="sum",
            name="res_sum",
        )

        self.br_a = LayerNode(
            Sequential(
                Convolution2D(2048, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Convolution2D(2048, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
                Convolution2D(192, 234, (1, 3), 1, "same", **self.basic_args),
                BatchNorm2D(224, self.momentum),
                self.activation(),
                Convolution1D(224, 256, (3, 1), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_cat = LayerNode(
            Sequential(
                Convolution2D(448, 2048, 1, 1, "same", **self.basic_args),
                BatchNorm2D(2048, self.momentum),
            ),
            merge_mode="chcat",
            name="br_cat",
        )
    
    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 2048, 8, 8


class _IncepRes_V2_Redux(LayerGraph): ...
