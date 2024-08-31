from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _IncepRes_V2_TypeA(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
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
            MergeMode.SUM,
            name="res_sum",
        )

        self.br_a = LayerNode(
            Sequential(
                Conv2D(384, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(384, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 32, 3, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Conv2D(384, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 48, 3, 1, "same", **self.basic_args),
                BatchNorm2D(48, self.momentum),
                self.activation(),
                Conv2D(48, 64, 3, 1, "same", **self.basic_args),
                BatchNorm2D(48, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )

        self.br_cat = LayerNode(
            Sequential(
                Conv2D(128, 384, 1, 1, "same", **self.basic_args),
                BatchNorm2D(384, self.momentum),
            ),
            MergeMode.CHCAT,
            name="br_cat",
        )

    @Tensor.force_shape((-1, 384, 35, 35))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 384, 35, 35))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 384, 35, 35


class _IncepRes_V2_TypeB(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
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
            MergeMode.SUM,
            name="res_sum",
        )

        self.br_a = LayerNode(
            Sequential(
                Conv2D(1280, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(1280, 128, 1, 1, "same", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
                Conv2D(128, 160, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(160, self.momentum),
                self.activation(),
                Conv2D(160, 192, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_cat = LayerNode(
            Sequential(
                Conv2D(384, 1280, 1, 1, "same", **self.basic_args),
                BatchNorm2D(1280, self.momentum),
            ),
            MergeMode.CHCAT,
            name="br_cat",
        )

    @Tensor.force_shape((-1, 1280, 17, 17))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1280, 17, 17))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1280, 17, 17


class _IncepRes_V2_TypeC(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
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
            MergeMode.SUM,
            name="res_sum",
        )

        self.br_a = LayerNode(
            Sequential(
                Conv2D(2272, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(2272, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
                Conv2D(192, 234, (1, 3), 1, "same", **self.basic_args),
                BatchNorm2D(224, self.momentum),
                self.activation(),
                Conv2D(224, 256, (3, 1), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_cat = LayerNode(
            Sequential(
                Conv2D(448, 2272, 1, 1, "same", **self.basic_args),
                BatchNorm2D(2272, self.momentum),
            ),
            MergeMode.CHCAT,
            name="br_cat",
        )

    @Tensor.force_shape((-1, 2272, 8, 8))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 2272, 8, 8))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 2272, 8, 8


class _IncepRes_V2_Redux(LayerGraph):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
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
        super(_IncepRes_V2_Redux, self).__init__(
            graph={
                self.rt_: [self.br_a, self.br_b, self.br_c, self.br_d],
                self.br_a: [self.cat_],
                self.br_b: [self.cat_],
                self.br_c: [self.cat_],
                self.br_d: [self.cat_],
            },
            root=self.rt_,
            term=self.cat_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")

        self.br_a = LayerNode(Pool2D(3, 2, "max", "valid"), name="br_a")
        self.br_b = LayerNode(
            Sequential(
                Conv2D(1280, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 384, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(384, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Conv2D(1280, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 288, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(288, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )
        self.br_d = LayerNode(
            Sequential(
                Conv2D(1280, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 288, 3, 1, "same", **self.basic_args),
                BatchNorm2D(288, self.momentum),
                self.activation(),
                Conv2D(288, 320, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(320, self.momentum),
                self.activation(),
            ),
            name="br_d",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

    @Tensor.force_shape((-1, 1280, 17, 17))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 2272, 8, 8))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 2272, 8, 8
