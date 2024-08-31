from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _IncepRes_V1_Stem(Sequential):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer = None,
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
        _ = do_batch_norm

        self.set_param_ranges(
            {
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_IncepRes_V1_Stem, self).__init__(
            Conv2D(3, 32, 3, 2, "valid", **basic_args),
            BatchNorm2D(32, momentum),
            activation(),
            Conv2D(32, 32, 3, 1, "valid", **basic_args),
            BatchNorm2D(32, momentum),
            activation(),
            Conv2D(32, 64, 3, 1, "same", **basic_args),
            BatchNorm2D(64, momentum),
            activation(),
            Pool2D(3, 2, "max", "valid"),
        )
        self.extend(
            Conv2D(64, 80, 1, 1, "same", **basic_args),
            BatchNorm2D(80, momentum),
            activation(),
            Conv2D(80, 192, 3, 1, "valid", **basic_args),
            BatchNorm2D(192, momentum),
            activation(),
            Conv2D(192, 256, 3, 2, "valid", **basic_args),
            BatchNorm2D(256, momentum),
            activation(),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @Tensor.force_shape((-1, 3, 299, 299))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 256, 35, 35))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 256, 35, 35


class _IncepRes_V1_TypeA(LayerGraph):
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
        super(_IncepRes_V1_TypeA, self).__init__(
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
                Conv2D(256, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(256, 32, 1, 1, "same", **self.basic_args),
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
                Conv2D(256, 32, 1, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 32, 3, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 32, 3, 1, "same", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )

        self.br_cat = LayerNode(
            Sequential(
                Conv2D(96, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
            ),
            MergeMode.CHCAT,
            name="br_cat",
        )

    @Tensor.force_shape((-1, 256, 35, 35))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 256, 35, 35))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 256, 35, 35


class _IncepRes_V1_TypeB(LayerGraph):
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
        super(_IncepRes_V1_TypeB, self).__init__(
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
                Conv2D(896, 128, 1, 1, "same", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(896, 128, 1, 1, "same", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
                Conv2D(128, 128, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(128),
                self.activation(),
                Conv2D(128, 128, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_cat = LayerNode(
            Sequential(
                Conv2D(128, 896, 1, 1, "same", **self.basic_args),
                BatchNorm2D(896, self.momentum),
            ),
            MergeMode.CHCAT,
            name="br_cat",
        )

    @Tensor.force_shape((-1, 896, 17, 17))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 896, 17, 17))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 896, 17, 17


class _IncepRes_V1_TypeC(LayerGraph):
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
        super(_IncepRes_V1_TypeC, self).__init__(
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
                Conv2D(1792, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(1792, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192),
                self.activation(),
                Conv2D(192, 192, (1, 3), 1, "same", **self.basic_args),
                BatchNorm2D(192),
                self.activation(),
                Conv2D(192, 192, (3, 1), 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_cat = LayerNode(
            Sequential(
                Conv2D(384, 1792, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
            ),
            MergeMode.CHCAT,
            name="br_cat",
        )

    @Tensor.force_shape((-1, 1792, 8, 8))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1792, 8, 8))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1792, 8, 8


class _IncepRes_V1_Redux(LayerGraph):
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
        super(_IncepRes_V1_Redux, self).__init__(
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
                Conv2D(896, 256, 1, 1, "same", **self.basic_args),
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
                Conv2D(896, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256),
                self.activation(),
                Conv2D(256, 256, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )
        self.br_d = LayerNode(
            Sequential(
                Conv2D(896, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 256, 3, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 256, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_d",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

    @Tensor.force_shape((-1, 896, 17, 17))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1792, 8, 8))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1792, 8, 8
