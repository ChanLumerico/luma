from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _Incep_V4_Stem(LayerGraph):
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
        super(_Incep_V4_Stem, self).__init__(
            graph={
                self.rt_seq: [self.br1_l, self.br1_r],
                self.br1_l: [self.br1_cat],
                self.br1_r: [self.br1_cat],
                self.br1_cat: [self.br2_l, self.br2_r],
                self.br2_l: [self.br2_cat],
                self.br2_r: [self.br2_cat],
                self.br2_cat: [self.br3_l, self.br3_r],
                self.br3_l: [self.br3_cat],
                self.br3_r: [self.br3_cat],
            },
            root=self.rt_seq,
            term=self.br3_cat,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_seq = LayerNode(
            Sequential(
                Conv2D(3, 32, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 32, 3, 1, "valid", **self.basic_args),
                BatchNorm2D(32, self.momentum),
                self.activation(),
                Conv2D(32, 64, 3, 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
            ),
            name="rt_seq",
        )

        self.br1_l = LayerNode(Pool2D(3, 2, "max", "valid"), name="br1_l")
        self.br1_r = LayerNode(
            Sequential(
                Conv2D(64, 96, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br1_r",
        )
        self.br1_cat = LayerNode(Identity(), MergeMode.CHCAT, name="br1_cat")

        self.br2_l = LayerNode(
            Sequential(
                Conv2D(160, 64, 1, 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
                Conv2D(64, 96, 3, 1, "valid", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br2_l",
        )
        self.br2_r = LayerNode(
            Sequential(
                Conv2D(160, 64, 1, 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
                Conv2D(64, 64, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
                Conv2D(64, 64, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
                Conv2D(64, 96, 3, 1, "valid", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br2_r",
        )
        self.br2_cat = LayerNode(Identity(), MergeMode.CHCAT, name="br2_cat")

        self.br3_l = LayerNode(
            Sequential(
                Conv2D(192, 192, 3, 1, "valid", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br3_l",
        )
        self.br3_r = LayerNode(Pool2D(2, 2, "max", "valid"), name="br3_r")
        self.br3_cat = LayerNode(Identity(), MergeMode.CHCAT, name="br3_cat")

    @Tensor.force_shape((-1, 3, 299, 299))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 384, 35, 35))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 384, 35, 35


class _Incep_V4_TypeA(LayerGraph):
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
        super(_Incep_V4_TypeA, self).__init__(
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

        self.br_a = LayerNode(
            Sequential(
                Pool2D(2, 2, "avg", "same"),
                Conv2D(384, 96, 1, 1, "same", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br1_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(384, 96, 1, 1, "same", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br1_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Conv2D(384, 64, 1, 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
                Conv2D(64, 96, 3, 1, "same", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br1_c",
        )
        self.br_d = LayerNode(
            Sequential(
                Conv2D(384, 64, 1, 1, "same", **self.basic_args),
                BatchNorm2D(64, self.momentum),
                self.activation(),
                Conv2D(64, 96, 3, 1, "same", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
                Conv2D(96, 96, 3, 1, "same", **self.basic_args),
                BatchNorm2D(96, self.momentum),
                self.activation(),
            ),
            name="br_d",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

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


class _Incep_V4_TypeB(LayerGraph):
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
        super(_Incep_V4_TypeB, self).__init__(
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

        self.br_a = LayerNode(
            Sequential(
                Pool2D(2, 2, "avg", "same"),
                Conv2D(1024, 128, 1, 1, "same", **self.basic_args),
                BatchNorm2D(128, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(1024, 384, 1, 1, "same", **self.basic_args),
                BatchNorm2D(384, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Conv2D(1024, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
                Conv2D(192, 224, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(224, self.momentum),
                self.activation(),
                Conv2D(224, 256, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )
        self.br_d = LayerNode(
            Sequential(
                Conv2D(1024, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
                Conv2D(192, 192, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
                Conv2D(192, 224, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(224, self.momentum),
                self.activation(),
                Conv2D(224, 224, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(224, self.momentum),
                self.activation(),
                Conv2D(224, 256, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_d",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

    @Tensor.force_shape((-1, 1024, 17, 17))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1024, 17, 17))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1024, 17, 17


class _Incep_V4_TypeC(LayerGraph):
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
        super(_Incep_V4_TypeC, self).__init__(
            graph={
                self.rt_: [self.br_a, self.br_b, self.br_c, self.br_d],
                self.br_a: [self.cat_],
                self.br_b: [self.cat_],
                self.br_c: [self.br_cl, self.br_cr],
                self.br_d: [self.br_dl, self.br_dr],
                self.br_cl: [self.cat_],
                self.br_cr: [self.cat_],
                self.br_dl: [self.cat_],
                self.br_dr: [self.cat_],
            },
            root=self.rt_,
            term=self.cat_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")

        self.br_a = LayerNode(
            Sequential(
                Pool2D(2, 2, "avg", "same"),
                Conv2D(1536, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_a",
        )
        self.br_b = LayerNode(
            Sequential(
                Conv2D(1536, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )

        self.br_c = LayerNode(
            Sequential(
                Conv2D(1536, 384, 1, 1, "same", **self.basic_args),
                BatchNorm2D(384, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )
        self.br_cl = LayerNode(
            Sequential(
                Conv2D(384, 256, (1, 3), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_cl",
        )
        self.br_cr = LayerNode(
            Sequential(
                Conv2D(384, 256, (3, 1), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_cr",
        )

        self.br_d = LayerNode(
            Sequential(
                Conv2D(1536, 384, 1, 1, "same", **self.basic_args),
                BatchNorm2D(384, self.momentum),
                self.activation(),
                Conv2D(384, 448, (1, 3), 1, "same", **self.basic_args),
                BatchNorm2D(448, self.momentum),
                self.activation(),
                Conv2D(448, 512, (3, 1), 1, "same", **self.basic_args),
                BatchNorm2D(512, self.momentum),
                self.activation(),
            ),
            name="br_d",
        )
        self.br_dl = LayerNode(
            Sequential(
                Conv2D(512, 256, (3, 1), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_dl",
        )
        self.br_dr = LayerNode(
            Sequential(
                Conv2D(512, 256, (1, 3), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
            ),
            name="br_dr",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

    @Tensor.force_shape((-1, 1536, 8, 8))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1536, 8, 8))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1536, 8, 8


class _Incep_V4_ReduxA(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels_arr: tuple[int, int, int, int],
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

        self.in_channels = in_channels
        self.k_, self.l_, self.m_, self.n_ = out_channels_arr

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.init_nodes()
        super(_Incep_V4_ReduxA, self).__init__(
            graph={
                self.rt_: [self.br_a, self.br_b, self.br_c],
                self.br_a: [self.cat_],
                self.br_b: [self.cat_],
                self.br_c: [self.cat_],
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
                Conv2D(self.in_channels, self.n_, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(self.n_, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Conv2D(self.in_channels, self.k_, 1, 1, "same", **self.basic_args),
                BatchNorm2D(self.k_, self.momentum),
                self.activation(),
                Conv2D(self.k_, self.l_, 3, 1, "same", **self.basic_args),
                BatchNorm2D(self.l_, self.momentum),
                self.activation(),
                Conv2D(self.l_, self.m_, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(self.m_, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

    @Tensor.force_shape((-1, 384, 35, 35))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1024, 17, 17))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, channels, _, _ = in_shape
        _, _, height, width = self.br_a.out_shape(in_shape)

        return batch_size, (channels + self.n_ + self.m_), height, width


class _Incep_V4_ReduxB(LayerGraph):
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
        super(_Incep_V4_ReduxB, self).__init__(
            graph={
                self.rt_: [self.br_a, self.br_b, self.br_c],
                self.br_a: [self.cat_],
                self.br_b: [self.cat_],
                self.br_c: [self.cat_],
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
                Conv2D(1024, 192, 1, 1, "same", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
                Conv2D(192, 192, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(192, self.momentum),
                self.activation(),
            ),
            name="br_b",
        )
        self.br_c = LayerNode(
            Sequential(
                Conv2D(1024, 256, 1, 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 256, (1, 7), 1, "same", **self.basic_args),
                BatchNorm2D(256, self.momentum),
                self.activation(),
                Conv2D(256, 320, (7, 1), 1, "same", **self.basic_args),
                BatchNorm2D(320, self.momentum),
                self.activation(),
                Conv2D(320, 320, 3, 2, "valid", **self.basic_args),
                BatchNorm2D(320, self.momentum),
                self.activation(),
            ),
            name="br_c",
        )

        self.cat_ = LayerNode(Identity(), MergeMode.CHCAT, name="cat_")

    @Tensor.force_shape((-1, 1024, 17, 17))
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_shape((-1, 1536, 8, 8))
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, _, _ = in_shape
        return batch_size, 1536, 8, 8
