from typing import Any, Self, override, ClassVar
from dataclasses import asdict

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy
from luma.preprocessing.encoder import LabelSmoothing

from luma.neural.base import NeuralModel
from luma.neural.block import (
    BaseBlockArgs,
    IncepBlock,
    IncepResBlock,
    XceptionBlock,
    SeparableConv2D,
)
from luma.neural.layer import (
    Conv2D,
    Pool2D,
    GlobalAvgPool2D,
    BatchNorm2D,
    Activation,
    Dropout,
    Dense,
    Flatten,
    Sequential,
)


__all__ = (
    "_Inception_V1",
    "_Inception_V2",
    "_Inception_V3",
    "_Inception_V4",
    "_InceptionRes_V1",
    "_InceptionRes_V2",
    "_Xception",
)


class _Inception_V1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = [
            [3, 64, 64, 192],
            [192, 256, 480, 512, 512, 512, 528, 832, 832],
            [1024, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        incep_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=False,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )

        self.model.extend(
            Conv2D(64, 64, 1, 1, "valid", **base_args),
            self.activation(),
            Conv2D(64, 192, 3, 1, "valid", **base_args),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )

        self.model.extend(
            (
                "Inception_3a",
                IncepBlock.V1(192, 64, 96, 128, 16, 32, 32, **asdict(incep_args)),
            ),
            (
                "Inception_3b",
                IncepBlock.V1(256, 128, 128, 192, 32, 96, 64, **asdict(incep_args)),
            ),
            Pool2D(3, 2, "max", "same"),
            deep_add=False,
        )

        self.model.extend(
            (
                "Inception_4a",
                IncepBlock.V1(480, 192, 96, 208, 16, 48, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4b",
                IncepBlock.V1(512, 160, 112, 224, 24, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4c",
                IncepBlock.V1(512, 128, 128, 256, 24, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4d",
                IncepBlock.V1(512, 112, 144, 288, 32, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4e",
                IncepBlock.V1(528, 256, 160, 320, 32, 128, 128, **asdict(incep_args)),
            ),
            Pool2D(3, 2, "max", "same"),
            deep_add=False,
        )

        self.model.extend(
            (
                "Inception_5a",
                IncepBlock.V1(832, 256, 160, 320, 32, 128, 128, **asdict(incep_args)),
            ),
            (
                "Inception_5b",
                IncepBlock.V1(832, 384, 192, 384, 48, 128, 128, **asdict(incep_args)),
            ),
            GlobalAvgPool2D(),
            Dropout(self.dropout_rate, self.random_state),
            deep_add=False,
        )

        self.model += Flatten()
        self.model += Dense(1024, self.out_features, **base_args)

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_Inception_V1, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Inception_V1, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Inception_V1, self).score_nn(X, y, metric, argmax)


class _Inception_V2(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = [
            [3, 32, 32, 64, 64, 80, 192, 288],
            [288, 288, 288, 768],
            [768, 768, 768, 768, 768, 1280],
            [1280, 2048, 2048],
            [2048, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        incep_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=False,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 32, 3, 2, "valid", **base_args),
            self.activation(),
            Conv2D(32, 32, 3, 1, "valid", **base_args),
            self.activation(),
            Conv2D(32, 64, 3, 1, "same", **base_args),
            self.activation(),
            Pool2D(3, 2, "max", "valid"),
        )

        self.model.extend(
            Conv2D(64, 80, 3, 1, "valid", **base_args),
            self.activation(),
            Conv2D(80, 192, 3, 2, "valid", **base_args),
            self.activation(),
            Conv2D(192, 288, 3, 1, "same", **base_args),
            self.activation(),
        )

        inception_3xA = [
            IncepBlock.V2_TypeA(288, 64, 48, 64, 64, (96, 96), 64, **asdict(incep_args))
            for _ in range(3)
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    ["Inception_3a", "Inception_3b", "Inception_3c"], inception_3xA
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx1",
                IncepBlock.V2_Redux(288, 64, 384, 64, (96, 96), **asdict(incep_args)),
            )
        )

        inception_5xB = [
            IncepBlock.V2_TypeB(
                768, 192, 128, 192, 128, (128, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 192, 192, 192, (192, 192), 192, **asdict(incep_args)
            ),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_4a",
                        "Inception_4b",
                        "Inception_4c",
                        "Inception_4d",
                        "Inception_4e",
                    ],
                    inception_5xB,
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx2",
                IncepBlock.V2_Redux(
                    768, 192, 320, 192, (192, 192), **asdict(incep_args)
                ),
            ),
        )

        inception_C_args = [320, 384, (384, 384), 448, 384, (384, 384), 192]
        inception_2xC = [
            IncepBlock.V2_TypeC(1280, *inception_C_args, **asdict(incep_args)),
            IncepBlock.V2_TypeC(2048, *inception_C_args, **asdict(incep_args)),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_5a",
                        "Inception_5b",
                    ],
                    inception_2xC,
                )
            ],
            deep_add=False,
        )

        self.model.add(GlobalAvgPool2D())
        self.model.add(Flatten())
        self.model.extend(
            Dropout(self.dropout_rate, self.random_state),
            Dense(2048, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_Inception_V2, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Inception_V2, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Inception_V2, self).score_nn(X, y, metric, argmax)


class _Inception_V3(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        optimizer: Optimizer,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = [
            [3, 32, 32, 64, 64, 80, 192, 288],
            [288, 288, 288, 768],
            [768, 768, 768, 768, 768, 1280],
            [1280, 2048, 2048],
            [2048, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "smoothing": ("0,1", None),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        incep_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 32, 3, 2, "valid", **base_args),
            BatchNorm2D(32),
            self.activation(),
            Conv2D(32, 32, 3, 1, "valid", **base_args),
            BatchNorm2D(32),
            self.activation(),
            Conv2D(32, 64, 3, 1, "same", **base_args),
            BatchNorm2D(64),
            self.activation(),
            Pool2D(3, 2, "max", "valid"),
        )

        self.model.extend(
            Conv2D(64, 80, 3, 1, "valid", **base_args),
            BatchNorm2D(80),
            self.activation(),
            Conv2D(80, 192, 3, 2, "valid", **base_args),
            BatchNorm2D(192),
            self.activation(),
            Conv2D(192, 288, 3, 1, "same", **base_args),
            BatchNorm2D(288),
            self.activation(),
        )

        inception_3xA = [
            IncepBlock.V2_TypeA(288, 64, 48, 64, 64, (96, 96), 64, **asdict(incep_args))
            for _ in range(3)
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    ["Inception_3a", "Inception_3b", "Inception_3c"], inception_3xA
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx1",
                IncepBlock.V2_Redux(288, 64, 384, 64, (96, 96), **asdict(incep_args)),
            )
        )

        inception_5xB = [
            IncepBlock.V2_TypeB(
                768, 192, 128, 192, 128, (128, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            IncepBlock.V2_TypeB(
                768, 192, 192, 192, 192, (192, 192), 192, **asdict(incep_args)
            ),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_4a",
                        "Inception_4b",
                        "Inception_4c",
                        "Inception_4d",
                        "Inception_4e",
                    ],
                    inception_5xB,
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx2",
                IncepBlock.V2_Redux(
                    768, 192, 320, 192, (192, 192), **asdict(incep_args)
                ),
            ),
        )

        inception_C_args = [320, 384, (384, 384), 448, 384, (384, 384), 192]
        inception_2xC = [
            IncepBlock.V2_TypeC(1280, *inception_C_args, **asdict(incep_args)),
            IncepBlock.V2_TypeC(2048, *inception_C_args, **asdict(incep_args)),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_5a",
                        "Inception_5b",
                    ],
                    inception_2xC,
                )
            ],
            deep_add=False,
        )

        self.model.add(GlobalAvgPool2D())
        self.model.add(Flatten())
        self.model.extend(
            Dropout(self.dropout_rate, self.random_state),
            Dense(2048, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_Inception_V3, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Inception_V3, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Inception_V3, self).score_nn(X, y, metric, argmax)


class _Inception_V4(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.8,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "smoothing": ("0,1", None),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        incep_args = BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model.add(
            ("Stem", IncepBlock.V4_Stem(**asdict(incep_args))),
        )
        for i in range(1, 5):
            self.model.add(
                (f"Inception_A{i}", IncepBlock.V4_TypeA(**asdict(incep_args))),
            )
        self.model.add(
            (
                "Inception_RA",
                IncepBlock.V4_ReduxA(384, (192, 224, 256, 384), **asdict(incep_args)),
            )
        )
        for i in range(1, 8):
            self.model.add(
                (f"Inception_B{i}", IncepBlock.V4_TypeB(**asdict(incep_args))),
            )
        self.model.add(
            ("Inception_RB", IncepBlock.V4_ReduxB(**asdict(incep_args))),
        )
        for i in range(1, 4):
            self.model.add(
                (f"Inception_C{i}", IncepBlock.V4_TypeC(**asdict(incep_args))),
            )

        self.model.extend(
            GlobalAvgPool2D(),
            Flatten(),
            Dropout(self.dropout_rate, self.random_state),
            Dense(1536, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_Inception_V4, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Inception_V4, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Inception_V4, self).score_nn(X, y, metric, argmax)


class _InceptionRes_V1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.8,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "smoothing": ("0,1", None),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        incep_args = BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model.add(
            ("Stem", IncepResBlock.V1_Stem(**asdict(incep_args))),
        )
        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_A{i}", IncepResBlock.V1_TypeA(**asdict(incep_args)))
            )
        self.model.add(
            (
                "IncepRes_RA",
                IncepBlock.V4_ReduxA(256, (192, 192, 256, 384), **asdict(incep_args)),
            )
        )
        for i in range(1, 11):
            self.model.add(
                (f"IncepRes_B{i}", IncepResBlock.V1_TypeB(**asdict(incep_args)))
            )
        self.model.add(("IncepRes_RB", IncepResBlock.V1_Redux(**asdict(incep_args))))

        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_C{i}", IncepResBlock.V1_TypeC(**asdict(incep_args)))
            )

        self.model.extend(
            GlobalAvgPool2D(),
            Flatten(),
            Dropout(self.dropout_rate, self.random_state),
            Dense(1792, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_InceptionRes_V1, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_InceptionRes_V1, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_InceptionRes_V1, self).score_nn(X, y, metric, argmax)


class _InceptionRes_V2(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.8,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "smoothing": ("0,1", None),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        incep_args = BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model.add(
            ("Stem", IncepBlock.V4_Stem(**asdict(incep_args))),
        )
        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_A{i}", IncepResBlock.V2_TypeA(**asdict(incep_args))),
            )
        self.model.add(
            (
                "IncepRes_RA",
                IncepBlock.V4_ReduxA(384, (256, 256, 384, 384), **asdict(incep_args)),
            ),
        )

        for i in range(1, 11):
            self.model.add(
                (f"IncepRes_B{i}", IncepResBlock.V2_TypeB(**asdict(incep_args))),
            )
        self.model.add(
            ("IncepRes_RB", IncepResBlock.V2_Redux(**asdict(incep_args))),
        )

        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_C{i}", IncepResBlock.V2_TypeC(**asdict(incep_args))),
            )

        self.model.extend(
            GlobalAvgPool2D(),
            Flatten(),
            Dropout(self.dropout_rate, self.random_state),
            Dense(2272, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_InceptionRes_V2, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_InceptionRes_V2, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_InceptionRes_V2, self).score_nn(X, y, metric, argmax)


class _Xception(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }

        self.model.add(("EntryFlow", XceptionBlock.Entry(**base_args)))
        for i in range(1, 9):
            self.model.add(
                (f"MiddleFlow_{i}", XceptionBlock.Middle(**base_args)),
            )

        self.model.extend(
            ("ExitFlow", XceptionBlock.Exit(**base_args)),
            SeparableConv2D(1024, 1536, 3, **base_args),
            BatchNorm2D(1536, self.momentum),
            self.activation(),
            SeparableConv2D(1536, 2048, 3, **base_args),
            BatchNorm2D(2048, self.momentum),
            self.activation(),
            GlobalAvgPool2D(),
            deep_add=False,
        )

        self.model += Flatten()
        self.model += Dense(2048, self.out_features, **base_args)

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_Xception, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Xception, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Xception, self).score_nn(X, y, metric, argmax)
