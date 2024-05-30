from typing import Any, Self, override
from dataclasses import asdict

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from metric.classification import Accuracy

from luma.neural import loss
from luma.neural.base import Loss, NeuralModel
from luma.neural.block import (
    InceptionBlock,
    InceptionBlockV2A,
    InceptionBlockV2B,
    InceptionBlockV2C,
    InceptionBlockV2R,
    InceptionBlockArgs,
)
from luma.neural.layer import (
    Convolution2D,
    Pooling2D,
    GlobalAvgPooling2D,
    Activation,
    Dropout,
    Dense,
    Flatten,
    Sequential,
)


__all__ = ("_Inception_V1", "_Inception_V2", "_Inception_V3")


class _Inception_V1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.ReLU,
        loss: Loss = loss.CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
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
        self.optimizer = optimizer
        self.loss = loss
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
            learning_rate,
            valid_size,
            early_stopping,
            patience,
            deep_verbose,
        )
        super().__init_model__()
        self.model = Sequential()
        self.optimizer.set_params(learning_rate=self.learning_rate)
        self.model.set_optimizer(optimizer=self.optimizer)

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
                "learning_rate": ("0<,+inf", None),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self._build_model()

    def _build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "optimizer": self.optimizer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        incep_args = InceptionBlockArgs(
            activation=self.activation,
            do_batch_norm=False,
            **base_args,
        )

        self.model.extend(
            Convolution2D(3, 64, 7, 2, 3, **base_args),
            self.activation(),
            Pooling2D(3, 2, "max", "same"),
        )

        self.model.extend(
            Convolution2D(64, 64, 1, 1, "valid", **base_args),
            self.activation(),
            Convolution2D(64, 192, 3, 1, "valid", **base_args),
            self.activation(),
            Pooling2D(3, 2, "max", "same"),
        )

        self.model.extend(
            (
                "Inception_3a",
                InceptionBlock(192, 64, 96, 128, 16, 32, 32, **asdict(incep_args)),
            ),
            (
                "Inception_3b",
                InceptionBlock(256, 128, 128, 192, 32, 96, 64, **asdict(incep_args)),
            ),
            Pooling2D(3, 2, "max", "same"),
            deep_add=False,
        )

        self.model.extend(
            (
                "Inception_4a",
                InceptionBlock(480, 192, 96, 208, 16, 48, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4b",
                InceptionBlock(512, 160, 112, 224, 24, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4c",
                InceptionBlock(512, 128, 128, 256, 24, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4d",
                InceptionBlock(512, 112, 144, 288, 32, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4e",
                InceptionBlock(528, 256, 160, 320, 32, 128, 128, **asdict(incep_args)),
            ),
            Pooling2D(3, 2, "max", "same"),
            deep_add=False,
        )

        self.model.extend(
            (
                "Inception_5a",
                InceptionBlock(832, 256, 160, 320, 32, 128, 128, **asdict(incep_args)),
            ),
            (
                "Inception_5b",
                InceptionBlock(832, 384, 192, 384, 48, 128, 128, **asdict(incep_args)),
            ),
            GlobalAvgPooling2D(),
            Dropout(self.dropout_rate, self.random_state),
            deep_add=False,
        )

        self.model += Flatten()
        self.model += Dense(1024, self.out_features, **base_args)

    @Tensor.force_dim(4)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_Inception_V1, self).fit_nn(X, y)

    @override
    @Tensor.force_dim(4)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Inception_V1, self).predict_nn(X, argmax)

    @override
    @Tensor.force_dim(4)
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
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.ReLU,
        loss: Loss = loss.CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
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
        self.optimizer = optimizer
        self.loss = loss
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
            learning_rate,
            valid_size,
            early_stopping,
            patience,
            deep_verbose,
        )
        super().__init_model__()
        self.model = Sequential()
        self.optimizer.set_params(learning_rate=self.learning_rate)
        self.model.set_optimizer(optimizer=self.optimizer)

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
                "learning_rate": ("0<,+inf", None),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self._build_model()

    def _build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "optimizer": self.optimizer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        incep_args = InceptionBlockArgs(
            activation=self.activation,
            do_batch_norm=False,
            **base_args,
        )

        self.model.extend(
            Convolution2D(3, 32, 3, 2, "valid", **base_args),
            self.activation(),
            Convolution2D(32, 32, 3, 1, "valid", **base_args),
            self.activation(),
            Convolution2D(32, 64, 3, 1, "same", **base_args),
            self.activation(),
            Pooling2D(3, 2, "max", "valid"),
        )

        self.model.extend(
            Convolution2D(64, 80, 3, 1, "valid", **base_args),
            self.activation(),
            Convolution2D(80, 192, 3, 2, "valid", **base_args),
            self.activation(),
            Convolution2D(192, 288, 3, 1, "same", **base_args),
            self.activation(),
        )

        inception_3xA = [
            InceptionBlockV2A(288, 64, 48, 64, 64, (96, 96), 64, **asdict(incep_args))
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
                InceptionBlockV2R(288, 64, 384, 64, (96, 96), **asdict(incep_args)),
            )
        )

        inception_5xB = [
            InceptionBlockV2B(
                768, 192, 128, 192, 128, (128, 192), 192, **asdict(incep_args)
            ),
            InceptionBlockV2B(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            InceptionBlockV2B(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            InceptionBlockV2B(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            InceptionBlockV2B(
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
                InceptionBlockV2R(768, 192, 320, 192, (192, 192), **asdict(incep_args)),
            ),
        )

        inception_C_args = [320, 384, (384, 384), 448, 384, (384, 384), 192]
        inception_2xC = [
            InceptionBlockV2C(1280, *inception_C_args, **asdict(incep_args)),
            InceptionBlockV2C(2048, *inception_C_args, **asdict(incep_args)),
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

        self.model.add(GlobalAvgPooling2D())
        self.model.add(Flatten())
        self.model.extend(
            Dropout(self.dropout_rate, self.random_state),
            Dense(2048, self.out_features, **base_args),
        )

    @Tensor.force_dim(4)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_Inception_V2, self).fit_nn(X, y)

    @override
    @Tensor.force_dim(4)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Inception_V2, self).predict_nn(X, argmax)

    @override
    @Tensor.force_dim(4)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Inception_V2, self).score_nn(X, y, metric, argmax)


class _Inception_V3(Estimator, Supervised, NeuralModel):
    NotImplemented
