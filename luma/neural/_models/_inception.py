from typing import Self, override
from dataclasses import asdict

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from metric.classification import Accuracy

from luma.neural.base import Loss, NeuralModel
from luma.neural.loss import CrossEntropy
from luma.neural.block import (
    InceptionBlock,
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
        loss: Loss = CrossEntropy(),
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
                "patience": (f"0<,+inf", int),
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
            Convolution2D(3, 64, filter_size=7, stride=2, padding=3, **base_args),
            self.activation(),
            Pooling2D(3, stride=2, mode="max", padding="same"),
        )

        self.model.extend(
            Convolution2D(64, 64, filter_size=1, padding="valid", **base_args),
            self.activation(),
            Convolution2D(64, 192, filter_size=3, padding="valid", **base_args),
            self.activation(),
            Pooling2D(3, stride=2, mode="max", padding="same"),
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
            Pooling2D(3, stride=2, mode="max", padding="same"),
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
            Pooling2D(3, stride=2, mode="max", padding="same"),
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
            Dropout(self.dropout_rate, random_state=self.random_state),
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
    NotImplemented


class _Inception_V3(Estimator, Supervised, NeuralModel):
    NotImplemented
