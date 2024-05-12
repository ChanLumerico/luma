from typing import Self, override

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import Clone, InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import Loss, NeuralModel
from luma.neural.loss import CrossEntropy
from luma.neural.block import ConvBlock2D, DenseBlock
from luma.neural.layer import (
    Activation,
    Dense,
    Flatten,
    LocalResponseNorm,
    Sequential,
)


__all__ = ("_AlexNet", "_ZFNet")


class _AlexNet(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.ReLU(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
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
            [3, 96, 256, 384, 384, 256],
            [256 * 6 * 6, 4096, 4096, self.out_features],
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
        self.model += (
            "ConvBlock_1",
            ConvBlock2D(
                3,
                96,
                filter_size=11,
                stride=4,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="valid",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_1",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_2",
            ConvBlock2D(
                96,
                256,
                filter_size=5,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_2",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_3",
            ConvBlock2D(
                256,
                384,
                filter_size=3,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                do_pooling=False,
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_3",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_4",
            ConvBlock2D(
                384,
                384,
                filter_size=3,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                do_pooling=False,
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_4",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_5",
            ConvBlock2D(
                384,
                256,
                filter_size=3,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_5",
            LocalResponseNorm(depth=5),
        )

        self.model += Flatten()
        self.model += (
            "DenseBlock_1",
            DenseBlock(
                256 * 6 * 6,
                4096,
                activation=Clone(self.activation).get,
                lambda_=self.lambda_,
                do_batch_norm=False,
                dropout_rate=self.dropout_rate,
                random_state=self.random_state,
            ),
        )
        self.model += (
            "DenseBlock_2",
            DenseBlock(
                4096,
                4096,
                activation=Clone(self.activation).get,
                lambda_=self.lambda_,
                do_batch_norm=False,
                dropout_rate=self.dropout_rate,
                random_state=self.random_state,
            ),
        )
        self.model += Dense(
            4096,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_AlexNet, self).fit_nn(X, y)

    @override
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_AlexNet, self).predict_nn(X, argmax)

    @override
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_AlexNet, self).score_nn(X, y, metric, argmax)


class _ZFNet(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.ReLU(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
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
            [3, 96, 256, 384, 384, 256],
            [256 * 6 * 6, 4096, 4096, self.out_features],
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
        self.model += (
            "ConvBlock_1",
            ConvBlock2D(
                3,
                96,
                filter_size=7,
                stride=2,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="valid",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_1",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_2",
            ConvBlock2D(
                96,
                256,
                filter_size=5,
                stride=2,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_2",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_3",
            ConvBlock2D(
                256,
                384,
                filter_size=3,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                do_pooling=False,
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_3",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_4",
            ConvBlock2D(
                384,
                384,
                filter_size=3,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                do_pooling=False,
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_4",
            LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_5",
            ConvBlock2D(
                384,
                256,
                filter_size=3,
                stride=1,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_5",
            LocalResponseNorm(depth=5),
        )

        self.model += Flatten()
        self.model += (
            "DenseBlock_1",
            DenseBlock(
                256 * 6 * 6,
                4096,
                activation=Clone(self.activation).get,
                lambda_=self.lambda_,
                do_batch_norm=False,
                dropout_rate=self.dropout_rate,
                random_state=self.random_state,
            ),
        )
        self.model += (
            "DenseBlock_2",
            DenseBlock(
                4096,
                4096,
                activation=Clone(self.activation).get,
                lambda_=self.lambda_,
                do_batch_norm=False,
                dropout_rate=self.dropout_rate,
                random_state=self.random_state,
            ),
        )
        self.model += Dense(
            4096,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ZFNet, self).fit_nn(X, y)

    @override
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ZFNet, self).predict_nn(X, argmax)

    @override
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ZFNet, self).score_nn(X, y, metric, argmax)
