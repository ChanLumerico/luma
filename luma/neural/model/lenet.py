from typing import Self, override, ClassVar

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.block import ConvBlock2D, DenseBlock
from luma.neural.layer import Activation, Dense, Flatten, Sequential


__all__ = ("_LeNet_1", "_LeNet_4", "_LeNet_5")


class _LeNet_1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
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
            [1, 4, 8],
            [8 * 4 * 4, self.out_features],
        ]
        self.feature_shapes_ = [
            [(1, 4), (4, 8)],
            [(8 * 4 * 4, self.out_features)],
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        self.model += ConvBlock2D(
            1,
            4,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += ConvBlock2D(
            4,
            8,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += Flatten()
        self.model += Dense(
            8 * 4 * 4,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 1, 28, 28)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_LeNet_1, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_LeNet_1, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_LeNet_1, self).score_nn(X, y, metric, argmax)


class _LeNet_4(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
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
            [1, 4, 16],
            [16 * 5 * 5, 120, self.out_features],
        ]
        self.feature_shapes_ = [
            [(1, 4), (4, 16)],
            [(16 * 5 * 5, 120), (120, self.out_features)],
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        self.model += ConvBlock2D(
            1,
            4,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += ConvBlock2D(
            4,
            16,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += Flatten()
        self.model += DenseBlock(
            16 * 5 * 5,
            120,
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_dropout=False,
            random_state=self.random_state,
        )
        self.model += Dense(
            120,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 1, 32, 32)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_LeNet_4, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_LeNet_4, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_LeNet_4, self).score_nn(X, y, metric, argmax)


class _LeNet_5(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
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
            [1, 6, 16],
            [16 * 5 * 5, 120, 84, self.out_features],
        ]
        self.feature_shapes_ = [
            [(1, 6), (6, 16)],
            [(16 * 5 * 5, 120), (120, 84), (84, self.out_features)],
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
        self.build_model()

    def build_model(self) -> None:
        self.model += ConvBlock2D(
            1,
            6,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += ConvBlock2D(
            6,
            16,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += Flatten()
        self.model += DenseBlock(
            16 * 5 * 5,
            120,
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_dropout=False,
            random_state=self.random_state,
        )
        self.model += DenseBlock(
            120,
            84,
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_dropout=False,
            random_state=self.random_state,
        )
        self.model += Dense(
            84,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 1, 32, 32)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_LeNet_5, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_LeNet_5, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_LeNet_5, self).score_nn(X, y, metric, argmax)
