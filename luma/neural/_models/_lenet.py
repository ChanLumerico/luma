from typing import Self, override

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import Clone, InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import Loss, NeuralModel
from luma.neural.block import ConvBlock, DenseBlock
from luma.neural.layer import Activation, Dense, Flatten, Sequential
from luma.neural.loss import CrossEntropy


__all__ = ("_LeNet_1", "_LeNet_4", "_LeNet_5")


class _LeNet_1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.Tanh(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
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
        self.lambda_ = lambda_
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
            [1, 4, 8],
            [8 * 4 * 4, 10],
        ]
        self.feature_shapes_ = [
            [(1, 4), (4, 8)],
            [(8 * 4 * 4, 10)],
        ]

        self.set_param_ranges(
            {
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
        self.model += ConvBlock(
            1,
            4,
            filter_size=5,
            stride=1,
            activation=Clone(self.activation).get,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += ConvBlock(
            4,
            8,
            filter_size=5,
            stride=1,
            activation=Clone(self.activation).get,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += Flatten()
        self.model += Dense(
            8 * 4 * 4,
            10,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_LeNet_1, self).fit_nn(X, y)

    @override
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_LeNet_1, self).predict_nn(X, argmax)

    @override
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
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.Tanh(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
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
            [1, 4, 16],
            [16 * 5 * 5, 120, 10],
        ]
        self.feature_shapes_ = [
            [(1, 4), (4, 16)],
            [(16 * 5 * 5, 120), (120, 10)],
        ]

        self.set_param_ranges(
            {
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
        self.model += ConvBlock(
            1,
            4,
            filter_size=5,
            stride=1,
            activation=Clone(self.activation).get,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += ConvBlock(
            4,
            16,
            filter_size=5,
            stride=1,
            activation=Clone(self.activation).get,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += Flatten()
        self.model += DenseBlock(
            16 * 5 * 5,
            120,
            activation=Clone(self.activation).get,
            lambda_=self.lambda_,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )
        self.model += Dense(
            in_features=120,
            out_features=10,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_LeNet_4, self).fit_nn(X, y)

    @override
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_LeNet_4, self).predict_nn(X, argmax)

    @override
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
        optimizer: Optimizer,
        activation: Activation.FuncType = Activation.Tanh(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
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
            [1, 6, 16],
            [16 * 5 * 5, 120, 84, 10],
        ]
        self.feature_shapes_ = [
            [(1, 6), (6, 16)],
            [(16 * 5 * 5, 120), (120, 84), (84, 10)],
        ]

        self.set_param_ranges(
            {
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
        self.model += ConvBlock(
            1,
            6,
            filter_size=5,
            stride=1,
            activation=Clone(self.activation).get,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += ConvBlock(
            6,
            16,
            filter_size=5,
            stride=1,
            activation=Clone(self.activation).get,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += Flatten()
        self.model += DenseBlock(
            16 * 5 * 5,
            120,
            activation=Clone(self.activation).get,
            lambda_=self.lambda_,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )
        self.model += DenseBlock(
            120,
            84,
            activation=Clone(self.activation).get,
            lambda_=self.lambda_,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )
        self.model += Dense(
            in_features=84,
            out_features=10,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_LeNet_5, self).fit_nn(X, y)

    @override
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_LeNet_5, self).predict_nn(X, argmax)

    @override
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_LeNet_5, self).score_nn(X, y, metric, argmax)
