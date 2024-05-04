from typing import Literal, Self, override

from luma.core.super import Estimator, Optimizer, Evaluator, Supervised

from luma.interface.typing import Tensor, Matrix, Vector
from luma.interface.util import InitUtil, Clone

from luma.neural.base import NeuralModel, Loss
from luma.neural.layer import Sequential, Dense, Dropout, Activation, Flatten
from luma.neural.block import ConvBlock, DenseBlock


__all__ = ("MLP", "CNN")


class MLP(Estimator, Supervised, NeuralModel):
    """
    An MLP (Multilayer Perceptron) is a type of artificial neural network
    composed of at least three layers: an input layer, one or more hidden
    layers, and an output layer. Each layer consists of nodes, or neurons,
    which are fully connected to the neurons in the next layer. MLPs use a
    technique called backpropagation for learning, where the output error
    is propagated backwards through the network to update the weights.
    They are capable of modeling complex nonlinear relationships between
    inputs and outputs. MLPs are commonly used for tasks like classification,
    regression, and pattern recognition.

    Structure
    ---------
    ```py
    (Dense -> Activation -> Dropout) -> ... -> Dense
    ```
    Parameters
    ----------
    `in_features` : Number of input features
    `out_features` : Number of output features
    `hidden_layers` : Numbers of the features in hidden layers
    (`int` for a single layer)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `initializer` : Type of weight initializer
    `activation` : Type of activation function
    `optimizer` : An optimizer used in weight update process
    `loss` : Type of loss function
    `dropout_rate` : Dropout rate
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whethter to shuffle the data at the beginning of every epoch

    Notes
    -----
    - If the data or the target is a 1D-Array(`Vector`), reshape it into a
        higher dimensional array.

    - For classification tasks, the target vector `y` must be
        one-hot encoded.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: list[int] | int,
        *,
        activation: Activation.FuncType,
        optimizer: Optimizer,
        loss: Loss,
        initializer: InitUtil.InitStr = None,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        valid_size: float = 0.1,
        dropout_rate: float = 0.5,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.lambda_ = lambda_
        self.shuffle = shuffle
        self.random_state = random_state
        self.fitted_ = False

        super().__init__(
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            early_stopping,
            patience,
        )
        super().__init_model__()
        self.model = Sequential()
        self.optimizer.set_params(learning_rate=self.learning_rate)
        self.model.set_optimizer(optimizer=self.optimizer)

        if isinstance(self.hidden_layers, int):
            self.hidden_layers = [self.hidden_layers]

        self.feature_sizes_ = [
            self.in_features,
            *self.hidden_layers,
            self.out_features,
        ]
        self.feature_shapes_ = self._get_feature_shapes(self.feature_sizes_)

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "learning_rate": ("0<,+inf", None),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,{self.n_epochs}", int),
            }
        )
        self.check_param_ranges()
        self._build_model()

    def _build_model(self) -> None:
        for i, (in_, out_) in enumerate(self.feature_shapes_):
            self.model += Dense(
                in_,
                out_,
                initializer=self.initializer,
                lambda_=self.lambda_,
                random_state=self.random_state,
            )
            if i < len(self.feature_shapes_) - 1:
                self.model += Clone(self.activation).get
                self.model += Dropout(
                    dropout_rate=self.dropout_rate,
                    random_state=self.random_state,
                )

    def fit(self, X: Matrix, y: Matrix) -> Self:
        return super(MLP, self).fit_nn(X, y)

    @override
    def predict(self, X: Matrix, argmax: bool = True) -> Matrix | Vector:
        return super(MLP, self).predict_nn(X, argmax)

    @override
    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator, argmax: bool = True
    ) -> float:
        return super(MLP, self).score_nn(X, y, metric, argmax)


class CNN(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        in_channels_list: list[int] | int,
        in_features_list: list[int] | int,
        out_channels: int,
        out_features: int,
        filter_size: int,
        *,
        activation: Activation.FuncType,
        optimizer: Optimizer,
        loss: Loss,
        initializer: InitUtil.InitStr = None,
        padding: Literal["same", "valid"] = "same",
        stride: int = 1,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        do_dropout: bool = True,
        dropout_rate: float = 0.5,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
    ) -> None:
        self.in_channels_list = in_channels_list
        self.in_features_list = in_features_list
        self.out_channels = out_channels
        self.out_features = out_features
        self.filter_size = filter_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.initializer = initializer
        self.padding = padding
        self.stride = stride
        self.do_pooling = do_pooling
        self.pool_filter_size = pool_filter_size
        self.pool_stride = pool_stride
        self.pool_mode = pool_mode
        self.do_dropout = do_dropout
        self.dropout_rate = dropout_rate
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
        )
        super().__init_model__()
        self.model = Sequential()
        self.optimizer.set_params(learning_rate=self.learning_rate)
        self.model.set_optimizer(optimizer=self.optimizer)

        if isinstance(self.in_channels_list, int):
            self.in_channels_list = [self.in_channels_list]
        if isinstance(self.in_features_list, int):
            self.in_features_list = [self.in_features_list]

        self.feature_sizes_ = [
            [*self.in_channels_list, self.out_channels],
            [*self.in_features_list, self.out_features],
        ]
        self.feature_shapes_ = [
            [*self._get_feature_shapes(self.feature_sizes_[0])],
            [*self._get_feature_shapes(self.feature_sizes_[1])],
        ]

        self.set_param_ranges(
            {
                "out_channels": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
                "dropout_rate": ("0,1", None),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "learning_rate": ("0<,+inf", None),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,{self.n_epochs}", int),
            }
        )
        self.check_param_ranges()
        self._build_model()

    def _build_model(self) -> None:
        for in_, out_ in self.feature_shapes_[0]:
            self.model += ConvBlock(
                in_,
                out_,
                self.filter_size,
                activation=Clone(self.activation).get,
                initializer=self.initializer,
                padding=self.padding,
                stride=self.stride,
                lambda_=self.lambda_,
                do_pooling=self.do_pooling,
                pool_filter_size=self.pool_filter_size,
                pool_stride=self.pool_stride,
                pool_mode=self.pool_mode,
                random_state=self.random_state,
            )

        self.model += Flatten()
        for i, (in_, out_) in enumerate(self.feature_shapes_[1]):
            if i < len(self.feature_shapes_[1]) - 1:
                self.model += DenseBlock(
                    in_,
                    out_,
                    activation=Clone(self.activation).get,
                    lambda_=self.lambda_,
                    do_dropout=self.do_dropout,
                    dropout_rate=self.dropout_rate,
                    random_state=self.random_state,
                )
            else:
                self.model += Dense(
                    in_,
                    out_,
                    lambda_=self.lambda_,
                    random_state=self.random_state,
                )

    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(CNN, self).fit_nn(X, y)

    @override
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(CNN, self).predict_nn(X, argmax)

    @override
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator, argmax: bool = True
    ) -> float:
        return super(CNN, self).score_nn(X, y, metric, argmax)
