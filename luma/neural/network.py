from typing import Self
import numpy as np

from luma.core.super import (
    Estimator,
    Optimizer,
    Evaluator,
    Supervised,
    NeuralModel,
)

from luma.interface.typing import TensorLike, Tensor, Matrix, Vector
from luma.interface.util import InitUtil, Clone, TrainProgress

from luma.model_selection.split import TrainTestSplit, BatchGenerator

from luma.neural.base import Loss
from luma.neural.layer import Sequential, Dense, Dropout, Activation
from luma.neural.optimizer import SGDOptimizer
from luma.neural.loss import CrossEntropy


__all__ = ("MLP", "SimpleCNN")  # TODO: Future implementations: SimpleCNN, RNN, ...


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
    `initializer` : Type of weight initializer (default `None`)
    `activation` : Type of activation function (default `ReLU`)
    `out_activation` : Type of activation function for the last layer
    (only applied in prediction, default `Softmax`)
    `optimizer` : An optimizer used in weight update process
    (default `SGDOptimizer`)
    `loss` : Type of loss function (default `CrossEntropy`)
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
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        initializer: InitUtil.InitStr = None,
        activation: Activation.FuncType = Activation.ReLU(),
        out_activation: Activation.FuncType = Activation.Softmax(),
        optimizer: Optimizer = SGDOptimizer(),
        loss: Loss = CrossEntropy(),
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
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.valid_size = valid_size
        self.initializer = initializer
        self.activation = activation
        self.out_activation = out_activation
        self.optimizer = optimizer
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.lambda_ = lambda_
        self.early_stopping = early_stopping
        self.patience = patience
        self.shuffle = shuffle
        self.random_state = random_state
        self.fitted_ = False

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
        self.feature_shapes_ = [
            (i, j)
            for i, j in zip(
                self.feature_sizes_[:-1],
                self.feature_sizes_[1:],
            )
        ]

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

    def fit(self, X: Matrix, y: Matrix) -> Self:
        X_train, X_val, y_train, y_val = TrainTestSplit(
            X,
            y,
            test_size=self.valid_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        ).get

        best_valid_loss = np.inf
        epochs_no_improve = 0

        train_prog = TrainProgress(n_epochs=self.n_epochs)
        with train_prog.progress as prog:
            train_prog.add_task(prog, self)

            for epoch in range(self.n_epochs):
                train_loss = self.train(X_train, y_train)
                train_loss_avg = np.average(train_loss)

                valid_loss = self.eval(X_val, y_val)
                valid_loss_avg = np.average(valid_loss)

                self.train_loss_.append(train_loss_avg)
                self.valid_loss_.append(valid_loss_avg)
                train_prog.update(
                    prog,
                    epoch,
                    epochs_no_improve,
                    [train_loss_avg, valid_loss_avg],
                )

                if valid_loss_avg < best_valid_loss:
                    best_valid_loss = valid_loss_avg
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if self.early_stopping and epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        self.fitted_ = True
        return self

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

    def train(self, X: TensorLike, y: TensorLike) -> list[float]:
        train_loss = []
        for X_batch, y_batch in BatchGenerator(
            X, y, batch_size=self.batch_size, shuffle=self.shuffle
        ):
            out = self.model(X_batch, is_train=True)
            loss = self.loss.loss(y_batch, out)
            d_out = self.loss.grad(y_batch, out)

            train_loss.append(loss)
            self.running_loss_.append(loss)

            self.model.backward(d_out)
            self.model.update()

        return train_loss

    def eval(self, X: TensorLike, y: TensorLike) -> list[float]:
        valid_loss = []
        for X_batch, y_batch in BatchGenerator(
            X, y, batch_size=self.batch_size, shuffle=self.shuffle
        ):
            out = self.model(X_batch, is_train=False)
            loss = self.loss.loss(y_batch, out)
            valid_loss.append(loss)

        return valid_loss

    def predict(self, X: Matrix, argmax: bool = True) -> Vector:
        out = self.model(X, is_train=False)
        y_pred = self.out_activation.forward(out)

        return np.argmax(y_pred, axis=1) if argmax else y_pred

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator, argmax: bool = True
    ) -> float:
        y_pred = self.predict(X, argmax=argmax)
        return metric.score(y_true=y, y_pred=y_pred)


class SimpleCNN(Estimator, Supervised, NeuralModel): ...
