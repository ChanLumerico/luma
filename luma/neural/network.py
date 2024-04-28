from typing import Self
from tqdm import tqdm
import numpy as np

from luma.core.super import (
    Estimator,
    Optimizer,
    Evaluator,
    Supervised,
    NeuralModel,
)

from luma.interface.typing import TensorLike, Tensor, Matrix, Vector
from luma.interface.util import InitUtil, Clone

from luma.preprocessing.encoder import OneHotEncoder
from luma.model_selection.split import TrainTestSplit, BatchGenerator

from luma.neural.base import Loss
from luma.neural.layer import Sequential, Dense, Dropout, Activation
from luma.neural.optimizer import SGDOptimizer
from luma.neural.loss import CrossEntropy


__all__ = "MLP"  # TODO: Future implementations: CNN, RNN, ...


class MLP(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: list[int] | int,
        batch_size: int = 100,
        epochs: int = 100,
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
        self.epochs = epochs
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
                "epochs": ("0<,+inf", int),
                "learning_rate": ("0<,+inf", None),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,{self.epochs}", int),
            }
        )
        self.check_param_ranges()
        self._build_model()

    def fit(self, X: Matrix, y: Matrix | Vector) -> Self:
        if len(y.shape) == 1:
            y = self._one_hot_encode_y(y)

        X_train, X_val, y_train, y_val = TrainTestSplit(
            X,
            y,
            shuffle=self.shuffle,
            random_state=self.random_state,
        ).get

        with tqdm(
            total=self.epochs,
            desc="Training",
            unit="epoch",
            ncols=100,
        ) as pbar:
            for _ in range(self.epochs):
                train_loss = self.train(X_train, y_train)
                train_loss_avg = np.average(train_loss)

                valid_loss = self.eval(X_val, y_val)
                valid_loss_avg = np.average(valid_loss)

                self.train_loss_.append(train_loss_avg)
                self.valid_loss_.append(valid_loss_avg)

                pbar.set_description(
                    f"(Train/Valid Loss: {train_loss_avg:.4f}/{valid_loss_avg:.4f})"
                )
                pbar.update(1)

        self.fitted_ = True
        return self

    def _one_hot_encode_y(self, y: Vector) -> Matrix:
        oh = OneHotEncoder()
        return oh.fit_transform(y.reshape(-1, 1))

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

    def predict(self, X: Matrix) -> Vector:
        out = self.model(X, is_train=False)
        y_pred = self.out_activation.forward(out)

        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X: Matrix) -> Matrix:
        out = self.model(X, is_train=False)
        y_pred = self.out_activation.forward(out)

        return y_pred

    def score(self, X: Matrix, y: Matrix, metric: Evaluator) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
