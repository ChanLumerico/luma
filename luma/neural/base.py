from abc import ABC, abstractmethod
from typing import Self
import numpy as np
import time

from luma.core.base import ModelBase, NeuralBase
from luma.core.super import Evaluator

from luma.interface.exception import NotFittedError
from luma.interface.typing import Matrix, Tensor, TensorLike
from luma.interface.util import InitUtil, TrainProgress
from luma.model_selection.split import BatchGenerator, TrainTestSplit


__all__ = ("Layer", "Loss", "Initializer", "NeuralModel")


class Layer(ABC, ModelBase):
    """
    An internal class for layers in neural networks.

    Neural network layers are composed of interconnected nodes,
    each performing computations on input data. Common types include
    fully connected, convolutional, and recurrent layers, each
    serving distinct roles in learning from data.

    Attributes
    ----------
    - `weights_` : Weight tensor
    - `biases_` : Bias tensor
    - `dX` : Gradient w.r.t. the input
    - `dW` : Gradient w.r.t. the weights
    - `dB` : Gradient w.r.t. the biases
    - `optimizer` : Optimizer for certain layer
    - `out_shape` : Shape of the output when forwarding

    Properties
    ----------
    To get its parameter size (weights, biases):
    ```py
    (property) param_size: Tuple[int, int]
    ```
    """

    def __init__(self) -> None:
        self.input_: TensorLike = None
        self.output_: TensorLike = None

        self.weights_: TensorLike = None
        self.biases_: TensorLike = None

        self.dX: TensorLike = None
        self.dW: TensorLike = None
        self.dB: TensorLike = None

        self.optimizer: object = None

    @abstractmethod
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike: ...

    @abstractmethod
    def backward(self, d_out: TensorLike) -> TensorLike: ...

    def update(self) -> None:
        if self.optimizer is None:
            return
        weights_, biases_ = self.optimizer.update(
            self.weights_, self.biases_, self.dW, self.dB
        )
        self.weights_ = Tensor(weights_)
        self.biases_ = Tensor(biases_)

    def init_params(self, w_shape: tuple, b_shape: tuple) -> None:
        init_type_: type = InitUtil(self.initializer).initializer_type

        if init_type_ is None:
            self.weights_ = 0.01 * self.rs_.randn(*w_shape)
        else:
            self.weights_ = init_type_(self.random_state).init_nd(*w_shape)

        self.biases_: TensorLike = np.zeros(b_shape)

    @property
    def param_size(self) -> tuple[int, int]:
        w_size, b_size = 0, 0
        w_list: list | list[TensorLike] = []
        b_list: list | list[TensorLike] = []

        if self.weights_ is not None:
            w_list.extend(self.weights_)
            for w in w_list:
                w_size += len(w.flatten())
        if self.biases_ is not None:
            b_list.extend(self.biases_)
            for b in b_list:
                b_size += len(b.flatten())

        return w_size, b_size

    @abstractmethod
    def out_shape(self, in_shape: tuple[int]) -> tuple[int]: ...

    def __call__(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return self.forward(X, is_train=is_train)

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        w_size, b_size = self.param_size
        return f"{type(self).__name__}({w_size + b_size} params)"


class Loss(ABC):
    """
    An internal class for loss functions used in neural networks.

    Loss functions, integral to the training process of machine
    learning models, serve as crucial metrics assessing the disparity
    between predicted outcomes and ground truth labels. They play a
    pivotal role in optimization algorithms, guiding parameter updates
    towards minimizing the discrepancy between predictions and true values.
    """

    def __init__(self) -> None:
        self.epsilon = 1e-12

    @abstractmethod
    def loss(self) -> float: ...

    @abstractmethod
    def grad(self) -> Matrix: ...

    def _clip(self, y: Matrix) -> Matrix:
        return np.clip(y, self.epsilon, 1 - self.epsilon)


class Initializer(ABC):
    """
    Abstract base class for initializing neural network weights.

    This class provides a structured way to implement weight
    initialization methods for different types of layers in a
    neural network.
    The class must be inherited by specific initializer implementations
    that define methods for 2D and 4D weight tensors.
    """

    def __init__(self, random_state: int) -> None:
        self.rs_ = np.random.RandomState(random_state)

    @classmethod
    def __class_alias__(cls) -> None: ...

    @abstractmethod
    def init_nd(self) -> TensorLike: ...


class NeuralModel(ABC, NeuralBase):
    """
    Neural networks are computational models inspired by the human brain,
    consisting of layers of interconnected nodes (neurons) that process
    information through weighted connections. These models include an input
    layer to receive data, hidden layers that perform computations, and an
    output layer to deliver results.
    """

    def __init__(
        self,
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        valid_size: float,
        early_stopping: bool,
        patience: int,
        deep_verbose: bool,
    ) -> None:
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.valid_size = valid_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.deep_verbose = deep_verbose

    def __init_model__(self) -> None:
        self.feature_sizes_: list = []
        self.feature_shapes_: list = []

        self.running_loss_: list[float] = []
        self.train_loss_: list[float] = []
        self.valid_loss_: list[float] = []

        self.model: object

    @abstractmethod
    def _build_model(self) -> None: ...

    def _get_feature_shapes(self, sizes: list) -> list[tuple]:
        return [(i, j) for i, j in zip(sizes[:-1], sizes[1:])]

    def fit_nn(self, X: TensorLike, y: TensorLike) -> Self:
        X_train, X_val, y_train, y_val = TrainTestSplit(
            X,
            y,
            test_size=self.valid_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        ).get

        best_valid_loss = np.inf
        epochs_no_improve = 0

        train_prog = TrainProgress(n_iter=self.n_epochs)
        with train_prog.progress as prog:
            train_prog.add_task(prog, self)

            for epoch in range(1, self.n_epochs + 1):
                train_loss = self.train(X_train, y_train, epoch=epoch)
                train_loss_avg = np.average(train_loss)

                valid_loss = self.eval(X_val, y_val)
                valid_loss_avg = np.average(valid_loss)

                self.train_loss_.append(train_loss_avg)
                self.valid_loss_.append(valid_loss_avg)
                train_prog.update(
                    prog,
                    epoch,
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

    def predict_nn(self, X: TensorLike, argmax: bool = True) -> TensorLike:
        if not self.fitted_:
            raise NotFittedError
        out = self.model(X, is_train=False)
        return np.argmax(out, axis=1) if argmax else out

    def score_nn(
        self, X: TensorLike, y: TensorLike, metric: Evaluator, argmax: bool = True
    ) -> float:
        y_pred = self.predict(X, argmax=argmax)
        return metric.score(y_true=y, y_pred=y_pred)

    def train(self, X: TensorLike, y: TensorLike, epoch: int) -> list[float]:
        train_loss = []
        train_batch = BatchGenerator(
            X, y, batch_size=self.batch_size, shuffle=self.shuffle
        )
        for i, (X_batch, y_batch) in enumerate(train_batch, start=1):
            t_start = time.time_ns()
            out = self.model(X_batch, is_train=True)
            loss = self.loss.loss(y_batch, out)
            d_out = self.loss.grad(y_batch, out)

            train_loss.append(loss)
            self.running_loss_.append(loss)
            self.model.backward(d_out)
            self.model.update()

            t_end = time.time_ns()
            if self.deep_verbose:
                print(
                    f"Epoch {epoch}/{self.n_epochs},",
                    f"Batch {i}/{train_batch.n_batches}",
                    f"- Loss: {loss}, Elapsed Time: {(t_end - t_start) / 1e9} s",
                )

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

    @property
    def param_size(self) -> tuple[int, int]:
        return self.model.param_size

    def summarize(self, in_shape: tuple[int]) -> None:
        title = f"Summary of '{str(self)}'"
        print(f"{title:^83}")
        print("-" * 83)
        print(
            "{:<20} {:<20} {:<20} {:<20}".format(
                "Name", "Layer/Block", "Output Shape", "Weights, Biases"
            )
        )
        print("=" * 83)

        n_layers = 0
        w_size, b_size = self.param_size
        for name, layer in self.model:
            n_layers += 1
            print(
                f"{name:<20}",
                f"{str(layer):<20}",
                f"{str(layer.out_shape(in_shape)):<20}",
                f"{str(layer.param_size):<20}",
            )
            in_shape = layer.out_shape(in_shape)
        print("=" * 83)
        print(f"Total Layers/Blocks: {n_layers}")
        print(
            f"Total Parameters: ({w_size:,} weights, {b_size:,} biases)",
            f"-> {w_size + b_size:,} params",
        )
        print("-" * 83)

    def __str__(self) -> str:
        return type(self).__name__
