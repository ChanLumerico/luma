from typing import List, Self
import numpy as np

from luma.core.super import Estimator, Optimizer, Supervised
from luma.interface.typing import TensorLike, Tensor, Matrix
from luma.interface.util import InitUtil

from luma.neural.base import Loss
from luma.neural.layer import Sequential, Dense, Dropout, Activation
from luma.neural.optimizer import SGDOptimizer
from luma.neural.loss import SoftmaxLoss


__all__ = "MLP"


class MLP(Estimator, Estimator.NeuralNet, Supervised):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int] | int,
        batch_size: int = 100,
        epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        initializer: InitUtil.InitType = None,
        activation: Activation.FuncType = Activation.ReLU(),
        optimizer: Optimizer = SGDOptimizer(),
        loss: Loss = SoftmaxLoss(),
        dropout_rate: float = 0.5,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model = Sequential()

        # TODO: Implement further

    def fit(self, *args) -> Self:
        return super().fit(*args)

    def predict(self, *args) -> Matrix:
        return super().predict(*args)


MLP()
