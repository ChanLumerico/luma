import numpy as np

from luma.interface.util import Matrix, Loss


__all__ = "CategoricalCrossEntropy"


class CategoricalCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-8

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        grad = (y_pred - y_true) / m
        return grad
