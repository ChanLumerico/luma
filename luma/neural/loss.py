import numpy as np

from luma.interface.typing import Matrix
from luma.neural.base import Loss


__all__ = ("SoftmaxLoss", "CrossEntropyLoss", "MSELoss")


class SoftmaxLoss(Loss):
    """
    Combines the softmax activation and cross-entropy loss into
    a single class, which is commonly used for multi-class
    classification problems. This loss function is particularly
    useful when the classes are mutually exclusive.

    This class assumes the true and predicted target values
    are one-hot encoded.
    """

    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        m = y_true.shape[0]
        y_pred = self._clip(y_pred)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        m = y_true.shape[0]
        y_pred = self._clip(y_pred)
        grad = (y_pred - y_true) / m
        return grad


class CrossEntropyLoss(Loss):
    """
    Computes the cross entropy loss between true labels and
    predicted probabilities, which is widely used for `binary`
    classification problems.

    This class assumes the true and predicted target values
    are one-hot encoded.
    """

    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        m = y_true.shape[0]
        y_pred = self._clip(y_pred)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss /= m
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        m = y_true.shape[0]
        y_pred = self._clip(y_pred)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / m


class MSELoss(Loss):
    """
    Calculates the mean squared error between the predicted values
    and the true values, typically used for regression problems.
    Mean squared error is a common measure of the accuracy of a regression
    model and provides a simple way to quantify the difference between
    predicted and actual values.

    Notes
    -----
    - This class is for the loss of neural networks, which differs from
        that of statistical ML models.

    - For statistical models, please refer to
        `luma.metric.regression.MeanSquaredError`.

    """

    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        m = y_true.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / m
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        m = y_true.shape[0]
        return 2 * (y_pred - y_true) / m
