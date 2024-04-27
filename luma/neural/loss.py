import numpy as np

from luma.interface.typing import Matrix
from luma.neural.base import Loss


__all__ = ("CrossEntropy", "BinaryCrossEntropy", "MSELoss")


class CrossEntropy(Loss):
    """
    Combines the softmax activation and cross-entropy loss into
    a single class, which is commonly used for multi-class
    classification problems. This loss function is particularly
    useful when the classes are mutually exclusive.

    This class assumes the true and predicted target values
    are one-hot encoded.
    """

    def __init__(self):
        super().__init__()

    def _softmax(self, y_pred: Matrix) -> Matrix:
        exp_shift = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        softmax_out = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)

        return softmax_out

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        softmax_probs = self._softmax(y_pred)
        clipped_probs = np.clip(softmax_probs, self.epsilon, 1.0)

        loss = -np.sum(y_true * np.log(clipped_probs)) / y_true.shape[0]
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        softmax_probs = self._softmax(y_pred)
        grad = (softmax_probs - y_true) / y_true.shape[0]
        return grad


class BinaryCrossEntropy(Loss):
    """
    Computes the cross entropy loss between true labels and
    predicted probabilities, which is widely used for binary
    classification problems.

    This class assumes the true and predicted target values
    are one-hot encoded.
    """

    def __init__(self) -> None:
        super().__init__()

    def _sigmoid(self, z: Matrix) -> Matrix:
        return 1 / (1 + np.exp(-z))

    def loss(self, y_true: Matrix, y_predZ: Matrix) -> float:
        m = y_true.shape[0]
        y_pred = self._sigmoid(y_predZ)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss /= m
        return loss

    def grad(self, y_true: Matrix, y_predZ: Matrix) -> Matrix:
        m = y_true.shape[0]
        y_pred = self._sigmoid(y_predZ)
        return (y_pred - y_true) / m


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
