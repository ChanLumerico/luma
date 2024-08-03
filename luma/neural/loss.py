import numpy as np

from luma.interface.typing import Matrix
from luma.neural.base import Loss


__all__ = (
    "CrossEntropy",
    "BinaryCrossEntropy",
    "MSELoss",
    "HingeLoss",
    "HuberLoss",
    "KLDivergenceLoss",
    "NLLLoss",
)


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

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        m = y_true.shape[0]
        y_pred = self._sigmoid(y_pred)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss /= m
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        m = y_true.shape[0]
        y_pred = self._sigmoid(y_pred)
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


class HingeLoss(Loss):
    """
    Computes the hinge loss, typically used for "maximum-margin"
    classification problems like SVMs. Suitable for binary classification
    tasks where the true labels are expected to be -1 or 1.

    Notes
    -----
    - The hinge loss function penalizes the wrong predictions by a linear
      function rather than the squared loss.
    """

    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        y_true = np.where(y_true <= 0, -1, 1)
        loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
        return loss

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        y_true = np.where(y_true <= 0, -1, 1)
        grad = np.where(y_true * y_pred < 1, -y_true, 0)
        return grad / y_true.shape[0]


class HuberLoss(Loss):
    """
    Calculates the Huber loss, which is less sensitive to outliers
    than squared error loss. Useful in regression problems where
    robustness to outliers is desired.

    Parameters
    ----------
    `delta` : float
        The point where the loss function changes from a quadratic
        to linear.

    Notes
    -----
    - The Huber loss approaches MSE when delta is large, and MAE when
      delta is small.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        error = y_pred - y_true
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * (error**2)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)

        return np.mean(
            np.where(
                is_small_error,
                squared_loss,
                linear_loss,
            )
        )

    def grad(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        error = y_pred - y_true
        is_small_error = np.abs(error) <= self.delta

        grad = np.where(is_small_error, error, self.delta * np.sign(error))
        return grad / y_true.shape[0]


class KLDivergenceLoss(Loss):
    """
    Computes the Kullback-Leibler divergence loss, useful in scenarios
    where the true distribution is known and you want to measure how
    well the predicted distribution matches it.

    This class assumes both true and predicted distributions are
    probability distributions.

    Notes
    -----
    - Commonly used in Variational Autoencoders (VAEs) and other
      generative models.
    """

    def __init__(self) -> None:
        super().__init__()

    def _softmax(self, z: Matrix) -> Matrix:
        exp_shift = np.exp(z - np.max(z, axis=1, keepdims=True))
        softmax_out = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return softmax_out

    def loss(self, y_true: Matrix, z: Matrix) -> float:
        softmax_probs = self._softmax(z)
        clipped_probs = np.clip(softmax_probs, self.epsilon, 1.0)
        kl_div = np.sum(y_true * np.log(y_true / clipped_probs), axis=1)
        return np.mean(kl_div)

    def grad(self, y_true: Matrix, z: Matrix) -> Matrix:
        softmax_probs = self._softmax(z)
        grad = -y_true / softmax_probs / y_true.shape[0]
        return grad


class NLLLoss(Loss):
    """
    Computes the Negative Log-Likelihood (NLL) loss, which is widely used
    in classification tasks where the outputs are raw logits. This loss
    function is suitable for scenarios where the true labels are provided
    as integer indices of the correct class.

    Notes
    -----
    - The loss function assumes the use of softmax to convert logits to
      probabilities during backpropagation.
    """

    def __init__(self):
        super().__init__()

    def _softmax(self, z: Matrix) -> Matrix:
        exp_shift = np.exp(z - np.max(z, axis=1, keepdims=True))
        softmax_out = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return softmax_out

    def loss(self, y_true: Matrix, z: Matrix) -> float:
        softmax_probs = self._softmax(z)
        clipped_probs = np.clip(softmax_probs, self.epsilon, 1.0)

        log_probs = -np.log(clipped_probs[np.arange(y_true.shape[0]), y_true])

        return np.mean(log_probs)

    def grad(self, y_true: Matrix, z: Matrix) -> Matrix:
        softmax_probs = self._softmax(z)

        softmax_probs[np.arange(y_true.shape[0]), y_true] -= 1
        grad = softmax_probs / y_true.shape[0]

        return grad
