import numpy as np

from luma.interface.typing import Matrix


__all__ = ("ReLU", "LeakyReLU", "ELU", "Tanh", "Sigmoid")


class ReLU:
    def func(self, X: Matrix) -> Matrix:
        return np.maximum(0, X)

    def grad(self, X: Matrix) -> Matrix:
        return (X > 0).astype(float)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def func(self, X: Matrix) -> Matrix:
        return np.where(X > 0, X, X * self.alpha)

    def grad(self, X: Matrix) -> Matrix:
        return np.where(X > 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def func(self, X: Matrix) -> Matrix:
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))

    def grad(self, X: Matrix) -> Matrix:
        return np.where(X > 0, 1, self.func(X) + self.alpha)


class Tanh:
    def func(self, X: Matrix) -> Matrix:
        return np.tanh(X)

    def grad(self, X: Matrix) -> Matrix:
        return 1 - np.tanh(X) ** 2


class Sigmoid:
    def func(self, X: Matrix) -> Matrix:
        return 1 / (1 + np.exp(-X))

    def grad(self, X: Matrix) -> Matrix:
        return X * (1 - X)
