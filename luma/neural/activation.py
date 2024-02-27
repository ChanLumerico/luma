import numpy as np


__all__ = (
    'ReLU',
    'LeakyReLU',
    'ELU',
    'Tanh', 
    'Sigmoid',
    'Softmax'
)

class _Matrix: pass

Matrix = _Matrix


class ReLU:
    def func(self, X: Matrix) -> Matrix:
        return np.maximum(0, X)
    
    def derivative(self, X: Matrix) -> Matrix:
        return (X > 0).astype(float)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def func(self, X: Matrix) -> Matrix:
        return np.where(X > 0, X, X * self.alpha)

    def derivative(self, X: Matrix) -> Matrix:
        return np.where(X > 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def func(self, X: Matrix) -> Matrix:
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))

    def derivative(self, X: Matrix) -> Matrix:
        return np.where(X > 0, 1, self.func(X) + self.alpha)


class Tanh:
    def func(self, X: Matrix) -> Matrix:
        return np.tanh(X)

    def derivative(self, X: Matrix) -> Matrix:
        return 1 - np.tanh(X) ** 2


class Sigmoid:
    def func(self, X: Matrix) -> Matrix:
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: Matrix) -> Matrix:
        return X * (1 - X)


class Softmax:
    def func(self, X: Matrix) -> Matrix:
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, X: Matrix) -> Matrix:
        m, n = X.shape
        soft_out = self.func(X)
        jacobian = np.zeros((m, n, n))
        
        for i in range(len(soft_out)):
            for j in range(len(soft_out[i])):
                for k in range(len(soft_out[i])):
                    if j == k: val = soft_out[i, j] * (1 - soft_out[i, j])
                    else: val = -soft_out[i, j] * soft_out[i, k]
                    jacobian[i, j, k] = val

        return jacobian

