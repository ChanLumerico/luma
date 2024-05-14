from typing import Self
import numpy as np

from luma.interface.typing import Matrix, Scalar
from luma.core.super import Transformer
from luma.interface.exception import NotFittedError


__all__ = ("StandardScaler", "MinMaxScaler")


class StandardScaler(Transformer):
    """
    Standard scaling is a data preprocessing technique to transform
    the data so that it has a mean of 0 and a standard deviation of 1.
    """

    def __init__(self) -> None:
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, X: Matrix) -> Self:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)
        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        epsilon = 1e-8
        return (X - self.mean) / (self.std + epsilon)

    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        return (X * self.std) + self.mean


class MinMaxScaler(Transformer, Transformer.Feature):
    """
    MinMax scaling (also known as Min-Max normalization)
    to rescale features to a specific range, typically between 0 and 1.
    The purpose of MinMax scaling is to transform the features in a way
    that they fall within a specific interval.

    Parameters
    ----------
    `feature_range` : tuple[Scalar, Scalar], default=(0,1)
        Range to be scaled

    """

    def __init__(self, feature_range: tuple[Scalar, Scalar] = (0, 1)) -> None:
        self.feature_range = feature_range
        self.min = None
        self.max = None
        self._fitted = False

    def fit(self, X: Matrix) -> Self:
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        min_val, max_val = self.feature_range
        scaled = (X - self.min) / (self.max - self.min) * (max_val - min_val)
        return scaled + min_val

    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        min_val, max_val = self.feature_range
        original = (X - min_val) / (max_val - min_val) * (self.max - self.min)
        return original + self.min
