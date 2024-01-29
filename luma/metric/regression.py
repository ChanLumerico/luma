from typing import *
import numpy as np

from luma.interface.util import Matrix
from luma.core.super import Evaluator


__all__ = (
    'MeanAbsoluteError', 
    'MeanSquaredError', 
    'RootMeanSquaredError',
    'MeanAbsolutePercentageError', 
    'RSquaredScore'
)


class MeanAbsoluteError(Evaluator):
    @staticmethod
    def score(y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean(np.abs(y_true - y_pred))


class MeanSquaredError(Evaluator):
    @staticmethod
    def score(y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Evaluator):
    @staticmethod
    def score(y_true: Matrix, y_pred: Matrix) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MeanAbsolutePercentageError(Evaluator):
    @staticmethod
    def score(y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class RSquaredScore(Evaluator):
    @staticmethod
    def score(y_true: Matrix, y_pred: Matrix) -> float:
        y_bar = np.mean(y_true)
        total_sum_of_squares = np.sum((y_true - y_bar) ** 2)
        residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2

