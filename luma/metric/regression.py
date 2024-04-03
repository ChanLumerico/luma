from typing import *
import numpy as np

from luma.interface.util import Matrix
from luma.core.super import Evaluator


__all__ = (
    "MeanAbsoluteError",
    "MeanSquaredError",
    "RootMeanSquaredError",
    "MeanAbsolutePercentageError",
    "RSquaredScore",
    "AdjustedRSquaredScore",
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
        sst = np.sum((y_true - y_bar) ** 2)
        ssr = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ssr / sst)
        return r2


class AdjustedRSquaredScore(Evaluator):
    @staticmethod
    def score(y_true: Matrix, y_pred: Matrix, n_predictors: int) -> float:
        m = len(y_true)
        y_bar = np.mean(y_true)
        sst = np.sum((y_true - y_bar) ** 2)
        ssr = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ssr / sst)
        r2_adj = 1 - ((1 - r2) * (m - 1) / (m - n_predictors - 1))
        return r2_adj
