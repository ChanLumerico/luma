from typing import *
import numpy as np

from luma.interface.super import Evaluator


__all__ = ['MeanAbsoluteError', 'MeanSquaredError', 'RootMeanSquaredError',
           'MeanAbsolutePercentageError', 'RSquaredScore', 'Complex']


class MeanAbsoluteError(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class MeanSquaredError(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MeanAbsolutePercentageError(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class RSquaredScore(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_bar = np.mean(y_true)
        total_sum_of_squares = np.sum((y_true - y_bar) ** 2)
        residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2


class Complex:
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        scores = dict()
        scores['mae'] = MeanAbsoluteError.compute(y_true, y_pred)
        scores['mse'] = MeanSquaredError.compute(y_true, y_pred)
        scores['rmse'] = RootMeanSquaredError.compute(y_true, y_pred)
        scores['mape'] = MeanAbsolutePercentageError.compute(y_true, y_pred)
        scores['rsqe'] = RSquaredScore.compute(y_true, y_pred)
        return scores

