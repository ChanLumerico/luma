import numpy as np

from luma.interface.typing import Matrix, ClassType
from luma.core.super import Evaluator


__all__ = (
    "MeanAbsoluteError",
    "MeanSquaredError",
    "RootMeanSquaredError",
    "MeanAbsolutePercentageError",
    "RSquaredScore",
    "AdjustedRSquaredScore",
)


@ClassType.non_instantiable()
class MeanAbsoluteError(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean(np.abs(y_true - y_pred))


@ClassType.non_instantiable()
class MeanSquaredError(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean((y_true - y_pred) ** 2)


@ClassType.non_instantiable()
class RootMeanSquaredError(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


@ClassType.non_instantiable()
class MeanAbsolutePercentageError(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


@ClassType.non_instantiable()
class RSquaredScore(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        y_bar = np.mean(y_true)
        sst = np.sum((y_true - y_bar) ** 2)
        ssr = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ssr / sst)
        return r2


@ClassType.non_instantiable()
class AdjustedRSquaredScore(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix, n_predictors: int) -> float:
        m = len(y_true)
        y_bar = np.mean(y_true)
        sst = np.sum((y_true - y_bar) ** 2)
        ssr = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ssr / sst)
        r2_adj = 1 - ((1 - r2) * (m - 1) / (m - n_predictors - 1))
        return r2_adj
