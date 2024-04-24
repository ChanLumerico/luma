import numpy as np

from luma.interface.typing import Matrix, ClassType
from luma.core.super import Evaluator


__all__ = ("Accuracy", "Precision", "Recall", "F1Score", "Specificity")


@ClassType.non_instantiable()
class Accuracy(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        return np.mean(y_true == y_pred)


@ClassType.non_instantiable()
class Precision(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        return true_positives / (true_positives + false_positives)


@ClassType.non_instantiable()
class Recall(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        return true_positives / (true_positives + false_negatives)


@ClassType.non_instantiable()
class F1Score(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        precision = Precision.score(y_true, y_pred)
        recall = Recall.score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)


@ClassType.non_instantiable()
class Specificity(Evaluator):
    @classmethod
    def score(cls, y_true: Matrix, y_pred: Matrix) -> float:
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        return true_negatives / (true_negatives + false_positives)
