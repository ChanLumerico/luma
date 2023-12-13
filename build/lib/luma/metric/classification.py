from typing import *
import numpy as np

from luma.interface.super import Evaluator


__all__ = ['Accuracy', 'Precision', 'Recall', 'F1Score', 'Specificity',
           'AUCCurveROC', 'Complex']


class Accuracy(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class Precision(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        return true_positives / (true_positives + false_positives)


class Recall(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        return true_positives / (true_positives + false_negatives)


class F1Score(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = Precision.compute(y_true, y_pred)
        recall = Recall.compute(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)


class Specificity(Evaluator):
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        return true_negatives / (true_negatives + false_positives)


class AUCCurveROC(Evaluator):
    @staticmethod
    def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        thresholds = np.sort(y_scores)
        tpr, fpr = [], []
        for threshold in thresholds:
            y_pred = y_scores >= threshold
            tpr.append(Recall.compute(y_true, y_pred))
            fpr.append(1 - Specificity.compute(y_true, y_pred))
        return fpr, tpr
    
    @staticmethod
    def auc_roc(fpr: float, tpr: float) -> float:
        n = len(fpr)
        auc = 0
        for i in range(1, n):
            auc += (fpr[i] - fpr[i - 1]) * tpr[i]
        return auc


class Complex:
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        scores = dict()
        scores['accuracy'] = Accuracy.compute(y_true, y_pred)
        scores['precision'] = Precision.compute(y_true, y_pred)
        scores['recall'] = Recall.compute(y_true, y_pred)
        scores['f1-score'] = F1Score.compute(y_true, y_pred)
        scores['specificity'] = Specificity.compute(y_true, y_pred)
        return scores

