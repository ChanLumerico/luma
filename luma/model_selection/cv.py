from typing import Tuple
import numpy as np

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Vector
from luma.model_selection.fold import FoldType, KFold


__all__ = "CrossValidator"


class CrossValidator(Evaluator):
    """
    Cross-validation is a technique in machine learning for assessing how
    a model generalizes to an independent dataset. It involves dividing
    the data into several parts, training the model on some parts and
    testing it on others. This process is repeated multiple times with
    different partitions, providing a robust estimate of the model's
    performance. It's especially useful in scenarios with limited data,
    ensuring that all available data is effectively utilized for both
    training and validation.

    Parameters
    ----------
    `estimator` : Estimator
        An estimator to validate
    `metric` : Evaluator
        Evaluation metric for validation
    `cv` : int, default=5
        Number of folds in splitting data
    `fold_type` : FoldType, default=KFold
        Fold type
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `random_state` : int, optional, default=None
        Seed for random sampling upon splitting data

    Attributes
    ----------
    `train_scores_` : list
        List of training scores for each fold
    `test_scores_` : list
        List of test(validation) scores for each fold

    Methods
    -------
    For getting mean train score and mean test score respectvely:
    ```py
        def score(self, X: Matrix, y: Vector) -> Tuple[float, float]
    ```
    """

    def __init__(
        self,
        estimator: Estimator,
        metric: Evaluator,
        cv: int = 5,
        fold_type: FoldType = KFold,
        shuffle: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.estimator = estimator
        self.metric = metric
        self.cv = cv
        self.fold_type = fold_type
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.train_scores_ = []
        self.test_scores_ = []

    def _fit(self, X: Matrix, y: Vector) -> None:
        fold = self.fold_type(
            X, y, n_folds=self.cv, shuffle=self.shuffle, random_state=self.random_state
        )

        for i, (train_indices, test_indices) in enumerate(fold.split):
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            self.estimator.fit(X_train, y_train)
            if self.metric:
                train_score = self.estimator.score(X_train, y_train, self.metric)
                test_score = self.estimator.score(X_test, y_test, self.metric)
            else:
                train_score = self.estimator.score(X_train, y_train)
                test_score = self.estimator.score(X_test, y_test)

            self.train_scores_.append(train_score)
            self.test_scores_.append(test_score)

            if self.verbose:
                print(
                    f"[CV] fold {i + 1} -",
                    f"train-score: {train_score:.3f},",
                    f"test-score: {test_score:.3f}",
                )

    def score(self, X: Matrix, y: Vector) -> Tuple[float, float]:
        self._fit(X, y)
        return np.mean(self.train_scores_), np.mean(self.test_scores_)
