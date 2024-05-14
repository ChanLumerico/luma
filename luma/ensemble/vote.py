from typing import List, Literal, Self
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.typing import Matrix, Vector, Scalar
from luma.interface.util import Clone
from luma.interface.exception import NotFittedError, UnsupportedParameterError

from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError
from luma.preprocessing.encoder import LabelEncoder


__all__ = ("VotingClassifier", "VotingRegressor")


class VotingClassifier(Estimator, Estimator.Meta, Supervised):
    """
    A Voting Classifier is an ensemble machine learning model that combines
    predictions from multiple base models. It uses a voting mechanism, either
    'hard' (majority rule) or 'soft' (weighted probabilities), to determine
    the final output. This approach can increase prediction accuracy and
    robustness by leveraging the strengths of diverse models. It's
    particularly useful when individual models have different error patterns.
    The Voting Classifier is widely used in classification tasks to improve
    performance over single models.

    Parameters
    ----------
    `estimators` : list of Esimators, required optional, default=None
        List of estimators to vote
    `voting` : {"label", "prob"}, default="label"
        Voting criterion ("label" to vote the most frequent and
        "prob" to vote the most probable)
    `weights` : Vector[Scalar], optional, default=None
        Weights for each classifier on voting (None for uniform weights)

    Examples
    --------
    >>> vot = VotingClassifier(
            estimators=[AnyClassifier(), ...],
            voting="label",
            weights=[0.25, 0.5, ...],
        )
    >>> vot.fit(X, y)
    >>> pred = vot.predict(X)
    >>> clf = vot[i] # Get i-th classifier from `vot`

    """

    def __init__(
        self,
        estimators: List[Estimator] | None = None,
        voting: Literal["label", "prob"] = "label",
        weights: Vector[Scalar] | None = None,
        verbose: bool = False,
    ) -> None:
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> Self:
        if self.voting not in ("label", "prob"):
            raise UnsupportedParameterError(self.voting)

        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError(f"Size mismatch between 'weights' and 'estimators'!")

        self.le = LabelEncoder()
        self.le.fit(y)

        self.classes = self.le.classes
        self.estimators_ = []

        for est in self.estimators:
            fitted_est = Clone(est).get
            if hasattr(fitted_est, "verbose"):
                fitted_est.verbose = self.verbose

            fitted_est.fit(X, self.le.transform(y))
            self.estimators_.append(fitted_est)

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)

        if self.voting == "label":
            preds = np.asarray([est.predict(X) for est in self.estimators_]).T
            majority_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=preds,
            )
        else:
            majority_vote = np.argmax(self.predict_proba(X), axis=1)

        majority_vote = self.le.inverse_transform(majority_vote)
        return majority_vote

    def predict_proba(self, X: Matrix) -> Matrix:
        probas = []
        for est in self.estimators_:
            if not hasattr(est, "predict_proba"):
                raise ValueError(f"'{type(est)}' does not support 'predict_proba'")
            probas.append(est.predict_proba(X))

        return np.average(probas, axis=0, weights=self.weights)

    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]


class VotingRegressor(Estimator, Estimator.Meta, Supervised):
    """
    A Voting Regressor is an ensemble machine learning model that combines
    predictions from multiple base regression models. It uses a voting mechanism,
    where the final output is the average of the predictions from all models.
    This approach can increase prediction accuracy and robustness by leveraging
    the strengths of diverse models. It's particularly useful when individual
    models have different error patterns. The Voting Regressor is widely used in
    regression tasks to improve performance over single models.

    Parameters
    ----------
    `estimators` : list of Esimators, required optional, default=None
        List of estimators to vote
    `weights` : Vector[Scalar], optional, default=None
        Weights for each classifier on voting (None for uniform weights)

    Examples
    --------
    >>> vot = VotingRegressor(
            estimators=[AnyRegressor(), ...],
            weights=[0.25, 0.5, ...],
        )
    >>> vot.fit(X, y)
    >>> pred = vot.predict(X)
    >>> reg = vot[i] # Get i-th regressor from `vot`

    """

    def __init__(
        self,
        estimators: List[Estimator] | None = None,
        weights: Vector[Scalar] | None = None,
        verbose: bool = False,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> Self:
        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError(f"Size mismatch between 'weights' and 'estimators'!")

        self.estimators_ = []
        for est in self.estimators:
            fitted_est = Clone(est).get
            if hasattr(fitted_est, "verbose"):
                fitted_est.verbose = self.verbose

            fitted_est.fit(X, y)
            self.estimators_.append(fitted_est)

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)

        predictions = np.asarray([est.predict(X) for est in self.estimators_])
        avg_predictions = np.average(predictions, axis=0, weights=self.weights)
        return avg_predictions

    def score(
        self, X: Matrix, y: Vector, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]
