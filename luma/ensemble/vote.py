from typing import List, Literal
import numpy as np

from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector, Scalar, Clone
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.classification import Accuracy
from luma.preprocessing.encoder import LabelEncoder


__all__ = (
    'MajorityVoteClassifier'
)


class MajorityVoteClassifier(Estimator, Supervised):
    
    """
    A majority vote classifier is an ensemble method that combines predictions 
    from multiple machine learning models. It aggregates the predictions from 
    each model and selects the final output based on the majority vote. The 
    voting can be based on either the predicted class labels or the predicted 
    probabilities. This approach often leads to improved model performance and 
    robustness compared to individual models.
    
    Parameters
    ----------
    `estimators` : List of estimators to vote
    `voting` : Voting critetion
    (`label` to vote the most frequent and `prob` to vote the most probable)
    `weights` : Weights for each classifier on voting
    (`None` for uniform weights)
    
    Examples
    --------
    >>> maj = MajorityVoteClassifier(estimators=[AnyEstimator(), ...],
                                     voting='label',
                                     weights=[0.25, 0.5, ...])
    >>> maj.fit(X, y)
    >>> pred = maj.predict(X)
    >>> est = maj[i] # Get i-th estimator from `maj`
    
    """
    
    def __init__(self,
                 estimators: List[Estimator] = None,
                 voting: Literal['label', 'prob'] = 'label',
                 weights: Vector[Scalar] = None,
                 verbose: bool = True) -> None:
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'MajorityVoteClassifier':
        if self.voting not in ('label', 'prob'):
            raise UnsupportedParameterError(self.voting)
        
        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError(f"Size mismatch between 'weights' and 'estimators'!")
        
        self.le = LabelEncoder()
        self.le.fit(y)
        
        self.classes = self.le.classes
        self.estimators_ = []
        
        for est in self.estimators:
            fitted_est = Clone(est).get
            if hasattr(fitted_est, 'verbose'):
                fitted_est.verbose = self.verbose
            
            fitted_est.fit(X, self.le.transform(y))
            self.estimators_.append(fitted_est)
        
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        
        if self.voting == 'label':
            preds = np.asarray([est.predict(X) for est in self.estimators_]).T
            majority_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=preds
            )
        else:
            majority_vote = np.argmax(self.predict_proba(X), axis=1)
        
        majority_vote = self.le.inverse_transform(majority_vote)
        return majority_vote
    
    def predict_proba(self, X: Matrix) -> Matrix:
        probas = []
        for est in self.estimators_:
            if not hasattr(est, 'predict_proba'):
                raise ValueError(f"'{type(est)}' does not support 'predict_proba'")
            probas.append(est.predict_proba(X))
        
        return np.average(probas, axis=0, weights=self.weights)

    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, 
                   estimators: List[Estimator] = None,
                   voting: Literal = None,
                   weights: Vector[Scalar] = None) -> None:
        if estimators is not None: self.estimators = estimators
        if voting is not None: self.voting = str(voting)
        if weights is not None: self.weights = weights

    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]

