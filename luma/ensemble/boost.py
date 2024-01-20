from typing import List
import numpy as np

from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector, Clone
from luma.interface.exception import NotFittedError
from luma.classifier.tree import DecisionTreeClassifier
from luma.preprocessing.encoder import LabelBinarizer
from luma.metric.classification import Accuracy


class AdaBoostClassifier(Estimator, Supervised):
    
    """
    The AdaBoost classifier is an ensemble method that combines multiple 
    weak learners, typically decision trees, to form a strong classifier. 
    During training, it iteratively adjusts the weights of misclassified 
    instances, making them more likely to be correctly predicted in subsequent 
    rounds. Each weak learner's contribution to the final prediction is 
    weighted based on its accuracy. AdaBoost is adaptive in the sense that 
    subsequent weak learners are tweaked in favor of instances that previous 
    learners misclassified. The final model makes predictions based on the 
    weighted majority vote of its weak learners.
    
    Parameters
    ----------
    `base_estimator` : Base estimator for training multiple models
    (Default `DecisionTreeClassifier`)
    `n_estimators` : Number of base estimators to fit
    `learning_rate` : Step size of class weights(`alpha`) update
    
    Examples
    --------
    >>> ada = AdaBoostClassifier(learning_rate=1.0, ...)
    >>> ada.fit(X, y)
    
    >>> pred = ada.predict(X)
    >>> est = ada[i] # Get i-th estimator from `ada`
    
    """
    
    def __init__(self, 
                 base_estimator: Estimator = DecisionTreeClassifier(),
                 n_estimators: int = 100, 
                 learning_rate: float = 1.0,
                 verbose: bool = False) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.estimators_: List[Estimator] = []
        self.weights_: List[float] = []
        self.classes_ = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'AdaBoostClassifier':
        self.classes_ = np.unique(y)
        m, _ = X.shape
        n_classes = len(self.classes_)
        sample_weights = np.full(m, 1 / m)
        
        lb = LabelBinarizer().fit(self.classes_)
        y_bin = lb.transform(y)

        for _ in range(self.n_estimators):
            clf = Clone(self.base_estimator).get
            if hasattr(clf, 'max_depth'): clf.max_depth = 1
            if hasattr(clf, 'sample_weights'): clf.sample_weights = sample_weights
            
            clf.fit(X, y)
            pred = clf.predict(X)
            pred_bin = lb.transform(pred)

            error_vect = np.sum(y_bin != pred_bin, axis=1) / (n_classes - 1)
            error = np.sum(sample_weights * error_vect) / np.sum(sample_weights)
            if error >= (n_classes - 1) / n_classes: continue

            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            sample_weights *= np.exp(alpha * error_vect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(clf)
            self.weights_.append(alpha)

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        clf_preds = np.array([clf.predict(X) for clf in self.estimators_])
        weighted_preds = np.zeros((m, len(self.classes_)))
        
        for i, clf_pred in enumerate(clf_preds):
            for j in range(m):
                class_index = np.where(self.classes_ == clf_pred[j])[0][0]
                weighted_preds[j, class_index] += self.weights_[i]
        
        return self.classes_[np.argmax(weighted_preds, axis=1)]

    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, 
                   base_estimator: Estimator = None,
                   n_estimators: int = None,
                   learning_rate: float = None) -> None:
        if base_estimator is not None: self.base_estimator = base_estimator
        if n_estimators is not None: self.n_estimators = n_estimators
        if learning_rate is not None: self.learning_rate = learning_rate

    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]

