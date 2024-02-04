from typing import List, Literal
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector, Clone
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.classifier.tree import DecisionTreeClassifier
from luma.regressor.tree import DecisionTreeRegressor
from luma.preprocessing.encoder import LabelBinarizer
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError


__all__ = (
    'AdaBoostClassifier',
    'AdaBoostRegressor'
)


class AdaBoostClassifier(Estimator, Estimator.Meta, Supervised):
    
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
    >>> ada_clf = AdaBoostClassifier(learning_rate=1.0, ...)
    >>> ada_clf.fit(X, y)
    
    >>> pred = ada_clf.predict(X)
    >>> est = ada_clf[i] # Get i-th estimator from `ada_clf`
    
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

        for i in range(self.n_estimators):
            clf = Clone(self.base_estimator).get
            if hasattr(clf, 'max_depth'): clf.max_depth = 1
            if hasattr(clf, 'sample_weights'): clf.sample_weights = sample_weights
            
            clf.fit(X, y)
            pred = clf.predict(X)
            pred_bin = lb.transform(pred)

            error_vect = np.sum(y_bin != pred_bin, axis=1) / (n_classes - 1)
            error = np.sum(sample_weights * error_vect) / np.sum(sample_weights)
            error = max(error, 1e-10)
            if error >= (n_classes - 1) / n_classes: continue

            alpha = self.learning_rate * np.log((1 - error) / error)
            sample_weights *= np.exp(alpha * error_vect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(clf)
            self.weights_.append(alpha)
            
            if self.verbose:
                print(f'[AdaBoost] Finished fitting {type(clf).__name__}',
                      f'{i + 1}/{self.n_estimators} with',
                      f'error: {error}, alpha: {alpha}')

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        clf_preds = Matrix([clf.predict(X) for clf in self.estimators_])
        weighted_preds = np.zeros((m, len(self.classes_)))
        
        for i, clf_pred in enumerate(clf_preds):
            for j in range(m):
                class_index = np.where(self.classes_ == clf_pred[j])[0][0]
                weighted_preds[j, class_index] += self.weights_[i]
            
            if self.verbose:
                print(f'[AdaBoost] Finished prediction of',
                      f'{type(self.base_estimator).__name__}',
                      f'{i + 1}/{self.n_estimators}')
        
        return self.classes_[np.argmax(weighted_preds, axis=1)]

    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]


class AdaBoostRegressor(Estimator, Estimator.Meta, Supervised):
    
    """
    The AdaBoost regressor is an ensemble method that combines multiple 
    weak learners, typically decision trees, to form a strong regressor. 
    During training, it focuses on reducing the errors of the ensemble by 
    adjusting the sample weights and fitting each weak learner to the 
    residual errors of the previous learners. The final output is a 
    combination of the weak learners' outputs, weighted by their individual 
    contributions to the ensemble's performance, which aims to produce 
    a more accurate regression model.
    
    Parameters
    ----------
    `base_estimator` : Base estimator for training multiple models
    (Default `DecisionTreeRegressor`)
    `n_estimators` : Number of base estimators to fit
    `learning_rate` : Step size of class weights(`alpha`) update
    `loss` : Type of loss function (e.g. `linear`, `square`, `exp`)

    Examples
    --------
    >>> ada_reg = AdaBoostRegressor(learning_rate=0.1, ...)
    >>> ada_reg.fit(X_train, y_train)

    >>> y_pred = ada_reg.predict(X_test)
    >>> est = ada_reg[i] # Get the i-th estimator from the `ada_reg`

    """
    
    def __init__(self, 
                 base_estimator: Estimator = DecisionTreeRegressor(),
                 n_estimators: int = 100,
                 learning_rate: float = 1.0,
                 loss: Literal['linear', 'sqaure', 'exp'] = 'linear',
                 verbose: bool = False) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.verbose = verbose
        self.estimators_: List[Estimator] = []
        self.weights_: List[float] = []
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'AdaBoostRegressor':
        m, _ = X.shape
        sample_weights = np.full(m, 1 / m)
        y_pred = np.zeros(m)
        
        for i in range(self.n_estimators):
            residual = y - y_pred
            clf = Clone(self.base_estimator).get
            if hasattr(clf, 'max_depth'): clf.max_depth = 1
            if hasattr(clf, 'sample_weights'): clf.sample_weights = sample_weights
            
            clf.fit(X, residual)
            pred = clf.predict(X)
            loss = self._compute_loss(residual, pred)
            
            error = max(np.sum(sample_weights * loss), 1e-10)
            alpha = self.learning_rate * np.log((1 - error) / error)
            
            sample_weights *= np.exp(alpha * loss)
            sample_weights /= np.sum(sample_weights)
            
            y_pred += alpha * pred
            self.estimators_.append(clf)
            self.weights_.append(alpha)
            
            if self.verbose:
                print(f'[AdaBoost] Finished fitting {type(clf).__name__}',
                      f'{i + 1}/{self.n_estimators} with',
                      f'error: {error}, alpha: {alpha}')
        
        self._fitted = True
        return self
    
    def _compute_loss(self, y: Vector, y_pred: Vector) -> Vector:
        if self.loss == 'linear': return np.abs(y - y_pred)
        elif self.loss == 'square': return (y - y_pred) ** 2
        elif self.loss == 'exp': return np.exp(-y * y_pred)
        else: raise UnsupportedParameterError(self.loss)
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        predictions = np.zeros(X.shape[0])
        for weight, estimator in zip(self.weights_, self.estimators_):
            predictions += weight * estimator.predict(X)
        
        return predictions
    
    def score(self, X: Matrix, y: Vector, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
    
    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]
    
