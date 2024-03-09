from typing import Any, Dict, List, Literal, Tuple
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
    >>> ada_est = AdaBoostClassifier(learning_rate=1.0, ...)
    >>> ada_est.fit(X, y)
    
    >>> y_pred = ada_est.predict(X)
    >>> est = ada_est[i] # Get i-th estimator from `ada_est`
    
    """
    
    def __init__(self, 
                 base_estimator: Estimator = DecisionTreeClassifier(),
                 n_estimators: int = 50, 
                 learning_rate: float = 1.0,
                 verbose: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.classes_ = None
        self.estimators_: List[Estimator] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        self._base_estimator_params = kwargs
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'AdaBoostClassifier':
        self.classes_ = np.unique(y)
        
        m, _ = X.shape
        n_classes = len(self.classes_)
        sample_weights = np.ones(m)
        
        lb = LabelBinarizer().fit(self.classes_)
        y_bin = lb.transform(y)

        for i in range(self.n_estimators):
            est = Clone(self.base_estimator).get
            if hasattr(est, 'max_depth'): est.max_depth = 3
            
            est.set_params(**self._base_estimator_params)
            est.fit(X, y, sample_weights=sample_weights)
            y_pred = est.predict(X)
            y_pred_bin = lb.transform(y_pred)

            error_vect = np.sum(y_bin != y_pred_bin, axis=1) / (n_classes - 1)
            error = np.sum(sample_weights * error_vect) / np.sum(sample_weights)
            error = max(error, 1e-10)
            if error >= (n_classes - 1) / n_classes: continue

            alpha = self.learning_rate * np.log((1 - error) / error)
            sample_weights *= np.exp(alpha * error_vect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            
            if self.verbose:
                print(f'[AdaBoost] Finished fitting {type(est).__name__}',
                      f'{i + 1}/{self.n_estimators} with',
                      f'error: {error}, alpha: {alpha}')

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        est_preds = Matrix([est.predict(X) for est in self.estimators_])
        weighted_preds = np.zeros((m, len(self.classes_)))
        
        for i, est_pred in enumerate(est_preds):
            for j in range(m):
                class_index = np.where(self.classes_ == est_pred[j])[0][0]
                weighted_preds[j, class_index] += self.estimator_weights_[i]
            
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
    >>> ada_reg = AdaBoostRegressor(learning_rate=1.0, ...)
    >>> ada_reg.fit(X_train, y_train)

    >>> y_pred = ada_reg.predict(X_test)
    >>> est = ada_reg[i] # Get the i-th estimator from the `ada_reg`

    """
    
    def __init__(self, 
                 base_estimator: Estimator = DecisionTreeRegressor(),
                 n_estimators: int = 50,
                 learning_rate: float = 1.0,
                 loss: Literal['linear', 'square', 'exp'] = 'linear', 
                 verbose: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.verbose = verbose
        self.estimators_: List[Estimator] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        self._base_estimator_params = kwargs
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'AdaBoostRegressor':
        m, _ = X.shape
        sample_weights = np.ones(m)
        
        for i in range(self.n_estimators):
            est = Clone(self.base_estimator).get
            if hasattr(est, 'max_depth'): est.max_depth = 3
            
            est.set_params(**self._base_estimator_params)
            est.fit(X, y, sample_weights=sample_weights)
            y_pred = est.predict(X)
            
            loss, error = self._get_loss(y, y_pred, sample_weights)
            if error == 0 or error >= 1.0: continue
            
            eps = 1e-10
            error = np.maximum(error, eps)
            alpha = self.learning_rate * np.log((1 - error) / error) / 2
            
            sample_weights *= np.power((1 - error) / error, 1 - loss)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            
            if self.verbose:
                print(f'[AdaBoost] Finished fitting {type(est).__name__}',
                      f'{i + 1}/{self.n_estimators} with',
                      f'error: {error}, alpha: {alpha}')
        
        self._fitted = True
        return self
    
    def _get_loss(self, 
                  y_true: Vector, 
                  y_pred: Vector, 
                  sample_weights: Vector) -> Tuple[Vector, float]:
        if self.loss == 'linear':
            loss = np.abs(y_true - y_pred)
        elif self.loss == 'square':
            loss = (y_true - y_pred) ** 2
        elif self.loss == 'exp':
            loss = 1 - np.exp(-np.abs(y_true - y_pred))
        else:
            raise UnsupportedParameterError(self.loss)
        
        weighted_error = np.sum(sample_weights * loss) / np.sum(sample_weights)
        return loss, weighted_error

    def predict(self, X: Matrix) -> Vector:
        preds = np.zeros(X.shape[0])
        for i, (est, weight) in enumerate(zip(self.estimators_, 
                                              self.estimator_weights_)):
            preds += weight * est.predict(X)
            if self.verbose:
                print(f'[AdaBoost] Finished prediction of',
                      f'{type(self.base_estimator).__name__}',
                      f'{i + 1}/{self.n_estimators}')
        
        return preds
    
    def score(self, X: Matrix, y: Vector, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]

