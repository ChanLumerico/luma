from typing import Any, List, Dict, Literal
import numpy as np

from luma.core.super import Estimator, Transformer, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.interface.util import Matrix, Vector
from luma.classifier.logistic import SoftmaxRegressor
from luma.regressor.linear import LinearRegressor
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError
from luma.model_selection.fold import FoldType, KFold


__all__ = (
    'StackingClassifier',
    'StackingRegressor'
)


class StackingClassifier(Estimator, Transformer, Supervised):
    
    """
    A stacking classifier is a machine learning model that combines multiple 
    classification models into a single predictive model by stacking the output 
    of individual classifiers as input to a final classifier. This final 
    classifier, often called a meta-classifier, is trained to make a final 
    prediction based on the outputs of the base classifiers. Stacking aims to 
    leverage the strengths of each base model to improve overall prediction 
    accuracy.
    
    Parameters
    ----------
    `estimators` : List of base estimators
    `final_estimator` : Final meta-estimator (Default `SoftmaxRegressor`)
    `pass_original` : Whether to pass the original data to final estimator
    `drop_threshold` : Omitting threshold for base estimators
    (`None` not to omit any estimators)
    `metric` : Scoring metric
    `method` : Methods called for each base estimator
    `cv` : Number of folds for cross-validation
    `fold_type` : Fold type (Default `KFold`)
    `shuffle` : Whether to shuffle the dataset when cross-validating
    `random_state` : Seed for random splitting
    `**kwargs` : Additional parameters for final estimator
    (i.e. `learning_rate`)
    
    Notes
    -----
    `StackingClassifier` can also be utilized as `Transformer`.
    
    - The method `transform` returns class labels or probabilities by each 
        base estimator as a stacked form:
    
        ```py
        def transform(self, X: Matrix) -> Matrix
        ```
    
    Each base estimators can be accessed via indexing of its instance:
    >>> estimator = stack[0]
    
    Examples
    --------
    ```py
    stack = StackingClassifier(estimators=[...],
                               final_estimator=SoftmaxRegressor(),
                               method='label',
                               cv=5,
                               fold_type=KFold)
    stack.fit(X, y)
    X_new = stack.transform(X)
    y_pred = stack.predict(X)
    ```
    """

    def __init__(self, 
                 estimators: List[Estimator],
                 final_estimator: Estimator = SoftmaxRegressor(), 
                 pass_original: bool = False,
                 drop_threshold: float = None, 
                 metric: Evaluator = Accuracy,
                 method: Literal['label', 'prob'] = 'label',
                 cv: int = 5,
                 fold_type: FoldType = KFold, 
                 shuffle: bool = True,
                 random_state: int = None,
                 verbose: bool = False, 
                 **kwargs: Dict[str, Any]) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.pass_original = pass_original
        self.drop_threshold = drop_threshold
        self.metric = metric
        self.method = method
        self.cv = cv
        self.fold_type = fold_type
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self._final_estimator_params = kwargs
        self._base_estimators: List[Estimator] = []
        self._fitted = False
        
        if self.method not in ('label', 'prob'):
            raise UnsupportedParameterError(self.method)

    def fit(self, X: Matrix, y: Vector) -> 'StackingClassifier':
        m, _ = X.shape
        self.n_classes = len(np.unique(y))
        
        fold = self.fold_type(X, y,
                              n_folds=self.cv,
                              shuffle=self.shuffle,
                              random_state=self.random_state)
        
        if self.method == 'label': X_new = np.zeros((m, len(self.estimators)))
        else: X_new = np.zeros((m, len(self.estimators) * self.n_classes))
        
        for i, est in enumerate(self.estimators):
            if self.method == 'label': preds = np.zeros(m)
            else: preds = np.zeros((m, self.n_classes))
            
            for train_indices, test_indices in fold.split:
                X_train, y_train = X[train_indices], y[train_indices]
                X_test = X[test_indices]
                
                est.fit(X_train, y_train)
                if self.method == 'label': pred = est.predict(X_test)
                else:
                    if not hasattr(est, 'predict_proba'):
                        raise ValueError(f"{type(est).__name__}" + 
                                         " does not support 'predict_proba'!")
                    pred = est.predict_proba(X_test)
                
                preds[test_indices] = pred
            
            score = est.score(X, y, metric=self.metric)
            if self.drop_threshold is not None and score < self.drop_threshold:
                if self.verbose:
                    print(f'[StackingClassifier] {type(est).__name__}',
                          f'dropped for low score: {score}')
                continue
            
            if self.method == 'label': X_new[:, i] = preds
            else: X_new[:, i * self.n_classes:(i + 1) * self.n_classes] = preds
            
            self._base_estimators.append(est)
            if self.verbose:
                print(f'[StackingClassifier] Finished CV fitting',
                      f'for {type(est).__name__} with score: {score}')
        
        if self.pass_original:
            X_new = np.hstack((X, X_new))
        
        self.final_estimator.set_params(**self._final_estimator_params)
        self.final_estimator.fit(X_new, y)
        
        if self.verbose:
            print(f'[StackingClassifier] Finished fitting for',
                  f'{type(self.final_estimator).__name__}(final)')
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        X_new = self.transform(X)
        
        return self.final_estimator.predict(X_new)

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        if self.method == 'label':
            raise UnsupportedParameterError(self.method)
        
        X_new = self.transform(X)
        if hasattr(self.final_estimator, 'predict_proba'):
            return self.final_estimator.predict_proba(X_new)
        else:
            raise ValueError(f"{type(self.final_estimator).__name__}" + 
                             " does not support 'predict_proba'!")
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        
        if self.method == 'label':
            X_new = np.zeros((m, len(self._base_estimators)))
            for i, est in enumerate(self._base_estimators):
                X_new[:, i] = est.predict(X)
        else:
            X_new = np.zeros((m, len(self._base_estimators) * self.n_classes))
            for i, est in enumerate(self._base_estimators):
                if hasattr(est, 'predict_proba'): 
                    preds = est.predict_proba(X)
                    X_new[:, i * self.n_classes:(i + 1) * self.n_classes] = preds
                else: 
                    raise ValueError(f"{type(est).__name__}" + 
                                    " does not support 'predict_proba'!")
        if self.pass_original:
            X_new = np.hstack((X, X_new))
        
        return X_new

    def fit_transform(self, X: Matrix, y: Vector) -> Matrix:
        self.fit(X, y)
        return self.transform(X)
    
    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __getitem__(self, index: int) -> Estimator:
        return self._base_estimators[index]


class StackingRegressor(Estimator, Transformer, Supervised):
    
    """
    A stacking regressor is a machine learning model that combines multiple 
    regression models into a single predictive model by stacking the output 
    of individual regressors as input to a final regressor. This final 
    regressor, often called a meta-regressor, is trained to make a final 
    prediction based on the outputs of the base regressors. Stacking aims to 
    leverage the strengths of each base model to improve overall prediction 
    accuracy.
    
    Parameters
    ----------
    `estimators` : List of base estimators
    `final_estimator` : Final meta-estimator (Default `LinearRegressor`)
    `pass_original` : Whether to pass the original data to final estimator
    `cv` : Number of folds for cross-validation
    `fold_type` : Fold type (Default `KFold`)
    `metric` : Scoring metric
    `shuffle` : Whether to shuffle the dataset when cross-validating
    `random_state` : Seed for random splitting
    `**kwargs` : Additional parameters for final estimator
    (i.e. `learning_rate`)
    
    Notes
    -----
    `StackingRegressor` can also be utilized as `Transformer`.
    
    - The method `transform` returns class labels or probabilities by each 
        base estimator as a stacked form:
    
        ```py
        def transform(self, X: Matrix) -> Matrix
        ```
    
    Each base estimators can be accessed via indexing of its instance:
    >>> estimator = stack[0]
    
    Examples
    --------
    ```py
    stack = StackingRegressor(estimators=[...],
                              final_estimator=LinearRegressor(),
                              cv=5,
                              fold_type=KFold)
    stack.fit(X, y)
    X_new = stack.transform(X)
    y_pred = stack.predict(X)
    ```
    """
    
    def __init__(self, 
                 estimators: List[Estimator],
                 final_estimator: Estimator = LinearRegressor(), 
                 pass_original: bool = False,
                 cv: int = 5,
                 fold_type: FoldType = KFold, 
                 metric: Evaluator = MeanSquaredError,
                 shuffle: bool = True,
                 random_state: int = None,
                 verbose: bool = False, 
                 **kwargs: Dict[str, Any]) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.pass_original = pass_original
        self.cv = cv
        self.fold_type = fold_type
        self.metric = metric
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self._final_estimator_params = kwargs
        self._base_estimators: List[Estimator] = []
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> 'StackingClassifier':
        m, _ = X.shape
        self.n_classes = len(np.unique(y))
        
        fold = self.fold_type(X, y,
                              n_folds=self.cv,
                              shuffle=self.shuffle,
                              random_state=self.random_state)
        
        X_new = np.zeros((m, len(self.estimators)))
        for i, est in enumerate(self.estimators):
            preds = np.zeros(m)
            for train_indices, test_indices in fold.split:
                X_train, y_train = X[train_indices], y[train_indices]
                X_test = X[test_indices]
                
                est.fit(X_train, y_train)
                pred = est.predict(X_test)
                preds[test_indices] = pred
            
            score = est.score(X, y, metric=self.metric)
            X_new[:, i] = preds
            
            self._base_estimators.append(est)
            if self.verbose:
                print(f'[StackingRegressor] Finished CV fitting',
                      f'for {type(est).__name__} with score: {score}')
        
        if self.pass_original:
            X_new = np.hstack((X, X_new))
        
        self.final_estimator.set_params(**self._final_estimator_params)
        self.final_estimator.fit(X_new, y)
        
        if self.verbose:
            print(f'[StackingRegressor] Finished fitting for',
                  f'{type(self.final_estimator).__name__}(final)')
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        X_new = self.transform(X)
        
        return self.final_estimator.predict(X_new)
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        m, _ = X.shape
        X_new = np.zeros((m, len(self._base_estimators)))
        for i, est in enumerate(self._base_estimators):
            X_new[:, i] = est.predict(X)
            
        if self.pass_original: 
            X_new = np.hstack((X, X_new))
        
        return X_new

    def fit_transform(self, X: Matrix, y: Vector) -> Matrix:
        self.fit(X, y)
        return self.transform(X)
    
    def score(self, X: Matrix, y: Vector, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __getitem__(self, index: int) -> Estimator:
        return self._base_estimators[index]

