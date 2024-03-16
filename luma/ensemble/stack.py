from typing import Any, List, Dict, Literal
import numpy as np

from luma.core.super import Estimator, Transformer, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.interface.util import Matrix, Vector
from luma.classifier.logistic import SoftmaxRegressor
from luma.metric.classification import Accuracy
from luma.model_selection.fold import FoldType, KFold


__all__ = (
    'StackingClassifier',
)


class StackingClassifier(Estimator, Transformer, Supervised):
    
    """
    
    """

    def __init__(self, 
                 estimators: List[Estimator],
                 final_estimator: Estimator = SoftmaxRegressor(), 
                 pass_original: bool = False,
                 drop_threshold: float = None, 
                 method: Literal['label', 'prob'] = 'label',
                 cv: int = 5,
                 fold_type: FoldType = KFold, 
                 shuffle: bool = True,
                 verbose: bool = False, 
                 random_state: int = None,
                 **kwargs: Dict[str, Any]) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.pass_original = pass_original
        self.drop_threshold = drop_threshold
        self.method = method
        self.cv = cv
        self.fold_type = fold_type
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
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
            
            for train_indices, test_indices in fold.split():
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
            
            score = est.score(X, y)
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
                
            if self.pass_original: 
                X_new = np.hstack((X, X_new))
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

