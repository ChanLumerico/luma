from typing import Any, Generator, List, Dict
import numpy as np

from luma.core.super import Estimator, Supervised
from luma.interface.exception import NotFittedError
from luma.interface.util import Matrix, Vector
from luma.classifier.logistic import SoftmaxRegressor
from luma.model_selection.fold import KFold


__all__ = (
    'StackingClassifier',
)


class StackingClassifier(Estimator, Supervised):

    def __init__(self, 
                 estimators: List[Estimator],
                 final_estimator: Estimator = SoftmaxRegressor(), 
                 pass_original: bool = False,
                 drop_threshold: float = None, 
                 cv: int = 5,
                 fold_generator: Generator = None, 
                 shuffle: bool = True,
                 verbose: bool = False, 
                 random_state: int = None,
                 **kwargs: Dict[str, Any]) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.pass_original = pass_original
        self.drop_threshold = drop_threshold
        self.cv = cv
        self.fold_generator = fold_generator
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self._final_estimator_params = kwargs
        self._base_estimators: List[Estimator] = []
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> 'StackingClassifier':
        m, _ = X.shape
        if self.fold_generator is None:
            fold = KFold(X, y,
                         n_folds=self.cv,
                         shuffle=self.shuffle,
                         random_state=self.random_state)
            self.fold_generator = fold.split
        
        X_new = np.zeros((m, len(self.estimators)))
        for i, est in enumerate(self.estimators):
            predictions = np.zeros(m)
            fold_idx = 0
            
            for X_train, y_train, X_test, _ in self.fold_generator:
                est.fit(X_train, y_train)
                pred = est.predict(X_test)
                
                start = fold_idx * (m // self.cv)
                if fold_idx == self.cv - 1: end = m
                else: end = start + (m // self.cv)
                
                predictions[start:end] = pred
                fold_idx += 1
            
            score = est.score(X, y)
            if self.drop_threshold is not None and score < self.drop_threshold:
                if self.verbose:
                    print(f'[StackingClassifier] {type(est).__name__}',
                          f'dropped for low score: {score}')
                continue
            
            X_new[:, i] = predictions
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
        m, _ = X.shape
        X_new = np.zeros((m, len(self._base_estimators)))
        
        for i, est in enumerate(self._base_estimators):
            X_new[:, i] = est.predict(X)
            
        if self.pass_original:
            X_new = np.hstack((X, X_new))
        
        return self.final_estimator.predict(X_new)

