from typing import *
import numpy as np

from luma.interface.super import Estimator, Evaluator
from luma.interface.exception import NotFittedError


__all__ = ['GridSearchCV']


class GridSearchCV:
    def __init__(self, 
                 model: Estimator, 
                 param_grid: dict, 
                 cv: int = 5, 
                 metric: Evaluator = None,
                 refit: bool = True, 
                 verbose: bool = False) -> None:
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.metric = metric
        self.refit = refit
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        best_score = None
        best_params = None
        param_combinations = self._get_param_combinations()
        max_iter = len(param_combinations)

        if self.verbose:
            print(f'Fitting {self.cv} folds for {max_iter} candidates,',
                  f'totalling {self.cv * max_iter} fits.\n')
        
        for i, params in enumerate(param_combinations, start=1):
            self.model.set_params(**params)
            scores = self._cross_validation(X, y)

            mean_score = np.mean(scores)
            if self.verbose:
                print(f'[{i}/{max_iter}] {params} - score:',
                      f'{mean_score:.3f}')
            if best_score is None or mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_score = best_score
        self.best_params = best_params
        
        if self.verbose:
            print(f'\n[GridSearchCV] Best params: {self.best_params}')
            print(f'[GridSearchCV] Best score: {self.best_score}')
        
        if self.refit:
            self.model.set_params(**best_params)
            self.model.fit(X, y)
        
        self._fitted = True
        return self.best_model

    def _cross_validation(self, X: np.ndarray, y: np.ndarray) -> list:
        num_samples = X.shape[0]
        fold_size = num_samples // self.cv
        scores = []

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for i in range(self.cv):
            start = i * fold_size
            end = start + fold_size if i < self.cv - 1 else num_samples

            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[0:start], indices[end:]))

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            self.model.fit(X_train, y_train)
            if self.metric: score = self.model.score(X_test, y_test, metric=self.metric)
            else: score = self.model.score(X_test, y_test)
            scores.append(score)
        
            if self.verbose:
                print(f'[GridSearchCV] fold {i + 1} - score: {score:.3f}')

        return scores

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        keys, values = zip(*self.param_grid.items())
        param_combinations = np.array(np.meshgrid(*values)).T.reshape(-1, len(keys))
        param_combinations = [dict(zip(keys, v)) for v in param_combinations]
        return param_combinations
    
    @property
    def best_model(self) -> Estimator:
        if not self._fitted: raise NotFittedError(self)
        return self.model

