from typing import Any, Dict, List, TypeVar
from scipy.stats import rv_continuous, rv_discrete
import numpy as np
import random

from luma.interface.util import Matrix, Vector
from luma.core.super import Estimator, Evaluator, Optimizer
from luma.interface.exception import NotFittedError
from luma.model_selection.cv import CrossValidator
from luma.model_selection.fold import FoldType, KFold

DT = TypeVar('DT', bound=rv_continuous | rv_discrete)


__all__ = (
    'GridSearchCV',
    'RandomizedSearchCV'
)


class GridSearchCV(Optimizer):
    
    """
    Grid seach with cross validation(CV) is a method used to tune hyperparameters 
    of a model by searching through a specified parameter grid. It evaluates all 
    combinations of parameter values to find the best combination. This process 
    is performed using cross-validation to ensure model performance is not biased 
    to a specific data split. The result is the optimal set of parameters that 
    yield the best model performance according to a chosen metric.
    
    Parameters
    ----------
    `estimator` : An estimator to fit and evaluate
    `param_grid` : Parameter grid for repetitive search
    `cv` : K-fold size for cross validation
    `metric` : Scoring metric for evaluation
    `maximize` : Whether to optimize in a way of maximizing certain metric
    `shuffle` : Whether to shuffle the dataset
    `fold_type` : Fold type (Default `KFold`)
    `refit` : Whether to re-fit the estimator with the best parameters found
    `random_state` : Seed for random sampling for cross validation
    
    Properties
    ----------
    Get the best(optimized) estimator:
        ```py
        @property
        def best_model(self) -> Estimator
        ```
    
    Notes
    -----
    * An instance of the estimator must be passed to `estimator`
    * For `metric`, both class or instance are possible
    * Only `Pipeline` is allowed for meta estimator
    
    Examples
    --------
    >>> param_grid = {'param_1': [...], 
                      'param_2': [...],
                      ...,
                      AnyStr: List[Any]}
    
    >>> grid = GridSearchCV(estimator=AnyEstimator(),
                            param_grid=param_grid,
                            cv=5,
                            metric=AnyEvaluator,
                            refit=True,
                            random_state=None)
    >>> grid.fit(X, y)
    >>> score = grid.best_score
    >>> model = grid.best_model
    
    """
    
    def __init__(self, 
                 estimator: Estimator, 
                 param_grid: dict, 
                 cv: int = 5, 
                 metric: Evaluator = None,
                 maximize: bool = True, 
                 refit: bool = True, 
                 shuffle: bool = True,
                 fold_type: FoldType = KFold, 
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.metric = metric
        self.maximize = maximize
        self.refit = refit
        self.shuffle = shuffle
        self.fold_type = fold_type
        self.random_state = random_state
        self.verbose = verbose
        self.scores_ = []
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> Estimator:
        best_score = None
        best_params = None
        param_combinations = self._get_param_combinations()
        max_iter = len(param_combinations)
        
        if self.verbose:
            print(f'Fitting {self.cv} folds for {max_iter} candidates,',
                  f'totalling {self.cv * max_iter} fits.\n')
        
        for i, params in enumerate(param_combinations, start=1): 
            self._set_params(params)
            cv_model = CrossValidator(estimator=self.estimator,
                                      metric=self.metric,
                                      cv=self.cv,
                                      shuffle=self.shuffle,
                                      fold_type=self.fold_type,
                                      random_state=self.random_state,
                                      verbose=self.verbose)
            
            _, mean_score = cv_model.score(X, y)
            self.scores_.append((params, mean_score))
            
            if self.verbose:
                params_f = {k: f'{v:.4f}' if isinstance(v, float) else v 
                            for k, v in params.items()}
                print(f'[GridSearchCV] candidate {i}/{max_iter} {params_f}',
                      f'- score:{mean_score:.3f}')
            
            if self.maximize:
                if best_score is None or mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            else:
                if best_score is None or mean_score < best_score:
                    best_score = mean_score
                    best_params = params

        self.best_score = best_score
        self.best_params = best_params
        
        if self.verbose:
            print(f'\n[GridSearchCV] Best params: {self.best_params}')
            print(f'[GridSearchCV] Best score: {self.best_score}')
        
        if self.refit:
            self.estimator = type(self.estimator)()
            self._set_params(best_params)
            self.estimator.fit(X, y)
        
        self._fitted = True
        return self.best_model
    
    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        keys, values = zip(*self.param_grid.items())
        param_combinations = Matrix(np.meshgrid(*values)).T.reshape(-1, len(keys))
        param_combinations = [dict(zip(keys, v)) for v in param_combinations]
        return param_combinations
    
    def _set_params(self, params: dict) -> None:
        self.estimator.set_params(**params)
    
    @property
    def best_model(self) -> Estimator:
        if not self._fitted: raise NotFittedError(self)
        return self.estimator


class RandomizedSearchCV(Optimizer):
    
    """
    Randomized search with cross-validation (CV) is a method used to tune 
    hyperparameters of a model by searching through a specified parameter 
    distribution. It randomly samples combinations of parameter values to find 
    the best combination. This process is performed using cross-validation to 
    ensure model performance is not biased to a specific data split. The result 
    is the optimal set of parameters that yield the best model performance 
    according to a chosen metric.

    Parameters
    ----------
    `estimator` : An estimator to fit and evaluate
    `param_dist` : Parameter distributions for random search
    `max_iter` : Number of parameter settings that are sampled
    `cv` : K-fold size for cross-validation
    `metric` : Scoring metric for evaluation
    `maximize` : Whether to optimize in a way of maximizing certain metric
    `refit` : Whether to re-fit the estimator with the best parameters found
    `shuffle` : Whether to shuffle the dataset
    `fold_type` : Fold type (Default `KFold`)
    `random_state` : Seed for random sampling for cross-validation and random search
    `verbose` : Whether to print progress messages
    
    Properties
    ----------
    Get the best(optimized) estimator:
        ```py
        @property
        def best_model(self) -> Estimator
        ```
    
    Notes
    -----
    * An instance of the estimator must be passed to `estimator`
    * For `metric`, both class or instance are possible
    * Only `Pipeline` is allowed for meta estimator
    * Type `DT` is a generic type for `scipy`'s distribution types
        (e.g. `rv_continuous`, `rv_discrete`)
    
    Examples
    --------
    >>> param_dist = {'param_1': [...], 
                      'param_2': [...],
                      ...,
                      AnyStr: Vector | DT }

    >>> rand = RandomizedSearchCV(estimator=AnyEstimator(),
                                  param_dist=param_dist,
                                  max_iter=100,
                                  cv=5,
                                  metric=AnyEvaluator,
                                  refit=True,
                                  random_state=None)
    >>> rand.fit(X, y)
    >>> score = rand.best_score
    >>> model = rand.best_model
    
    """
    
    def __init__(self,
                 estimator: Estimator,
                 param_dist: Dict[str, Vector | DT],
                 max_iter: int = 100,
                 cv: int = 5,
                 metric: Evaluator = None,
                 maximize: bool = True, 
                 refit: bool = True,
                 shuffle: bool = True,
                 fold_type: FoldType = KFold,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.param_dist = param_dist
        self.max_iter = max_iter
        self.cv = cv
        self.metric = metric
        self.maximize = maximize
        self.refit = refit
        self.shuffle = shuffle
        self.fold_type = fold_type
        self.random_state = random_state
        self.verbose = verbose
        self.scores_ = []
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> Estimator:
        best_score = None
        best_params = None
        
        if self.verbose:
            print(f'Fitting {self.cv} folds for {self.max_iter} candidates,',
                  f'toatlling {self.cv * self.max_iter} fits.\n')
        
        for i in range(self.max_iter):
            params = self._sample_params()
            self._set_params(params)
            
            cv_model = CrossValidator(estimator=self.estimator,
                                      metric=self.metric,
                                      cv=self.cv,
                                      shuffle=self.shuffle,
                                      fold_type=self.fold_type,
                                      random_state=self.random_state,
                                      verbose=self.verbose)

            _, mean_score = cv_model.score(X, y)
            self.scores_.append((params, mean_score))
            
            if self.verbose:
                params_f = {k: f'{v:.4f}' if isinstance(v, float) else v 
                            for k, v in params.items()}
                print(f'[RandomSearchCV] candidate {i + 1}/{self.max_iter}',
                      f'{params_f} - score: {mean_score:.3f}')
            
            if self.maximize:
                if best_score is None or mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            else:
                if best_score is None or mean_score < best_score:
                    best_score = mean_score
                    best_params = params
        
        self.best_score = best_score
        self.best_params = best_params
        
        if self.verbose:
            print(f'\n[RandomSearchCV] Best params: {self.best_params}')
            print(f'[RandomSearchCV] Best score: {self.best_score}')
        
        if self.refit:
            self.estimator = type(self.estimator)()
            self._set_params(best_params)
            self.estimator.fit(X, y)
        
        self._fitted = True
        return self.best_model

    def _set_params(self, params: dict) -> None:
        self.estimator.set_params(**params)
        
    def _sample_params(self) -> Dict[str, Any]:
        params = {}
        for key, dist in self.param_dist.items():
            if hasattr(dist, 'rvs'): params[key] = dist.rvs()
            else: params[key] = random.choice(dist)
        
        return params

    @property
    def best_model(self) -> Estimator:
        if not self._fitted: raise NotFittedError(self)
        return self.estimator

