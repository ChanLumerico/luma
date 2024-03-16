from itertools import combinations
from typing import Tuple
import numpy as np

from luma.core.super import Estimator, Evaluator, Transformer, Supervised
from luma.interface.util import Matrix, Vector, Clone
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.cv import CrossValidator
from luma.model_selection.fold import FoldType, KFold


__all__ = (
    'SBS',
    'SFS',
    'RFE'
)


class SBS(Transformer, Transformer.Feature, Supervised):
    
    """
    Sequential Backward Selection (SBS) is a feature selection technique 
    used in machine learning. It starts with all features and iteratively
    removes the least significant feature at each step_size. The goal is to 
    reduce dimensionality while maintaining or improving model performance. 
    The process continues until the desired number of features is reached 
    or performance criteria are met.
    
    Parameters
    ----------
    `estimator` : An estimator to fit and evaluate
    `n_features` : Number of features to select (`0~1` value for proportion)
    `metric` : Scoring metric for selecting features
    `test_size` : Proportional size of the validation set
    `cv` : K-fold size for cross validation (`0` to disable CV)
    `shuffle` : Whether to shuffle the dataset
    `stratify` : Whether to perform stratified split
    `fold_type` : Fold type (Default `KFold`)
    `random_state` : Seed for splitting the data
    
    Notes
    -----
    * An instance of the estimator must be passed to `estimator`
    * For `metric`, both class or instance are possible
    
    Examples
    --------
    >>> sbs = SBS(estimator=AnyEstimator(),
                  n_features=0.25,
                  metric=AnyEvaluator,
                  test_size=0.2,
                  cv=5,
                  random_state=None)
    >>> sbs.fit(X, y)
    >>> Z = sbs.transform(X)
    
    """
    
    def __init__(self, 
                 estimator: Estimator = None,
                 n_features: int | float = 1,
                 metric: Evaluator = None,
                 test_size: float = 0.2,
                 cv: int = 5,
                 shuffle: bool = True,
                 stratify: bool = False,
                 fold_type: FoldType = KFold,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.n_features = n_features
        self.metric = metric
        self.test_size = test_size
        self.cv = cv
        self.shuffle = shuffle
        self.stratify = stratify
        self.fold_type = fold_type
        self.random_state = random_state
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'SBS':
        _, n = X.shape
        self.estimator = Clone(self.estimator).get
        
        if 0 < self.n_features < 1:
            self.n_features = np.ceil(n * self.n_features).astype(int)
        elif self.n_features <= 0 or self.n_features > n:
            raise UnsupportedParameterError(self.n_features)
        
        if self.cv:
            cv_model = CrossValidator(estimator=self.estimator,
                                      metric=self.metric, 
                                      cv=self.cv,
                                      shuffle=self.shuffle,
                                      fold_type=self.fold_type,
                                      random_state=self.random_state)
        else:
            Xy_split = TrainTestSplit(X, y, 
                                      test_size=self.test_size,
                                      shuffle=self.shuffle,
                                      stratify=self.stratify,
                                      random_state=self.random_state).get
            X_train, X_test, y_train, y_test = Xy_split

        self.indices = tuple(range(n))
        self.subsets = [self.indices]
        
        if self.cv: _, score = cv_model.score(X, y)
        else: score = self._calculate_score(X_train, X_test, 
                                            y_train, y_test, indices=self.indices)
        self.scores = [score]
        
        iter = 1
        while n > self.n_features:
            scores = []
            subsets = []
            for p in combinations(self.indices, n - 1):
                if self.cv: _, score = cv_model.score(X[:, p], y)
                else: score = self._calculate_score(X_train, X_test, 
                                                    y_train, y_test, indices=p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            self.scores.append(scores[best])
            
            if self.verbose:
                print(f'[SBS] Feature added: {self.indices[-1]}', 
                      f'with score: {scores[best]}')
            
            n -= 1
            iter += 1
        
        self.n_score = self.scores[-1]
        if self.verbose:
            print(f'[SBS] Selection finished with final features: {self.indices}')
        
        self._fitted = True
        return self
    
    def _calculate_score(self, 
                         X_train: Matrix, X_test: Matrix, 
                         y_train: Matrix, y_test: Matrix, indices: Tuple) -> float:
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        
        return self.metric.score(y_true=y_test, y_pred=y_pred)
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return X[:, self.indices]
    
    def fit_transform(self, X: Matrix, y: Vector) -> Matrix:
        self.fit(X, y)
        return self.transform(X)


class SFS(Transformer, Transformer.Feature, Supervised):
    
    """
    Sequential Forward Selection (SFS) is a feature selection technique
    used in machine learning. It starts with no features and iteratively
    adds the most significant feature at each step. The goal is to
    reduce dimensionality while maintaining or improving model performance.
    The process continues until the desired number of features is reached
    or performance criteria are met.

    Parameters
    ----------
    `estimator` : An estimator to fit and evaluate
    `n_features` : Number of features to select (`0~1` value for proportion)
    `metric` : Scoring metric for selecting features
    `test_size` : Proportional size of the validation set
    `cv` : K-fold size for cross validation (`0` to disable CV)
    `shuffle` : Whether to shuffle the dataset
    `stratify` : Whether to perform stratified split
    `fold_type` : Fold type (Default `KFold`)
    `random_state` : Seed for splitting the data
    
    Notes
    -----
    * An instance of the estimator must be passed to `estimator`
    * Do not use the estimator for `SFS` if it is supposed to be the
        main estimator for the model
    * For `metric`, both class or instance are possible
    
    Examples
    --------
    >>> sfs = SFS(estimator=AnyEstimator(),
                  n_features=0.25,
                  metric=AnyEvaluator,
                  test_size=0.2,
                  cv=5,
                  random_state=None)
    >>> sfs.fit(X, y)
    >>> Z = sbs.transform(X)
    
    """

    def __init__(self, 
                 estimator: Estimator = None,
                 n_features: int | float = 1,
                 metric: Evaluator = None,
                 test_size: float = 0.2,
                 cv: int = 5,
                 shuffle: bool = True,
                 stratify: bool = False,
                 fold_type: FoldType = KFold,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.n_features = n_features
        self.metric = metric
        self.test_size = test_size
        self.cv = cv
        self.shuffle = shuffle
        self.stratify = stratify
        self.fold_type = fold_type
        self.random_state = random_state
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> 'SFS':
        _, n = X.shape
        self.estimator = Clone(self.estimator).get
        
        if isinstance(self.n_features, float) and 0 < self.n_features < 1:
            self.n_features = np.ceil(n * self.n_features).astype(int)
        elif self.n_features <= 0 or self.n_features > n:
            raise UnsupportedParameterError(self.n_features)

        if self.cv:
            cv_model = CrossValidator(estimator=self.estimator,
                                      metric=self.metric, 
                                      cv=self.cv,
                                      shuffle=self.shuffle,
                                      fold_type=self.fold_type,
                                      random_state=self.random_state)
        else:
            Xy_split = TrainTestSplit(X, y, 
                                      test_size=self.test_size,
                                      shuffle=self.shuffle,
                                      stratify=self.stratify,
                                      random_state=self.random_state).get
            X_train, X_test, y_train, y_test = Xy_split

        self.indices = ()
        self.subsets = [self.indices]
        score = -np.inf
        self.scores = [score]

        while len(self.indices) < self.n_features:
            scores = []
            subsets = []
            for p in combinations(range(n), len(self.indices) + 1):
                if set(self.indices).issubset(set(p)):
                    if self.cv: _, score = cv_model.score(X[:, p], y)
                    else: score = self._calculate_score(X_train, X_test, 
                                                        y_train, y_test, indices=p)
                    scores.append(score)
                    subsets.append(p)

            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            self.scores.append(scores[best])

            if self.verbose:
                print(f'[SFS] Feature added: {self.indices[-1]}', 
                      f'with score: {scores[best]}')

        self.n_score = self.scores[-1]
        if self.verbose:
            print(f'[SFS] Selection finished with final features: {self.indices}')

        self._fitted = True
        return self
    
    def _calculate_score(self, 
                         X_train: Matrix, X_test: Matrix, 
                         y_train: Matrix, y_test: Matrix, indices: Tuple) -> float:
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        
        return self.metric.score(y_true=y_test, y_pred=y_pred)
    
    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return X[:, self.indices]
    
    def fit_transform(self, X: Matrix, y: Vector) -> Matrix:
        self.fit(X, y)
        return self.transform(X)


class RFE(Transformer, Transformer.Feature, Supervised):
    
    """
    Recursive Feature Elimination (RFE) is a feature selection method 
    used in machine learning. It works by recursively removing the least 
    important features based on a given model's feature importance or 
    coefficients. RFE starts with all features and eliminates a specified 
    number or fraction of features at each step. The process continues 
    until the desired number of features is reached. This method helps in 
    enhancing model performance and interpretability by reducing complexity.
    
    Parameters
    ----------
    `estimator` : An estimator to fit and evaluate
    `n_features` : Number of features to select (`0~1` value for proportion)
    `metric` : Scoring metric for selecting features
    `step_size` : Number of features to eliminate in each step
    `cv` : K-fold size for cross validation (`0` to disable CV)
    `shuffle` : Whether to shuffle the dataset
    `fold_type` : Fold type (Default `KFold`)
    `random_state` : Seed for splitting the data
    
    Notes
    -----
    * An instance of the estimator must be passed to `estimator`
    * Do not use the estimator for `RFE` if it is supposed to be the
        main estimator for the model
    * For `metric`, both class or instance are possible
    
    Examples
    --------
    >>> rfe = RFE(estimator=AnyEstimator(),
                  n_features=0.25,
                  metric=AnyEvaluator,
                  step_size=1,
                  cv=5,
                  random_state=None)
    >>> rfe.fit(X, y)
    >>> Z = rfe.transform(X)
    
    """
    
    def __init__(self,
                 estimator: Estimator = None,
                 n_features: int | float = 1,
                 metric: Evaluator = None,
                 step_size: int = 1,
                 cv: int = 5,
                 shuffle: bool = True,
                 fold_type: FoldType = KFold,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.n_features = n_features
        self.metric = metric
        self.step_size = step_size
        self.cv = cv
        self.shuffle = shuffle
        self.fold_type = fold_type
        self.random_state = random_state
        self.verbose = verbose
        self._fitted = False

    def fit(self, X: Matrix, y: Vector) -> 'RFE':
        _, n = X.shape
        self.estimator = Clone(self.estimator).get
        
        if 0 < self.n_features < 1:
            self.n_features = np.floor(n * self.n_features).astype(int)
        elif self.n_features <= 0 or self.n_features > n:
            raise UnsupportedParameterError(self.n_features)
        
        self.cv_model = CrossValidator(estimator=self.estimator, 
                                       metric=self.metric,
                                       cv=self.cv,
                                       shuffle=self.shuffle,
                                       fold_type=self.fold_type,
                                       random_state=self.random_state)

        self.support = np.ones(n, dtype=bool)
        self.ranking = np.ones(n, dtype=int)
        
        while np.sum(self.support) > self.n_features:
            scores = self._calculate_score(X, y)
            indices = np.argsort(scores)[::-1][:self.step_size]
            self.support[indices] = False
            self.ranking[indices] += 1

            if self.verbose:
                print(f'[RFE] Removed {self.step_size} features,', 
                      f'remaining features: {np.sum(self.support)}',
                      f'with best-score: {scores[:self.step_size]}')
        
        features = tuple(i for i, sup in enumerate(self.support) if sup)
        if self.verbose:
            print(f'[RFE] Elimination finished with final features: {features}')
        
        self._fitted = True
        return self

    def _calculate_score(self, X: Matrix, y: Vector) -> Vector[float]:
        scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if not self.support[i]: continue
            indices = self.support.copy()
            indices[i] = False
            
            if self.cv:
                _, score = self.cv_model.score(X[:, indices], y)
            else:
                self.estimator.fit(X[:, indices], y)
                y_pred = self.estimator.predict(X[:, indices])
                score = self.metric.score(y_true=y, y_pred=y_pred)
                
            scores[i] = score
        
        return scores

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return X[:, self.support]

    def fit_transform(self, X: Matrix, y: Vector) -> Matrix:
        self.fit(X, y)
        return self.transform(X)

