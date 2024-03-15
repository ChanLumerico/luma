from typing import Generator, Tuple
import numpy as np

from luma.core.super import Estimator, Evaluator
from luma.interface.util import Matrix, Vector


__all__ = (
    'KFold',
    'StratifiedKFold',
    'CrossValidator'
)


class KFold:
    
    """
    K-Fold cross-validation is a model evaluation method that divides 
    the dataset into `k` equal or nearly equal sized folds. In each 
    of `k` iterations, one fold is used as the test set, and the 
    remaining `k-1` folds are combined to form the training set. This 
    process ensures that every data point gets to be in the test set 
    exactly once and in the training set `k-1` times. It's widely used 
    to assess the performance of a model with limited data, providing 
    a robust estimate of its generalization capability.
    
    Parameters
    ----------
    `X` : Input data
    `y` : Target data
    `n_folds` : Number of folds
    `shuffle` : Whether to shuffle the dataset
    `random_state` : Seed for random shuffling
    
    Properties
    ----------
    ```py
    @property
    def split(self)
    ```
    This returns a generator with the type:
    ```py
    Generator[Tuple[Matrix, Vector, Matrix, Vector], None, None]
    ```
    yielding `X_train`, `y_train`, `X_test`, `y_test`.
    
    Examples
    --------
    Usage of the generator returned by the property `split`:
    
    ```py
    kfold = KFold(X, y, n_folds=5, shuffle=True)
    for X_train, y_train, X_test, y_test in kfold.split:
        ...
    ```
    """
    
    def __init__(self,
                 X: Matrix, 
                 y: Vector,
                 n_folds: int = 5,
                 shuffle: bool = True,
                 random_state: int = None) -> None:
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_folds = n_folds
    
    @property
    def split(self) -> Generator[Tuple[Matrix, Vector, 
                                       Matrix, Vector], None, None]:
        np.random.seed(self.random_state)
        m, _ = self.X.shape
        fold_size = m // self.n_folds

        indices = np.arange(m)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else m
            
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))
            
            X_train, X_test = self.X[train_indices], self.X[test_indices]
            y_train, y_test = self.y[train_indices], self.y[test_indices]
            
            yield X_train, y_train, X_test, y_test


class StratifiedKFold:
    
    """
    Stratified K-Fold cross-validation is a model evaluation method similar 
    to KFold but it divides the dataset in a way that preserves the 
    percentage of samples for each class. This is especially useful for 
    handling imbalances in the dataset. It ensures each fold is a good 
    representative of the whole by maintaining the same class distribution 
    in each fold as in the complete dataset.

    Parameters
    ----------
    `X` : Input data
    `y` : Target data
    `n_folds` : Number of folds
    `shuffle` : Whether to shuffle the dataset
    `random_state` : Seed for random shuffling

    Properties
    ----------
    ```py
    @property
    def split(self)
    ```
    Returns a generator with the type:
    ```py
    Generator[Tuple[Matrix, Vector, Matrix, Vector], None, None]
    ```
    yielding `X_train`, `y_train`, `X_test`, `y_test`.
    
    Examples
    --------
    Usage of the generator returned by the property `split`:
    
    ```py
    kfold = StratifiedKFold(X, y, n_folds=5, shuffle=True)
    for X_train, y_train, X_test, y_test in kfold.split:
        ...
    ```
    """
    
    def __init__(self, 
                 X: Matrix, 
                 y: Vector, 
                 n_folds: int = 5, 
                 shuffle: bool = True, 
                 random_state: int = None) -> None:
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
    
    @property
    def split(self) -> Generator[Tuple[Matrix, Vector, 
                                       Matrix, Vector], None, None]:
        ...


class CrossValidator(Evaluator):
    
    """
    Cross-validation is a technique in machine learning for assessing how 
    a model generalizes to an independent dataset. It involves dividing 
    the data into several parts, training the model on some parts and 
    testing it on others. This process is repeated multiple times with 
    different partitions, providing a robust estimate of the model's 
    performance. It's especially useful in scenarios with limited data, 
    ensuring that all available data is effectively utilized for both 
    training and validation.
    
    Parameters
    ----------
    `estimator` : An estimator to validate
    `metric` : Evaluation metric for validation
    `cv` : Number of folds in splitting data
    `fold_generator` : Generator for yielding a single fold
    (uses `KFold`'s when set to `None`)
    `shuffle` : Whether to shuffle the dataset
    `random_state` : Seed for random sampling upon splitting data
    
    Attributes
    ----------
    `train_scores_` : List of training scores for each fold
    `test_scores_` : List of test(validation) scores for each fold
    
    Methods
    -------
    For getting mean train score and mean test score respectvely:
    ```py
        def score(self, X: Matrix, y: Vector) -> Tuple[float, float]
    ```
    """
    
    def __init__(self,
                 estimator: Estimator,
                 metric: Evaluator,
                 cv: int = 5,
                 fold_generator: Generator = None, 
                 shuffle: bool = True,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.metric = metric
        self.cv = cv
        self.fold_generator = fold_generator
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.train_scores_ = []
        self.test_scores_ = []
    
    def _fit(self, X: Matrix, y: Vector) -> None:
        if self.fold_generator is None:
            kfold = KFold(X, y, self.cv, self.shuffle, self.random_state)
            self.fold_generator = kfold.split
            
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.fold_generator):
            self.estimator.fit(X_train, y_train)
            if self.metric: 
                train_score = self.estimator.score(X_train, y_train, self.metric)
                test_score = self.estimator.score(X_test, y_test, self.metric)
            else: 
                train_score = self.estimator.score(X_train, y_train)
                test_score = self.estimator.score(X_test, y_test)
            
            self.train_scores_.append(train_score)
            self.test_scores_.append(test_score)
            
            if self.verbose:
                print(f'[CV] fold {i + 1} -',
                      f'train-score: {train_score:.3f},',
                      f'test-score: {test_score:.3f}')
        
    def score(self, X: Matrix, y: Vector) -> Tuple[float, float]:
        self._fit(X, y)
        return np.mean(self.train_scores_), np.mean(self.test_scores_)

