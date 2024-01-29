from typing import Tuple
import numpy as np

from luma.core.super import Estimator, Evaluator
from luma.interface.util import Matrix, Vector


__all__ = (
    'CrossValidator'
)


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
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.estimator = estimator
        self.metric = metric
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.train_scores_ = []
        self.test_scores_ = []
    
    def _fit(self, X: Matrix, y: Vector) -> None:
        num_samples = X.shape[0]
        fold_size = num_samples // self.cv

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in range(self.cv):
            start = i * fold_size
            end = start + fold_size if i < self.cv - 1 else num_samples
            
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
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
                      f'train-score: {train_score:.3f}, test-score: {test_score:.3f}')
        
    def score(self, X: Matrix, y: Vector) -> Tuple[float, float]:
        self._fit(X, y)
        return np.mean(self.train_scores_), np.mean(self.test_scores_)

