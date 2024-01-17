import numpy as np

from luma.interface.super import Estimator, Evaluator
from luma.interface.util import Matrix, Vector


__all__ = (
    'CrossValidator'
)


class CrossValidator(Evaluator):
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
    
    def score(self, X: Matrix, y: Vector) -> float:
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

            self.estimator.fit(X_train, y_train)
            if self.metric: score = self.estimator.score(X_test, y_test, metric=self.metric)
            else: score = self.estimator.score(X_test, y_test)
            scores.append(score)
        
            if self.verbose:
                print(f'[CV] fold {i + 1} - score: {score:.3f}')
        
        return np.mean(scores)

    def compute(self, X: Matrix, y: Vector) -> float:
        return self.score(X, y)

