from typing import Tuple
import numpy as np


__all__ = ['TrainTestSplit']


class TrainTestSplit:
    @staticmethod
    def split(X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, 
              random_state: int = None) -> Tuple[np.ndarray]:
        if X.shape[0] != y.shape[0]:
            raise ValueError()
        
        num_samples = X.shape[0]
        num_test_samples = int(test_size * num_samples)
        indices = np.arange(num_samples)
        
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

