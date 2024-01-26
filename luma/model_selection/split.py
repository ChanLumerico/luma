from typing import Tuple
import numpy as np

from luma.interface.util import Matrix, Vector


__all__ = (
    'TrainTestSplit'
)


class TrainTestSplit:
    
    """
    Splits the original dataset into the train set and the test set.
    
    Parameters
    ----------
    `X` : Feature data
    `y` : Target data (as a 1-D `Vector`)
    `test_size` : Proportional size of the test set (e.g. `0.2`, `0.3`)
    `random_state` : Seed for random sampling for split
    
    Properties
    ----------
    `get` : Returns the split data as a 4-tuple
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = TrainTestSplit(X, y, ...).get
    
    """
    
    def __init__(self, 
                 X: Matrix,
                 y: Vector,
                 test_size: int | float = 0.3,
                 random_state: int = None) -> None:
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
    
    @property
    def get(self) -> Tuple[Matrix, Matrix, Vector, Vector]:
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Sample size mismatch between 'X' and 'y'!")
        
        num_samples = self.X.shape[0]
        indices = np.arange(num_samples)
        
        if self.test_size < 1.0:
            num_test_samples = int(self.test_size * num_samples)
        else:
            num_test_samples = self.test_size
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        np.random.shuffle(indices)

        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]

        X_train = self.X[train_indices]
        X_test = self.X[test_indices]
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]

        return X_train, X_test, y_train, y_test

