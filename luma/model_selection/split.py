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
    `shuffle` : Whether to shuffle the dataset
    `stratify` : Whether to perform stratified split (Default `False`)
    `random_state` : Seed for random sampling for split
    
    Properties
    ----------
    `get` : Returns the split data as a 4-tuple
    
    Examples
    --------
    ```py
    X_train, X_test, y_train, y_test = TrainTestSplit(X, y, ...).get
    ```
    """
    
    def __init__(self, 
                 X: Matrix,
                 y: Vector,
                 test_size: int | float = 0.3,
                 shuffle: bool = True,
                 stratify: bool = False,
                 random_state: int = None) -> None:
        self.X = X
        self.y = y
        self.test_size = test_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_state = random_state
    
    @property
    def get(self) -> Tuple[Matrix, Matrix, Vector, Vector]:
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Sample size mismatch between 'X' and 'y'!")
        
        if self.stratify: return self._stratified_split()
        else: return self._split()
    
    def _split(self) -> Tuple[Matrix, Matrix, Vector, Vector]:
        num_samples = self.X.shape[0]
        indices = np.arange(num_samples)
        
        if isinstance(self.test_size, float):
            num_test_samples = int(self.test_size * num_samples)
        else:
            num_test_samples = self.test_size
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.shuffle:
            np.random.shuffle(indices)

        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]

        X_train = self.X[train_indices]
        X_test = self.X[test_indices]
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]

        return X_train, X_test, y_train, y_test
    
    def _stratified_split(self) -> Tuple[Matrix, Matrix, Vector, Vector]:
        if isinstance(self.test_size, float):
            test_size_func = lambda u: int(self.test_size * len(u))
        else:
            total_size = len(self.y)
            proportion = self.test_size / total_size
            test_size_func = lambda u: int(proportion * len(u))
        
        unique_classes, y_indices = np.unique(self.y, return_inverse=True)
        train_indices, test_indices = [], []
        
        for class_index in unique_classes:
            class_mask = (y_indices == class_index)
            class_indices = np.where(class_mask)[0]
            
            if self.shuffle:
                np.random.shuffle(class_indices)
            
            num_test_samples = test_size_func(class_indices)
            test_indices.extend(class_indices[:num_test_samples])
            train_indices.extend(class_indices[num_test_samples:])
        
        if self.shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        X_train = self.X[train_indices]
        X_test = self.X[test_indices]
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]

        return X_train, X_test, y_train, y_test

