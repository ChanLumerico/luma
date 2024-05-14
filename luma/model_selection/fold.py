from collections import Counter, defaultdict
from itertools import chain
from typing import Generator, Tuple, Union
import numpy as np

from luma.interface.typing import Matrix, Vector

FoldType = Union["KFold", "StratifiedKFold", "GroupKFold"]


__all__ = ("KFold", "StratifiedKFold", "GroupKFold")


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
    `X` : Matrix
        Input data
    `y` : Vector
        Target data
    `n_folds` : int, default=5
        Number of folds
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `random_state` : int, optional, default=True
        Seed for random shuffling

    Properties
    ----------
    ```py
    @property
    def split(self)
    ```
    This returns a generator with the type:
    ```py
    Generator[Tuple[Vector, Vector], None, None]
    ```
    yielding train indices and test indices.

    Examples
    --------
    Usage of the generator returned by the property `split`:
    ```py
    kfold = KFold(X, y, n_folds=5, shuffle=True)

    for train_indices, test_indices in kfold.split:
        X_train, y_train = X[train_indices], y[train_indices]
        ...
    ```
    """

    def __init__(
        self,
        X: Matrix,
        y: Vector,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
    ) -> None:
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_folds = n_folds

    @property
    def split(self) -> Generator[Tuple[Vector, Vector], None, None]:
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

            yield train_indices, test_indices


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
    `X` : Matrix
        Input data
    `y` : Vector
        Target data
    `n_folds` : int, default=5
        Number of folds
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `random_state` : int, optional, default=None
        Seed for random shuffling

    Properties
    ----------
    ```py
    @property
    def split(self)
    ```
    This returns a generator with the type:
    ```py
    Generator[Tuple[Vector, Vector], None, None]
    ```
    yielding train indices and test indices.

    Examples
    --------
    Usage of the generator returned by the property `split`:
    ```py
    kfold = StratifiedKFold(X, y, n_folds=5, shuffle=True)

    for train_indices, test_indices in kfold.split:
        X_train, y_train = X[train_indices], y[train_indices]
        ...
    ```
    """

    def __init__(
        self,
        X: Matrix,
        y: Vector,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
    ) -> None:
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    @property
    def split(self) -> Generator[Tuple[Vector, Vector], None, None]:
        np.random.seed(self.random_state)

        indices_per_class = defaultdict(list)
        for idx, class_ in enumerate(self.y):
            indices_per_class[class_].append(idx)

        if self.shuffle:
            for indices in indices_per_class.values():
                np.random.shuffle(indices)

        folds = defaultdict(list)
        for class_, indices in indices_per_class.items():
            fold_size = len(indices) // self.n_folds
            for i in range(self.n_folds):
                start = i * fold_size
                end = start + fold_size if i < self.n_folds - 1 else len(indices)
                folds[i].extend(indices[start:end])

        if self.shuffle:
            for fold_indices in folds.values():
                np.random.shuffle(fold_indices)

        for i in range(self.n_folds):
            test_indices = folds[i]
            train_indices = list(
                chain.from_iterable(folds[j] for j in range(self.n_folds) if j != i)
            )

            yield Vector(train_indices), Vector(test_indices)


class GroupKFold:
    """
    Group K-Fold cross-validation provides train/test indices to split data into
    train/test sets. Each set contains group samples that are entirely in the set
    of training or test. This cross-validation object is a variation of KFold
    that ensures samples from the same group are not split between training and
    testing sets.

    Parameters
    ----------
    `X` : Matrix
        Input data
    `y` : Vector
        Target data
    `groups` : Vector
        Array of group labels for the samples
    `n_folds` : int, default=5
        Number of folds. Must be at least 2.
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `random_state` : int, optional, default=None
        Seed for random shuffling

    Properties
    ----------
    ```py
    @property
    def split(self)
    ```
    This returns a generator with the type:
    ```py
    Generator[Tuple[Vector, Vector], None, None]
    ```
    yielding train indices and test indices.

    Examples
    --------
    Usage of the generator returned by the property `split`:
    ```py
    kfold = GroupKFold(X, y, n_folds=5, shuffle=True)

    for train_indices, test_indices in kfold.split:
        X_train, y_train = X[train_indices], y[train_indices]
        ...
    ```
    """

    def __init__(
        self,
        X: Matrix,
        y: Vector,
        groups: Vector,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
    ) -> None:
        self.X = X
        self.y = y
        self.groups = groups
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    @property
    def split(self) -> Generator[Tuple[Vector, Vector], None, None]:
        np.random.seed(self.random_state)
        _, groups_indices = np.unique(self.groups, return_inverse=True)

        group_counts = Counter(groups_indices)
        group_indices = defaultdict(list)
        for idx, group_idx in enumerate(groups_indices):
            group_indices[group_idx].append(idx)

        sorted_groups = sorted(group_counts, key=group_counts.get, reverse=True)

        if self.shuffle:
            np.random.shuffle(sorted_groups)

        folds = [[] for _ in range(self.n_folds)]
        for group in sorted_groups:
            smallest = min(range(self.n_folds), key=lambda x: len(folds[x]))
            folds[smallest].extend(group_indices[group])

        for i in range(self.n_folds):
            test_indices = folds[i]
            train_indices = list(set(range(len(self.X))) - set(test_indices))

            yield Vector(train_indices), Vector(test_indices)
