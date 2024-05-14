from typing import Iterator, Tuple
import numpy as np

from luma.interface.typing import Matrix, Vector, TensorLike


__all__ = ("TrainTestSplit", "BatchGenerator")


class TrainTestSplit:
    """
    Splits the original dataset into the train set and the test set.

    Parameters
    ----------
    `X` : Matrix
        Feature data
    `y` : Vector
        Target data
    `test_size` : int or float, default=0.3
        Proportional size of the test set
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `stratify` : bool, default=False
        Whether to perform stratified split
    `random_state` : int, optional, default=None
        Seed for random sampling for split

    Properties
    ----------
    `get` : tuple[Matrix, Matrix, Vector, Vector]
        Returns the split data as a 4-tuple

    Notes
    -----
    For `stratify == True`, one-hot encoded `y` is not compatible.

    Examples
    --------
    ```py
    X_train, X_test, y_train, y_test = TrainTestSplit(X, y, ...).get
    ```
    """

    def __init__(
        self,
        X: Matrix,
        y: Vector,
        test_size: int | float = 0.3,
        shuffle: bool = True,
        stratify: bool = False,
        random_state: int = None,
    ) -> None:
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

        if self.stratify:
            return self._stratified_split()
        else:
            return self._split()

    def _split(self) -> Tuple[Matrix, Matrix, Vector, Vector]:
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)

        if isinstance(self.test_size, float):
            num_test_samples = int(self.test_size * n_samples)
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
            class_mask = y_indices == class_index
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


class BatchGenerator:
    """
    A class for generating mini-batches of data for training machine
    learning models including neural networks.

    Parameters
    ----------
    `X` : TensorLike
        Input features
    `y` : TensorLike
        Targets or labels
    `batch_size` : int, default=100
        Size of a mini-batch
    `shuffle` : bool, default=True
        Whether to shuffle the data for every batch generation

    Iterator
    --------
    The iterator of `BatchGenerator` returns a 2-tuple of
    `TensorLike` objects.
    ```py
    def __iter__(self) -> Iterator[Tuple[TensorLike, TensorLike]]
    ```
    Examples
    --------
    An instance of `BatchGenerator` can be used as an iterator.

    - With instantiation:

        ```py
        batch_gen = BatchGenerator(X, y, batch_size=100)
        for X_batch, y_batch in batch_gen:
            pass
        ```
    - Without instantiation:

        ```py
        for X_batch, y_batch in BatchGenerator(X, y, batch_size=100):
            pass
        ```
    """

    def __init__(
        self,
        X: TensorLike,
        y: TensorLike,
        batch_size: int = 100,
        shuffle: bool = True,
    ) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_samples = X.shape[0]
        self.n_batches = self.n_samples // batch_size

        if self.n_samples % batch_size != 0:
            self.n_batches += 1

        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self) -> Iterator[Tuple[TensorLike, TensorLike]]:
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)

            batch_indices = self.indices[start_idx:end_idx]
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]

            yield X_batch, y_batch
