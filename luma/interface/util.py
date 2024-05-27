from typing import Any, Callable, Iterable, Literal, Self, Type, TypeGuard
from rich.progress import Progress, BarColumn, TextColumn
import numpy as np

from luma.interface.exception import UnsupportedParameterError, InvalidRangeError
from luma.interface.typing import Matrix, Scalar, Vector
from luma.neural import init


__all__ = (
    "DecisionTreeNode",
    "NearestNeighbors",
    "SilhouetteUtil",
    "DBUtil",
    "KernelUtil",
    "ActivationUtil",
    "InitUtil",
    "Clone",
    "ParamRange",
    "TrainProgress",
)


class DecisionTreeNode:
    """
    Internal class for node used in tree-based models.

    Parameters
    ----------
    `feature_index` : int, optional, default=None
        Feature of node
    `threshold` : float, optional, default=None
        Threshold for split point
    `left` : Self, optional, default=None
        Left-child node
    `right` : Self, optional, default=None
        Right-child node
    `value` : Any, optional, default=None
        Most popular label of leaf node

    """

    def __init__(
        self,
        feature_index: int | None = None,
        threshold: float | None = None,
        left: Self | None = None,
        right: Self | None = None,
        value: Any | None = None,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def isLeaf(self) -> bool:
        return self.value is not None


class NearestNeighbors:
    """
    Internal class for computing nearest neighbors of given data.

    Parameters
    ----------
    `data` : Matrix
        Data to be handled
    `n_neighbors` : int
        Number of nearest neighbors

    """

    def __init__(self, data: Matrix, n_neighbors: int) -> None:
        self.data = data
        self.n_neighbors = n_neighbors
        self._size = data.shape[0]

    @property
    def index_matrix(self) -> Matrix:
        data = self.data
        dist = np.linalg.norm(data[:, np.newaxis, :] - data, axis=2)
        sorted_indices = np.argsort(dist, axis=1)
        return sorted_indices[:, 1 : self.n_neighbors + 1]

    @property
    def adjacent_matrix(self) -> Matrix:
        indices = self.index_matrix
        adj_mat = np.zeros((self._size, self._size))
        for i in range(self._size):
            adj_mat[i, indices[i]] = 1

        return adj_mat.astype(int)


class SilhouetteUtil:
    """
    Internal class for computing various distances used in
    Silhouette Coefficient calculation.

    Parameters
    ----------
    `idx` : int
        Index of a single data point
    `cluster` : int
        Current cluster number
    `labels` : Vector
        Labels assigned by clustering estimator
    `distances` : Matrix
        Square-form distance matrix of the data

    """

    def __init__(
        self, idx: int, cluster: int, labels: Vector, distances: Matrix
    ) -> None:
        self.idx = idx
        self.cluster = cluster
        self.labels = labels
        self.distances = distances

    @property
    def avg_dist_others(self) -> Matrix:
        others = set(self.labels) - {self.cluster}
        sub_avg = [
            np.mean(
                self.distances[self.idx][self.labels == other],
            )
            for other in others
        ]

        return np.mean(sub_avg)

    @property
    def avg_dist_within(self) -> Matrix | int:
        within_cluster = self.distances[self.idx][self.labels == self.cluster]
        if len(within_cluster) <= 1:
            return 0
        return np.mean([dist for dist in within_cluster if dist != 0])


class DBUtil:
    """
    Internal class for supporting Davies-Bouldin Index (DBI) computation.

    Parameters
    ----------
    `data` : Matrix
        Original data
    `labels` : Vector
        Labels assigned by clustering estimator

    """

    def __init__(self, data: Matrix, labels: Vector) -> None:
        self.data = data
        self.labels = labels

    @property
    def cluster_centroids(self) -> Matrix:
        unique_labels = np.unique(self.labels)
        centroids = [
            self.data[self.labels == label].mean(axis=0) for label in unique_labels
        ]
        return Matrix(centroids)

    @property
    def within_cluster_scatter(self) -> Matrix:
        centroids = self.cluster_centroids
        scatter = np.zeros(len(centroids))

        for i, centroid in enumerate(centroids):
            cluster_points = self.data[self.labels == i]
            diff_sq = (cluster_points - centroid) ** 2
            scatter[i] = np.mean(np.sqrt(np.sum(diff_sq, axis=1)))

        return scatter

    @property
    def separation(self) -> Matrix:
        centroids = self.cluster_centroids
        n_clusters = len(centroids)
        separation = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    continue
                diff_sq = (centroids[i] - centroids[j]) ** 2
                separation[i, j] = np.sqrt(np.sum(diff_sq))

        return separation


class KernelUtil:
    """
    Internal class for kernel methods(tricks).

    This class facilitates transferring kernel type strings
    into actual specific kernel function.

    Parameters
    ----------
    `kernel` : FuncType
        Type of kernel
    `alpha` : float, default=1.0
        Shape parameter for RBF and sigmoid kernels
     `gamma` : float, default=0.0
        Shape parameter of Gaussian curve for RBF kernel
    `coef` : float, default=1.0
        Coefficient for polynomial and sigmoid kernel
    `deg` : int, default=3
        Polynomial Degree for polynomial kernel

    Example
    -------
    >>> util = KernelUtil(kernel='rbf', **params)
    >>> util.kernel_func
    KernelUtil.rbf_kernel: Callable[[Matrix, Matrix | None], Matrix]

    """

    FuncType = Literal[
        "lin",
        "linear",
        "poly",
        "polynoimal",
        "rbf",
        "gaussian",
        "Gaussian",
        "tanh",
        "sigmoid",
        "lap",
        "laplacian",
    ]

    def __init__(
        self,
        kernel: FuncType,
        alpha: float = 1.0,
        gamma: float = 1.0,
        coef: float = 0.0,
        deg: int = 2,
    ) -> None:
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.coef = coef
        self.deg = deg

    def linear_kernel(self, Xi: Matrix, Xj: Matrix = None) -> Matrix:
        if Xj is None:
            Xj = Xi.copy()
        return np.dot(Xi, Xj.T)

    def polynomial_kernel(self, Xi: Matrix, Xj: Matrix = None) -> Matrix:
        if Xj is None:
            Xj = Xi.copy()
        return (self.gamma * np.dot(Xi, Xj.T) + self.coef) ** self.deg

    def rbf_kernel(self, Xi: Matrix, Xj: Matrix = None) -> Matrix:
        if Xj is None:
            Xj = Xi.copy()
        _left = np.sum(Xi**2, axis=1).reshape(-1, 1)
        _right = np.sum(Xj**2, axis=1) - 2 * np.dot(Xi, Xj.T)

        return np.exp(-self.gamma * (_left + _right))

    def sigmoid_kernel(self, Xi: Matrix, Xj: Matrix = None) -> Matrix:
        if Xj is None:
            Xj = Xi.copy()
        return np.tanh(self.gamma * np.dot(Xi, Xj.T) + self.coef)

    def laplacian_kernel(self, Xi: Matrix, Xj: Matrix = None) -> Matrix:
        if Xj is None:
            Xj = Xi.copy()
        manhattan_dists = np.sum(np.abs(Xi[:, np.newaxis] - Xj), axis=2)

        return np.exp(-self.gamma * manhattan_dists)

    @property
    def kernel_func(self) -> Callable[[Matrix, Matrix | None], Matrix]:
        if self.kernel in ("linear", "lin"):
            return self.linear_kernel
        elif self.kernel in ("poly", "polynomial"):
            return self.polynomial_kernel
        elif self.kernel in ("rbf", "gaussian", "Gaussian"):
            return self.rbf_kernel
        elif self.kernel in ("sigmoid", "tanh"):
            return self.sigmoid_kernel
        elif self.kernel in ("laplacian", "lap"):
            return self.laplacian_kernel
        else:
            raise UnsupportedParameterError(self.kernel)


class Clone:
    """
    A utility class for cloning LUMA models.

    This class creates a copy of a given LUMA model,
    which can be either an Estimator or a Transformer.
    The clone includes all parameters of the original model.
    Optionally, the trained state of the model can also be copied
    if applicable.

    Parameters
    ----------
    `model` : object, optional, default=None
        The model to be cloned
    `pass_fitted` : bool, default=True
        Whether to copy the fitted state of the original model

    Examples
    --------
    >>> original_model = AnyModel(...)
    >>> cloned_model = Clone(model=original_model, pass_fitted=True).get

    """

    def __init__(
        self,
        model: object | None = None,
        pass_fitted: bool = False,
    ) -> None:
        self.model = model
        self.pass_fitted = pass_fitted

    @property
    def get(self) -> object:
        model_cls = type(self.model)
        new_model = model_cls()

        for param, val in self.model.__dict__.items():
            try:
                new_model.set_params(**{param: val})
            except:
                continue

        if hasattr(self.model, "_fitted") and self.pass_fitted:
            new_model._fitted = self.model._fitted

        return new_model


class ParamRange:
    """
    A utility class for setting and checking the range of a specific parameter.

    This class provides a user-friendly functionality which checks whether
    a certain parameter value falls within its preferred numerical range
    depending on its algorithm.

    Parameters
    ----------
    `param_range` : RangeStr
        An interval of a parameter (customizable)
    `param_type` : Type[Scalar], optional, default=None
        Data type of a parameter to be forced to have
        (None for both `int` and `float` types)

    Method
    ------
    To check its validity:
    ```py
    def check(self, param_value: Any) -> None
    ```
    Raises
    ------
    - `InvalidRangeError` :
        When the parameter does not fall in its preferred range
    - `UnsupportedParameterError` :
        When the parameter is not numeric or has wrong numeric type

    Examples
    --------
    ```py
    class SomeModel:
        def __init__(self, a: int) -> None:
            self.a = a
            a_range = ParamRange(param_range="0,+inf", param_type=int)
            a_range.check(param_value=self.a)
    ```
    Notes
    -----
    - When setting a custom range for `param_range`, it must follow the form of
        "lower_bound,upper_bound". (i.e. `-inf,20`, `-5,5`)
    - For open intervals, add '<' inside the range. (i.e. `0<,+inf`, `0<,<10`)
    """

    type RangeStr = str

    def __init__(
        self,
        param_range: RangeStr,
        param_type: Type[Scalar] = None,
    ) -> None:
        self.param_range = param_range

        if param_type is None:
            self.param_type = int | float
        else:
            if not ParamRange.validate_type(param_type):
                raise UnsupportedParameterError(param_type)
            self.param_type = param_type

    @classmethod
    def validate_type(cls, param_type) -> TypeGuard[Scalar]:
        return param_type in (int, float)

    def check(self, param_name: str, param_value: Any) -> None:
        if param_value is None:
            return
        if isinstance(param_value, (tuple, list)):
            for element in param_value:
                self._type_check(element)
        else:
            self._type_check(param_value)

        if not self.condition(param_value):
            raise InvalidRangeError(param_value, param_name, self.param_range)

    def _type_check(self, param_value: Any) -> None:
        if not isinstance(param_value, self.param_type):
            raise UnsupportedParameterError(param_value)

    @property
    def condition(self) -> Callable[[Scalar], bool]:
        try:
            left, right = self.param_range.split(",")
        except:
            raise UnsupportedParameterError(self.param_range)

        lower_open, upper_open = False, False
        if left[-1] == "<":
            lower_open = True
            lower = np.float64(left[:-1])
        else:
            lower = np.float64(left)

        if right[0] == "<":
            upper_open = True
            upper = np.float64(right[1:])
        else:
            upper = np.float64(right)

        if lower_open and not upper_open:
            return lambda x: lower < x <= upper
        elif lower_open and upper_open:
            return lambda x: lower < x < upper
        elif not lower_open and upper_open:
            return lambda x: lower <= x < upper
        elif not lower_open and not upper_open:
            return lambda x: lower <= x <= upper
        else:
            NotImplemented


class InitUtil:
    """
    An utility class for weight initializers used in neural networks.

    Parameters
    ----------
    `initializer` : InitStr, optional, default=True
        Name of an initializer

    Properties
    ----------
    To get the corresponding initializer type:
    ```py
    @property
    def initializer_type(self) -> type | None
    # `None` for random init
    ```
    """

    InitStr = Literal["he", "kaiming", "xavier", "glorot"] | None

    def __init__(self, initializer: InitStr) -> None:
        self.initializer = initializer

    @property
    def initializer_type(self) -> type | None:
        if self.initializer is None:
            return None
        if self.initializer in ("he", "kaiming"):
            return init.KaimingInit
        elif self.initializer in ("xavier", "glorot"):
            return init.XavierInit


class TrainProgress:
    """
    An utility class for managing auto-updating progress bar during training.

    It facilitates the progress bar from the module `rich`, which provides
    well-structured progress bar with colored indications for better readability.

    Parameters
    ----------
    `n_iter` : int
        Number of iterations(or epochs)
    `bar_width` : int, default=50
        Width size of a progress bar

    Property
    --------
    To get an iterable object from `rich.progress.Progress`:
    ```py
    (property) progress: (self: Self@TrainProgress) -> Iterable
    ```

    Examples
    --------
    Create an instance for `TrainProgress`:
    >>> train_prog = TrainProgress(n_iter=100)

    Generate a task and update the progress bar:
    ```py
    with train_prog.progress as progress:
        train_prog.add_task(progress=progress, model=AnyModelInstance)

        for i in range(n_iter):
            train_prog.update(progress=progress, cur=i, losses=[...])
            # losses is a list of [train_loss, valid_loss]
    ```
    """

    def __init__(self, n_iter: int, bar_width: int = 50) -> None:
        self.n_iter = n_iter
        self.bar_width = bar_width
        self.task = None

    @property
    def progress(self) -> Iterable:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=self.bar_width),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold green]{task.fields[train_loss]:.4f}"),
            TextColumn("[bold red]{task.fields[valid_loss]:.4f}"),
            expand=True,
        )

    def _check_task_exist(self) -> None:
        if self.task is None:
            raise RuntimeError(f"'{type(self).__name__}' does not have any tasks!")

    def add_task(self, progress: Progress, model: object) -> None:
        self.task = progress.add_task(
            f"[purple]Start {type(model).__name__} training "
            + f"with {self.n_iter} epochs.",
            total=self.n_iter,
            train_loss=0.0,
            valid_loss=0.0,
        )

    def update(
        self,
        progress: Progress,
        cur: int,
        losses: list[float, float],
    ) -> None:
        self._check_task_exist()
        progress.update(
            self.task,
            advance=1,
            train_loss=losses[0],
            valid_loss=losses[1],
            description=f"Epoch: {cur}/{self.n_iter} - "
            + f"Train/Valid Loss: {losses[0]:.4f}/{losses[1]:.4f}",
        )
