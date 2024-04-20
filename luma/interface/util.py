from typing import Any, AnyStr, Callable, Literal, Self, Type, TypeGuard
import numpy as np

from luma.interface.exception import UnsupportedParameterError, InvalidRangeError
from luma.neural import activation, init


__all__ = (
    "Matrix",
    "Vector",
    "Tensor",
    "Scalar",
    "DecisionTreeNode",
    "NearestNeighbors",
    "SilhouetteUtil",
    "DBUtil",
    "KernelUtil",
    "ActivationUtil",
    "InitUtil",
    "Clone",
    "ParamRange",
)


type TensorLike = Matrix | Tensor | Vector


class Matrix(np.ndarray):
    """
    Internal class for matrices(2D-array) that extends `numpy.ndarray`.

    This class provides a way to create matrix objects that have
    all the capabilities of numpy arrays with the potential for
    additional functionalities and readability.

    Example
    -------
    >>> m = Matrix([1, 2, 3])
    >>> isinstance(m, numpy.ndarray) # True

    """

    def __new__(cls, array_like: Any) -> Self:
        if isinstance(array_like, (list, np.matrix)):
            obj = np.array(array_like)
        else:
            obj = array_like
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if obj is None:
            return


class Vector(Matrix):
    """
    Internal class for vectors(1D-array) that extends `Matrix`.

    This class represents a single row/column vector with its
    type of `numpy.ndarray` or `Matrix`.

    """

    def __new__(cls, array_like: Any) -> Self:
        if isinstance(array_like, list):
            obj = Matrix(array_like)
        else:
            obj = array_like
        return obj


class Tensor(Matrix):
    """
    Internal class for tensors(>=3D-arrray) that extends `Matrix`.

    This class provides a way to create tensor objects that have
    all the capabilities of numpy arrays with the potential for
    additional functionalities and readability.
    """

    type Tensor_3D = "Tensor"
    type Tensor_4D = "Tensor"

    def __new__(cls, array_like: Any) -> Self:
        if isinstance(array_like, list):
            obj = Matrix(array_like)
        else:
            obj = array_like
        return obj


class Scalar:
    """
    A placeholder class for scalar type.

    This class encompasses `int` and `float`.
    """

    def __new__(cls, value: int | float) -> Self:
        return float(value)


class DecisionTreeNode:
    """
    Internal class for node used in tree-based models.

    Parameters
    ----------
    `feature_index` : Feature of node
    `threshold` : Threshold for split point
    `left` : Left-child node
    `right` : Right-child node
    `value` : Most popular label of leaf node

    """

    def __init__(
        self,
        feature_index: int = None,
        threshold: float = None,
        left: "DecisionTreeNode" = None,
        right: "DecisionTreeNode" = None,
        value: Any = None,
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
    `data` : Data to be handled
    `n_neighbors` : Number of nearest neighbors

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
    `idx` : Index of a single data point
    `cluster` : Current cluster number
    `labels` : Labels assigned by clustering estimator
    `distances` : Square-form distance matrix of the data

    """

    def __init__(
        self, idx: int, cluster: int, labels: Matrix, distances: Matrix
    ) -> None:
        self.idx = idx
        self.cluster = cluster
        self.labels = labels
        self.distances = distances

    @property
    def avg_dist_others(self) -> Matrix:
        others = set(self.labels) - {self.cluster}
        sub_avg = [
            np.mean(self.distances[self.idx][self.labels == other]) for other in others
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
    `data` : Original data
    `labels` : Labels assigned by clustering estimator

    """

    def __init__(self, data: Matrix, labels: Matrix) -> None:
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
        kernel: str,
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


class ActivationUtil:
    """
    Internal class for activaiton functions used in neural networks.

    Properties
    ----------
    For getting an activation function class:
    ```py
    @property
    def activation_type(self) -> type
    ```

    Examples
    --------
    >>> act = ActivationUtil(activation='relu')
    >>> relu = act.activation_type
    ReLU()

    """

    FuncType = Literal[
        "relu",
        "ReLU",
        "leaky-relu",
        "leaky-ReLU",
        "elu",
        "ELU",
        "tanh",
        "sigmoid",
        "sig",
        "softmax",
    ]

    def __init__(self, activation: str) -> None:
        self.activation = activation

    @property
    def activation_type(self) -> type:
        if self.activation in ("relu", "ReLU"):
            return activation.ReLU
        elif self.activation in ("leaky-relu", "leaky-ReLU"):
            return activation.LeakyReLU
        elif self.activation in ("elu", "ELU"):
            return activation.ELU
        elif self.activation in ("tanh"):
            return activation.Tanh
        elif self.activation in ("sigmoid", "sig"):
            return activation.Sigmoid
        elif self.activation in ("softmax"):
            return activation.Softmax


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
    `model` : The model to be cloned
    `pass_fitted` : Whether to copy the fitted state of the original model

    Examples
    --------
    >>> original_model = AnyModel(...)
    >>> cloned_model = Clone(model=original_model, pass_fitted=True).get

    """

    def __init__(self, model: object = None, pass_fitted: bool = False) -> None:
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
    `param_range` : An interval of a parameter (customizable)
    `param_type` : Data type of a parameter to be forced to have
    (`None` for both `int` and `float` types)

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

    type RangeStr = AnyStr

    def __init__(self, param_range: RangeStr, param_type: Type[Scalar] = None) -> None:
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
    TODO: Write docstring
    """

    InitType = Literal["he", "kaiming", "xavier", "auto", "random"]

    def __init__(
        self,
        initializer: InitType,
        activation: ActivationUtil.FuncType,
    ) -> None:
        self.initializer = initializer
        self.activation = activation

    @property
    def initializer_type(self) -> type | None:
        if self.initializer == "auto":
            return self._auto_init()
        else:
            return self._manual_init()

    def _manual_init(self) -> type | None:
        if self.initializer in ("he", "kaiming"):
            return init.KaimingInit
        elif self.initializer in ("xavier"):
            return init.XavierInit

    def _auto_init(self) -> type | None:
        if self.activation in ("relu", "ReLU"):
            return init.KaimingInit
        elif self.activation in ("tanh", "sigmoid", "sig"):
            return init.XavierInit
