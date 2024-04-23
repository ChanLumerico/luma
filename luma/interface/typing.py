from typing import Any, Self
import numpy as np


__all__ = ("TensorLike", "Matrix", "Vector", "Tensor", "Scalar")


class TensorLike(np.ndarray):
    """
    Internal base class for n-dimensional arrays that extends
    `numpy.ndarray`.

    Apart from the type `Tensor`, this class encompasses all the
    mathematical notion of generalized tensors, including
    vectors(1D-array), matrices(2D-array).
    """

    def __init__(self) -> None:
        super().__init__()

    def __array_finalize__(self, obj: None | np.ndarray[Any, np.dtype[Any]]) -> None:
        return super().__array_finalize__(obj)

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


class Matrix(TensorLike):
    """
    Internal class for matrices(2D-array) that extends `TensorLike`.

    This class provides a way to create matrix objects that have
    all the capabilities of numpy arrays with the potential for
    additional functionalities and readability.

    Example
    -------
    >>> m = Matrix([1, 2, 3])

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


class Vector(TensorLike):
    """
    Internal class for vectors(1D-array) that extends `TensorLike`.

    This class represents a single row/column vector with its
    type of `TensorLike`.

    """

    def __new__(cls, array_like: Any) -> Self:
        if isinstance(array_like, list):
            obj = Matrix(array_like)
        else:
            obj = array_like
        return obj


class Tensor(TensorLike):
    """
    Internal class for tensors(>=3D-arrray) that extends `TensorLike`.

    This class provides a way to create tensor objects that have
    all the capabilities of numpy arrays with the potential for
    additional functionalities and readability.

    Notes
    -----
    In general mathematics and physics, a term 'Tensor' refers to a
    generalized n-dimensional arrays, whereas in `luma`, `Tensor` only
    refers to an array with its dimensionality higher then 2.
    """

    type Tensor_3D = Self
    type Tensor_4D = Self

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
