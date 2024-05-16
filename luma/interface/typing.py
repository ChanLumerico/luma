from functools import wraps
from typing import Any, Callable, Generic, NoReturn, Self, Tuple, Type, TypeVar
import sys
import numpy as np


__all__ = (
    "TensorLike",
    "Matrix",
    "Vector",
    "Tensor",
    "Scalar",
    "ClassType",
)


T = TypeVar("T", bound=type)
D = TypeVar("D", bound=int)


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


class Tensor(TensorLike, Generic[D]):
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

    def __new__(cls, array_like: Any) -> Self:
        if isinstance(array_like, list):
            obj = Matrix(array_like)
        else:
            obj = array_like
        return obj

    @classmethod
    def force_dim(cls, *dim_consts: int) -> Callable:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(self, *args: Any, **kwargs: Any) -> Any:
                arg_names = func.__code__.co_varnames
                all_args = {**dict(zip(arg_names, (self,) + args)), **kwargs}

                for i, n_dim in enumerate(dim_consts):
                    param_name = arg_names[i + 1]

                    if param_name in all_args:
                        tensor = all_args[param_name]
                        if not isinstance(tensor, (Tensor, np.ndarray)):
                            raise TypeError(f"'{param_name}' must be of type Tensor.")
                        if tensor.ndim != n_dim:
                            raise ValueError(
                                f"'{param_name}' must be {n_dim}D-tensor",
                                +f" got {tensor.ndim}D-tensor.",
                            )

                return func(self, *args, **kwargs)

            return wrapper

        return decorator


class Scalar:
    """
    A placeholder class for scalar type.

    This class encompasses `int` and `float`.
    """

    def __new__(cls, value: int | float) -> Self:
        return value


class ClassType:
    """
    An interface class designed to facilitate the annotation of
    dedicated types for classes in a systematic way.

    The `ClassType` class includes several classmethods that act
    as decorator factories. These decorators can be applied to classes
    to specify their types or roles within the application more
    explicitly.

    Decorator Factories
    -------------------
    `non_instantiable` : callable
        Makes a class non-instantiable.
    `private` : callable
        Makes a class private; accesses are restricted outside its module.

    """

    @classmethod
    def non_instantiable(cls) -> Callable:
        def decorator(cls: Type[T]) -> Type[T]:
            def wrapper(*args, **kwargs) -> NoReturn:
                args, kwargs
                raise TypeError(
                    f"'{cls.__name__}'" + " is not instantiable!",
                )

            cls.__new__ = wrapper
            return cls

        return decorator

    @classmethod
    def private(cls) -> Callable:
        def decorator(cls: Type[T]) -> Type[T]:

            @wraps(cls, updated=())
            class PrivateClassWrapper(cls):
                def __new__(cls, *args, **kwargs) -> Any:
                    args, kwargs
                    caller = sys._getframe(1)

                    if caller.f_globals["__name__"] == cls.__module__:
                        return super().__new__(cls)
                    else:
                        raise TypeError(
                            f"'{cls.__name__}' is a private class and cannot "
                            + "be instantiated outside its module.",
                        )

                def __init__(self, *args, **kwargs) -> None:
                    super(PrivateClassWrapper, self).__init__(*args, **kwargs)

            for attr in dir(cls):
                if not attr.startswith("_"):
                    setattr(PrivateClassWrapper, attr, getattr(cls, attr))

            return PrivateClassWrapper

        return decorator
