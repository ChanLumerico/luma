from functools import wraps
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Generic,
    NoReturn,
    Self,
    Type,
    TypeVar,
)
import sys
import numpy as np


__all__ = (
    "TensorLike",
    "Matrix",
    "Vector",
    "Tensor",
    "Scalar",
    "ClassType",
    "LayerLike",
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
        """
        Decorator factory to enforce the dimensionality of tensor arguments.

        This decorator ensures that specified tensor arguments to the decorated
        function have the expected number of dimensions.

        Parameters
        ----------
        `*dim_constraints` : int
            Variable length argument list where each value specifies the required
            number of dimensions for the corresponding tensor argument in the
            decorated function.

        Returns
        -------
        `Callable`
            The decorated function with dimensionality constraints enforced on
            specified tensor arguments.

        Raises
        ------
        `TypeError`
            If the argument is not a numpy ndarray.
        `ValueError`
            If the argument does not have the specified number of dimensions.

        Examples
        --------
        >>> class MyModel:
        ...     @Tensor.force_dim(4)
        ...     def foo(self, tensor: Tensor) -> Any:
        ...         print("Processing foo")
        ...
        >>> model = MyModel()
        >>> tensor_4d = np.random.rand(2, 3, 4, 5).view(Tensor)
        >>> model.forward(tensor_4d)
        Processing foo

        >>> tensor_3d = np.random.rand(2, 3, 4).view(Tensor)
        >>> model.foo(tensor_3d)
        Traceback (most recent call last): ...
        ValueError: 'X' must be 4-dimensional, got 3 dimensions

        """

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
                            raise TypeError(
                                f"'{param_name}' must be an insatnce of Tensor.",
                            )
                        if tensor.ndim != n_dim:
                            raise ValueError(
                                f"'{param_name}' must be {n_dim}D-tensor,"
                                + f" got {tensor.ndim}D-tensor.",
                            )

                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def force_shape(cls, *shape_consts: tuple[int]) -> Callable:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(self, *args: Any, **kwargs: Any) -> Any:
                arg_names = func.__code__.co_varnames
                all_args = {**dict(zip(arg_names, (self,) + args)), **kwargs}

                mismatch_dict = defaultdict(lambda: np.empty((0, 3)))
                for i, shape in enumerate(shape_consts):
                    param_name = arg_names[i + 1]

                    if param_name in all_args:
                        tensor = all_args[param_name]
                        if not isinstance(tensor, (Tensor, np.ndarray)):
                            raise TypeError(
                                f"'{param_name}' must be an instance of Tensor.",
                            )

                        if tensor.ndim != len(shape):
                            raise ValueError(
                                f"Dimensionalities of '{param_name}' and"
                                + f" the constraint '{shape}' does not match!"
                            )

                        for axis, (s, ts) in enumerate(zip(shape, tensor.shape)):
                            if s == -1:
                                continue
                            if s != ts:
                                mismatch_dict[param_name] = np.vstack(
                                    (mismatch_dict[param_name], [axis, s, ts])
                                )

                def _tuplize(vec: Vector):
                    return tuple(int(v) for v in vec)

                if len(mismatch_dict):
                    title = (
                        f"{"Argument":^14} {"Axes":^14} {"Expected":^14} {"Shape":^14}"
                    )
                    msg = str()

                    for name in mismatch_dict.keys():
                        errmat = mismatch_dict[name]

                        axes = str(_tuplize(errmat[:, 0]))
                        expect = str(_tuplize(errmat[:, 1]))
                        got = str(_tuplize(errmat[:, 2]))

                        msg += f"{name:^14} {axes:<14} {expect:<14} {got:<14}\n"

                    raise ValueError(
                        f"Shape mismatch(es) detected as follows:"
                        + f"\n{title}"
                        + f"\n{"-" * (14 * 4 + 3)}"
                        + f"\n{msg}",
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


class LayerLike:
    """
    Internal class for layer-like neural components which has
    both feed-forwaring and backpropagating mechanisms implemented.

    This class currently encompasses `Layer`, `Sequential`, and
    `LayerGraph` of Luma's AutoProp system.

    """

    def forward(self, X: TensorLike, *args) -> TensorLike: ...

    def backward(self, d_out: TensorLike, *args) -> TensorLike: ...

    def update(self, *args) -> TensorLike: ...

    @property
    def param_size(self) -> tuple[int, int]: ...

    def out_shape(self, in_shape: tuple[int]) -> tuple[int]: ...
