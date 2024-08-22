import warnings
import functools


__all__ = ("not_used",)


def not_used(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> None:
        warnings.warn(
            f"The function '{func.__name__}' is marked as 'not_used'"
            + f" and is not intended to be executed.",
            Warning,
            stacklevel=2,
        )
        return None

    return wrapper
