import sys
import luma


__all__ = "Luma"


class Luma:
    """
    A root class for `luma` module.
    """

    def __validate__(self) -> None:
        self

    def __alloc__(self, *args, **kwargs) -> None:
        args, kwargs

    def __dealloc__(self) -> None:
        del self

    def __doc__(self) -> str:
        return luma.__doc__

    if sys.version_info < (3, 12):
        print("Luma requires Python 3.12 or more", file=sys.stderr)
