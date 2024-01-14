from typing import Any
import sys


__all__ = (
    'LUMA'
)


class LUMA:
    
    """
    LUMA
    ====
    A Comprehensive Python Module for Machine Learning and Data Science
    """

    def __init__(self) -> None: ...
    
    def __call__(self, *args, **kwargs) -> Any: ...
    
    def __init_subclass__(cls) -> None: ...

    if sys.version_info < (3, 10):
        print("LUMA requires Python 3.10 or more", file=sys.stderr)

