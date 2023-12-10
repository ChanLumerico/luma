from typing import *
import sys


__all__ = ['LUMA']


class LUMA:

    def __init__(self) -> None: ...
    
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...
    
    def __init_subclass__(cls) -> None: ...

    if sys.version_info < (3, 10):
        print("LUMA requires Python 3.10 or more", file=sys.stderr)
