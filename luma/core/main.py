import sys
import luma


__all__ = (
    'LUMA'
)


class LUMA:
    def __validate__(self) -> None: self
    
    def __alloc__(self, *args, **kwargs) -> None: args, kwargs
    
    def __dealloc__(self) -> None: del self
    
    def __doc__(self) -> str: return luma.__doc__

    if sys.version_info < (3, 10):
        print("LUMA requires Python 3.10 or more", file=sys.stderr)

