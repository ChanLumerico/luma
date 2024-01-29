from typing import Any
from luma.core.super import Transformer, Estimator


__all__ = (
    'NotFittedError', 
    'UnsupportedParameterError', 
    'NotConvergedError',
    'ModelExtensionError'
)


def get_name(model: Any) -> str: return type(model).__name__


class NotFittedError(Exception):
    def __init__(self, model: Transformer | Estimator | Any) -> None:
        super().__init__(f"'{get_name(model)}' is not fitted!" + 
                         f" Call '{get_name(model)}.fit()' to fit the model.")
        

class UnsupportedParameterError(Exception):
    def __init__(self, param: Any) -> None:
        super().__init__(f"'{param}' is unsupported!")


class NotConvergedError(Exception):
    def __init__(self, model: Transformer | Estimator | Any) -> None:
        super().__init__(f"'{get_name(model)}' did not converged!" + 
                         f" Try setting '{get_name(model)}.tol' to bigger value.")


class ModelExtensionError(Exception):
    def __init__(self, filename: str) -> None:
        super().__init__(f"'{filename}' is not a '.luma' model file!")

