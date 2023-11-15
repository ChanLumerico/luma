from typing import Any
from LUMA.Interface.Type import Transformer, Estimator


class NotFittedError(Exception):
    def __init__(self, model: Transformer | Estimator | Any) -> None:
        super().__init__(f'{model.__class__.__name__} is not fitted!' + 
                         f' Call {model.__class__.__name__}.fit() to fit the model.')
        

class UnsupportedParameterError(Exception):
    def __init__(self, param: Any) -> None:
        super().__init__(f'{param} is unsupported!')

