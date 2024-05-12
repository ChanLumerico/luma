from typing import Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.neural.base import Layer


__all__ = "_Dropout"


class _Dropout(Layer):
    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int = None,
    ) -> None:
        super(_Dropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        else:
            return X

    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape
