import numpy as np

from luma.neural.base import Scheduler


__all__ = ("StepLR", "ExponentialLR")


class StepLR(Scheduler):
    def __init__(
        self,
        init_lr: float,
        step_size: int = 1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(init_lr)
        self.init_lr = init_lr
        self.step_size = step_size
        self.gamma = gamma

        self.type_ = "epoch"

    @property
    def new_learning_rate(self) -> float:
        epoch_index = self.iter // self.n_iter
        factor = epoch_index // self.step_size

        new_lr = self.init_lr * (self.gamma**factor)
        self.lr_trace.append(new_lr)
        return new_lr


class ExponentialLR(Scheduler):
    def __init__(
        self,
        init_lr: float,
        gamma: float = 0.9,
    ) -> None:
        super().__init__(init_lr)
        self.init_lr = init_lr
        self.gamma = gamma

        self.type_ = "epoch"

    @property
    def new_learning_rate(self) -> float:
        epoch_index = self.iter // self.n_iter
        new_lr = self.init_lr * (self.gamma ** epoch_index)
        
        self.lr_trace.append(new_lr)
        return new_lr
    