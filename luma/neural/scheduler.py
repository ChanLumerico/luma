import numpy as np
import math

from luma.neural.base import Scheduler


__all__ = (
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
)

# TODO: Please add docstrings in the future development.


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
        new_lr = self.init_lr * (self.gamma**epoch_index)

        self.lr_trace.append(new_lr)
        return new_lr


class CosineAnnealingLR(Scheduler):
    def __init__(
        self,
        init_lr: float,
        T_max: int,
        eta_min: float = 0,
    ) -> None:
        super().__init__(init_lr)
        self.init_lr = init_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.type_ = "epoch"

    @property
    def new_learning_rate(self) -> float:
        epoch_index = self.iter // self.n_iter
        new_lr = (
            self.eta_min
            + (self.init_lr - self.eta_min)
            * (1 + math.cos(math.pi * epoch_index / self.T_max))
            / 2
        )
        self.lr_trace.append(new_lr)
        return new_lr
