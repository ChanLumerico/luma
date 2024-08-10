from typing import Literal

import numpy as np
import math

from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Scheduler


__all__ = (
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "CycleLR",
    "ReduceLROnPlateau",
)


class StepLR(Scheduler):
    def __init__(
        self,
        init_lr: float,
        step_size: int = 1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(init_lr)
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


class CyclicLR(Scheduler):
    def __init__(
        self,
        init_lr: float,
        max_lr: float,
        step_size_up: int,
        step_size_down: int | None = None,
        mode: Literal["tri", "tri2", "exp"] = "tri",
        gamma: float = 1.0,
    ) -> None:
        super().__init__(init_lr)
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = (
            step_size_down if step_size_down is not None else step_size_up
        )
        self.mode = mode
        self.gamma = gamma

        self.type_ = "batch"
        self.cycle_len = self.step_size_up + self.step_size_down
        self.cycle_count = 0

    @property
    def new_learning_rate(self) -> float:
        cycle_prog = self.iter % self.cycle_len
        if cycle_prog < self.step_size_up:
            factor = cycle_prog / self.step_size_up
        else:
            factor = 1 - (cycle_prog - self.step_size_up) / self.step_size_down

        if self.mode == "tri":
            new_lr = self.init_lr + (self.max_lr - self.init_lr) * factor
        elif self.mode == "tri2":
            new_lr = self.init_lr + (self.max_lr - self.init_lr) * factor * (
                0.5**self.cycle_count
            )
        elif self.mode == "exp":
            new_lr = (
                self.init_lr
                + (self.max_lr - self.init_lr) * (self.gamma**self.iter) * factor
            )
        else:
            raise UnsupportedParameterError(self.mode)

        self.lr_trace.append(new_lr)
        return new_lr
