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
    "OneCycleLR",
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


class OneCycleLR(Scheduler):
    def __init__(
        self,
        init_lr: float,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: Literal["cos"] | str = "cos",
        final_div_factor: float = 1e4,
    ) -> None:
        super().__init__(init_lr)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.final_div_factor = final_div_factor

        self.type_ = "batch"
        self.up_step = int(self.total_steps * self.pct_start)
        self.down_step = self.total_steps - self.up_step

    @property
    def new_learning_rate(self) -> float:
        if self.iter <= self.up_step:
            factor = self.iter / self.up_step
            new_lr = self.init_lr + (self.max_lr - self.init_lr) * factor
        else:
            factor = (self.iter - self.up_step) / self.down_step
            if self.anneal_strategy == "cos":
                new_lr = self.max_lr * (1 + math.cos(math.pi * factor)) / 2
            else:
                new_lr = (
                    self.max_lr
                    - (self.max_lr - self.init_lr / self.final_div_factor) * factor
                )

        self.lr_trace.append(new_lr)
        return new_lr


class ReduceLROnPlateau(Scheduler):
    def __init__(
        self,
        init_lr: float,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
    ) -> None:
        super().__init__(init_lr)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr

        self.type_ = "epoch"
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_cnt = 0
        self.mode_worse = np.inf if self.mode == "min" else -np.inf
        self.best = self.mode_worse

    @property
    def new_learning_rate(self) -> float:
        current = self.valid_loss_arr[-1] if self.valid_loss_arr else None

        if current is None:
            new_lr = self.lr_trace[-1]
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.cooldown_cnt > 0:
                self.cooldown_cnt -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs > self.patience:
                new_lr = max(self.lr_trace[-1] * self.factor, self.min_lr)
                self.cooldown_cnt = self.cooldown
                self.num_bad_epochs = 0
            else:
                new_lr = self.lr_trace[-1]

        self.lr_trace.append(new_lr)
        return new_lr

    def is_better(self, current: float, best: float) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return current < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return current < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return current > best * rel_epsilon

        else:
            return current > best + self.threshold
