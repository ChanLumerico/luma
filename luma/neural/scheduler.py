import numpy as np

from luma.neural.base import Scheduler


__all__ = ("StepLR",)


class StepLR(Scheduler):
    """
    This scheduler is based on epochs and reduces the 
    learning rate by a factor every specified number of epochs.

    TODO: Refine this class.
    """

    def __init__(
        self,
        initial_lr: float,
        step_size: int,
        gamma: float,
    ) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

        self.type_ = "epoch"

    @property
    def new_learning_rate(self) -> float:
        epoch_index = self.iter // self.n_iter
        factor = epoch_index // self.step_size

        return self.initial_lr * (self.gamma**factor)
