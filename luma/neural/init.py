import numpy as np


__all__ = ("KaimingInit", "XavierInit")


type Matrix = Matrix
type Tensor = Tensor


class KaimingInit:
    def __init__(self, random_state: int = None) -> None:
        super().__init__()
        self.rs_ = np.random.RandomState(random_state)

    def init_2d(self, input_size: int, output_size: int) -> Matrix:
        stddev = np.sqrt(2.0 / input_size)
        return self.rs_.normal(
            0.0,
            stddev,
            (input_size, output_size),
        )

    def init_4d(
        self, input_size: int, output_size: int, height: int, width: int
    ) -> Tensor:
        stddev = np.sqrt(2.0 / (input_size * height * width))
        return self.rs_.normal(
            0.0,
            stddev,
            (input_size, output_size, height, width),
        )


class XavierInit: ...


# TODO: Further implementation
