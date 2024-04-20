import numpy as np


__all__ = ("KaimingInit", "XavierInit")


type TensorLike = TensorLike


class Initializer:
    @classmethod
    def __class_alias__(cls) -> None: ...


class KaimingInit(Initializer):
    def __init__(self, random_state: int = None) -> None:
        super().__init__()
        self.rs_ = np.random.RandomState(random_state)

    def init_2d(self, input_size: int, output_size: int) -> TensorLike:
        stddev = np.sqrt(2.0 / input_size)
        return self.rs_.normal(
            0.0,
            stddev,
            (input_size, output_size),
        )

    def init_4d(
        self, input_size: int, output_size: int, height: int, width: int
    ) -> TensorLike:
        stddev = np.sqrt(2.0 / (input_size * height * width))
        return self.rs_.normal(
            0.0,
            stddev,
            (input_size, output_size, height, width),
        )


class XavierInit(Initializer): ...


# TODO: Further implementation
