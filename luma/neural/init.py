import numpy as np

from luma.interface.typing import Vector, Matrix, Tensor, TensorLike


__all__ = ("KaimingInit", "XavierInit")


class Initializer:
    @classmethod
    def __class_alias__(cls) -> None: ...


class KaimingInit(Initializer):
    def __init__(self, random_state: int = None) -> None:
        super().__init__()
        self.rs_ = np.random.RandomState(random_state)

    def init_nd(
        self,
        input_size: int,
        output_size: int,
        *shape: int,
    ) -> TensorLike:
        stddev = np.sqrt(2.0 / (input_size * np.prod(Vector(shape))))
        return self.rs_.normal(
            0.0,
            stddev,
            (input_size, output_size, *shape),
        )


class XavierInit(Initializer):
    def __init__(self, random_state: int = None) -> None:
        super().__init__()
        self.rs_ = np.random.RandomState(random_state)

    def init_nd(
        self,
        input_size: int,
        output_size: int,
        *shape: int,
    ) -> TensorLike:
        stddev = np.sqrt(
            2.0
            / (
                input_size * np.prod(Vector(shape))
                + output_size * np.prod(Vector(shape))
            )
        )
        return self.rs_.normal(
            0.0,
            stddev,
            (input_size, output_size, *shape),
        )
