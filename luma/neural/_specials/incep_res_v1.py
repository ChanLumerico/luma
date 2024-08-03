from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph


class _IncepRes_V1_Stem(Sequential):
    def __init__(
        self,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }
        _ = do_batch_norm

        self.set_param_ranges(
            {
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(_IncepRes_V1_Stem, self).__init__(
            Convolution2D(3, 32, 3, 2, "valid", **basic_args),
            BatchNorm2D(32, momentum),
            activation(),
            Convolution2D(32, 32, 3, 1, "valid", **basic_args),
            BatchNorm2D(32, momentum),
            activation(),
            Convolution2D(32, 64, 3, 1, "same", **basic_args),
            BatchNorm2D(64, momentum),
            activation(),
            Pooling2D(3, 2, "max", "valid"),
        )
        self.model.extend(
            Convolution2D(64, 80, 1, 1, "same", **basic_args),
            BatchNorm2D(80, momentum),
            activation(),
            Convolution2D(80, 192, 3, 1, "valid", **basic_args),
            BatchNorm2D(192, momentum),
            activation(),
            Convolution2D(192, 256, 3, 2, "valid", **basic_args),
            BatchNorm2D(256, momentum),
            activation(),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)
