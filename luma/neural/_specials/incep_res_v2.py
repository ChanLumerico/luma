from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph


class _IncepRes_V2_TypeA(LayerGraph): ...


class _IncepRes_V2_TypeB(LayerGraph): ...


class _IncepRes_V2_TypeC(LayerGraph): ...


class _IncepRes_V2_Redux(LayerGraph): ...
