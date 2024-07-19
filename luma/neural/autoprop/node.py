from typing import List

from luma.interface.typing import TensorLike
from luma.neural.layer import LayerLike


class LayerNode:
    def __init__(
        self, 
        layer: LayerLike, 
        prev_nodes: List[LayerLike],
        next_nodes: List[LayerLike],
    ) -> None:
        ...
        # TODO: Begin from here

