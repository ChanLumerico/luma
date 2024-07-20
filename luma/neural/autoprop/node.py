from typing import List, Literal
import numpy as np

from luma.interface.typing import TensorLike
from luma.neural.layer import LayerLike


class LayerNode:
    def __init__(
        self, 
        layer: LayerLike, 
        prev_nodes: List[LayerLike] = [],
        next_nodes: List[LayerLike] = [],
        merge_mode: Literal["chcat", "sum"] = "chcat",
    ) -> None:
        self.layer = layer
        self.prev_nodes = prev_nodes
        self.next_nodes = next_nodes
        self.merge_mode = merge_mode

        self.f_queue = []
        self.b_queue = []
        self.n_forward, self.n_backward = 0, 0

        self.cum_ch = [0]
        self.visited: bool = False
    
    def for_enqueue(self, X: TensorLike) -> None:
        self.n_forward += 1
        self.f_queue.append(X)
        self.cum_ch.append(self.cum_ch[-1] + X.shape[1])
    
    def back_enqueue(self, d_out: TensorLike) -> None:
        self.n_backward += 1
        self.b_queue.append(d_out)
    
    def forward(self, is_train: bool = False) -> TensorLike:
        match self.merge_mode:
            case "chcat":
                X = np.concatenate(self.f_queue, axis=1)
            case "sum":
                X = np.sum(self.f_queue, axis=0)
        out = self.layer(X, is_train)
        return out
    
    def backward(self) -> List[TensorLike]:
        ...
