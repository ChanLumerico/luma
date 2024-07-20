from typing import List, Literal, Dict
from collections import deque
import numpy as np

from luma.interface.typing import TensorLike
from luma.neural.layer import LayerLike


__all__ = ("LayerNode", "LayerGraph")


class LayerNode:
    def __init__(
        self,
        layer: LayerLike,
        prev_nodes: List[LayerLike] = [],
        next_nodes: List[LayerLike] = [],
        merge_mode: Literal["chcat", "sum"] = "chcat",
    ) -> None:
        self.layer: LayerLike = layer
        self.prev_nodes: List[LayerNode] = prev_nodes
        self.next_nodes: List[LayerNode] = next_nodes
        self.merge_mode = merge_mode

        self.f_queue: List[TensorLike] = []
        self.b_queue: List[TensorLike] = []
        self.n_forward, self.n_backward = 0, 0

        self.cum_ch = [0]
        self.f_visited: bool = False
        self.b_visited: bool = False

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
        d_cum = np.sum(self.b_queue, axis=0)
        d_out = self.layer.backward(d_cum)
        if not self.n_backward:
            return [d_out]

        d_out_arr = []
        for i in range(self.n_backward):
            if self.merge_mode == "chcat":
                d_out_arr.append(
                    d_out[
                        :,
                        self.cum_ch[i] : self.cum_ch[i + 1],
                        ...,
                    ]
                )
            elif self.merge_mode == "sum":
                d_out_arr.append(d_out)

        return d_out_arr
    
    def flush(self) -> None:
        self.n_forward, self.n_backward = 0, 0
        self.f_queue.clear()
        self.b_queue.clear()

        self.f_visited = False
        self.b_visited = False
    
    def __call__(self, is_train: bool = False) -> TensorLike:
        return self.forward(is_train)


class LayerGraph:
    def __init__(
        self,
        graph: Dict[LayerNode, List[LayerNode]],
        root: LayerNode,
        term: LayerNode,
    ) -> None:
        self.graph = graph
        self.root = root
        self.term = term

        self.nodes: List[LayerNode] = []
        self.built: bool = False
    
    def build(self) -> None:
        for kn, vn in self.graph.items():
            if not vn:
                continue
            for v in vn:
                kn.next_nodes.append(v)
                v.prev_nodes.append(kn)
            
            if kn not in self.nodes:
                self.nodes.append(kn)
        
        for node in self.nodes:
            if not node.prev_nodes and not node.next_nodes:
                raise RuntimeError(f"'{self}' is not fully connected!")
        self.built = True
        return
    
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return self._forward_bfs(X, is_train)
    
    def backward(self, d_out: TensorLike) -> TensorLike:
        return self._backward_bfs(d_out)
    
    def _forward_bfs(self, X: TensorLike, is_train: bool) -> TensorLike:
        queue = deque([self.root])
        self.root.for_enqueue(X)
        self.root.f_visited = True

        while queue:
            cur = queue.pop()
            X = cur(is_train)
            for next in cur.next_nodes:
                if next.f_visited:
                    continue
                next.for_enqueue(X)
                next.f_visited = True
                queue.append(next)

        return X
    
    def _backward_bfs(self, d_out: TensorLike) -> TensorLike:
        queue = deque([self.term])
        self.term.back_enqueue(d_out)
        self.term.b_visited = True

        while queue:
            cur = queue.pop()
            d_out_arr = cur.backward()
            for prev, dx in zip(cur.prev_nodes, d_out_arr):
                if prev.b_visited:
                    continue
                prev.back_enqueue(dx)
                prev.b_visited = True
                queue.append(prev)
            
            cur.flush()

        d_out = d_out_arr.pop()
        if d_out_arr:
            raise RuntimeError(f"'{self}' has more than one root nodes!")
        
        return d_out
