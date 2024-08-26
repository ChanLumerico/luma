from typing import List, Dict, Self, Any, Tuple
from collections import deque
import numpy as np

from luma.interface.typing import TensorLike, LayerLike
from luma.interface.util import Clone
from luma.interface.exception import NotFittedError
from luma.core.super import Optimizer

from .merge import MergeMode


__all__ = ("LayerNode", "LayerGraph")


class LayerNode:
    def __init__(
        self,
        layer: LayerLike,
        merge_mode: MergeMode = MergeMode.SUM,
        name: str | None = None,
    ) -> None:
        self.layer: LayerLike = layer
        self.prev_nodes: List[LayerNode] = []
        self.next_nodes: List[LayerNode] = []
        self.merge_mode = merge_mode
        self.name = name

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
        X = self.merge_mode.forward(self.f_queue)
        return self.layer(X, is_train)

    def backward(self) -> List[TensorLike]:
        d_cum = np.sum(self.b_queue, axis=0)
        d_out = self.layer.backward(d_cum)
        if not self.n_backward:
            return [d_out]

        d_out_arr = []
        for i in range(self.n_forward):
            d_out_arr.append(self.merge_mode.backward(self.f_queue, d_out, i))

        return d_out_arr

    def update(self) -> None:
        self.layer.update()

    def set_optimizer(self, optimizer: Optimizer, **params: Any) -> None:
        if hasattr(self.layer, "set_optimizer"):
            self.layer.set_optimizer(optimizer, **params)
        elif hasattr(self.layer, "optimizer"):
            optim: Optimizer = Clone(optimizer).get
            optim.set_params(**params)
            self.layer.optimizer = optim

    def update_lr(self, new_lr: float) -> None:
        self.layer.update_lr(new_lr)

    def flush(self) -> None:
        self.n_forward, self.n_backward = 0, 0
        self.f_queue.clear()
        self.b_queue.clear()

        self.f_visited = False
        self.b_visited = False

    @property
    def param_size(self) -> Tuple[int, int]:
        return self.layer.param_size

    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        return self.layer.out_shape(in_shape)

    def __call__(self, is_train: bool = False) -> TensorLike:
        return self.forward(is_train)

    def __str__(self) -> str:
        if self.name is None:
            return type(self).__name__
        return self.name

    def __repr__(self) -> str:
        return f"({str(self)}: {self.layer})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayerNode):
            return False
        return self.name == other.name and self.layer == other.layer

    def __hash__(self) -> int:
        return hash((self.name, self.layer))


class LayerGraph(LayerLike):
    def __init__(
        self,
        graph: Dict[LayerNode, List[LayerNode]] | None = None,
        root: LayerNode | None = None,
        term: LayerNode | None = None,
    ) -> None:
        self.graph = graph if graph is not None else dict()
        self.root = root
        self.term = term

        self.nodes: List[LayerNode] = []
        self.built_: bool = False

        if self.graph:
            self._init_predefined_graph()

    def _init_predefined_graph(self):
        all_nodes = set(self.graph.keys())
        for _, vn in self.graph.items():
            for v in vn:
                all_nodes.add(v)
        self.nodes = list(all_nodes)

    def add_node(
        self,
        node: LayerNode,
        prev_nodes: List[LayerNode] = [],
        next_nodes: List[LayerNode] = [],
    ) -> None:
        if node in self.graph:
            raise ValueError(f"Node {node} already exists in the graph.")

        self.graph[node] = next_nodes
        for prev in prev_nodes:
            if prev not in self.graph:
                self.graph[prev] = []
            self[prev].append(node)

        node.prev_nodes.extend(prev_nodes)
        node.next_nodes.extend(next_nodes)

        if node not in self.nodes:
            self.nodes.append(node)

        for prev in prev_nodes:
            if prev not in self.nodes:
                self.nodes.append(prev)

        for next in next_nodes:
            if next not in self.nodes:
                self.nodes.append(next)

        self.built_ = False

    def remove_node(self, node: LayerNode) -> None:
        if node in self.graph:
            del self.graph[node]

            for prev_node in node.prev_nodes:
                self[prev_node].remove(node)
                prev_node.next_nodes.remove(node)

            for next_node in node.next_nodes:
                next_node.prev_nodes.remove(node)

            node.prev_nodes.clear()
            node.next_nodes.clear()
            node.flush()

            self.nodes.remove(node)
            self.built_ = False
        else:
            raise ValueError(f"'{node}' does not exist in the graph.")

    def build(self) -> None:
        if self.built_:
            return
        all_nodes = set(self.graph.keys())
        for kn, vn in self.graph.items():
            kn.next_nodes = list(vn)

            for v in vn:
                if kn not in v.prev_nodes:
                    v.prev_nodes.append(kn)
                all_nodes.add(v)

        self.nodes = list(all_nodes)
        visited = set()

        def _dfs(node: LayerNode) -> None:
            if node in visited:
                return
            visited.add(node)
            for next_node in node.next_nodes:
                _dfs(next_node)

        _dfs(self.root)

        if visited != set(self.nodes):
            raise RuntimeError(f"'{self}' is not fully connected!")
        if self.detect_cycle():
            raise RuntimeError(f"'{self}' contains a cycle!")

        self.built_ = True

    def detect_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def _visit(node: LayerNode) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            for next_node in node.next_nodes:
                if _visit(next_node):
                    return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if _visit(node):
                return True

        return False

    def set_optimizer(self, optimizer: Optimizer, **params: Any) -> None:
        self.check_is_built()
        for node in self.nodes:
            node.set_optimizer(optimizer, **params)

    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        self.check_is_built()
        return self._forward_bfs(X, is_train)

    def backward(self, d_out: TensorLike) -> TensorLike:
        self.check_is_built()
        return self._backward_bfs(d_out)

    def update(self) -> None:
        self.check_is_built()
        for node in self.nodes:
            node.update()

    def update_lr(self, new_lr: float) -> None:
        for node in self.nodes:
            node.update_lr(new_lr)

    def check_is_built(self) -> None:
        if not self.built_:
            raise NotFittedError(
                f"'{self}' has not built! Call 'build()' to build the graph."
            )

    def _forward_bfs(self, X: TensorLike, is_train: bool) -> TensorLike:
        queue = deque([self.root])
        self.root.for_enqueue(X)
        self.root.f_visited = True

        while queue:
            cur = queue.popleft()
            X = cur(is_train)

            for next in cur.next_nodes:
                next.for_enqueue(X)
                if not next.f_visited:
                    next.f_visited = True
                    queue.append(next)

        return X

    def _backward_bfs(self, d_out: TensorLike) -> TensorLike:
        queue = deque([self.term])
        self.term.back_enqueue(d_out)
        self.term.b_visited = True

        while queue:
            cur = queue.popleft()
            d_out_arr = cur.backward()

            for prev, dx in zip(cur.prev_nodes, d_out_arr):
                prev.back_enqueue(dx)
                if not prev.b_visited:
                    prev.b_visited = True
                    queue.append(prev)

            cur.flush()

        d_out = d_out_arr.pop()
        if d_out_arr:
            raise RuntimeError(f"'{self}' has more than one root nodes!")

        return d_out

    def get_path(
        self,
        start: LayerNode | None = None,
        end: LayerNode | None = None,
    ) -> List[LayerNode]:
        if not self.built_:
            raise NotFittedError(
                f"'{self}' has not built! Call 'build()' to build the graph."
            )
        path = []
        cur = self.root if start is None else start
        final = self.term if end is None else end

        while cur != final:
            path.append(cur)
            if not cur.next_nodes:
                raise RuntimeError(f"Path is broken at '{cur}'!")
            cur = cur.next_nodes[0]

        path.append(final)
        return path

    def clear(self) -> None:
        self.graph.clear()
        self.nodes.clear()
        self.built_ = False

    @property
    def param_size(self) -> Tuple[int, int]:
        w_size, b_size = 0, 0
        for node in self.nodes:
            w_, b_ = node.param_size
            w_size += w_
            b_size += b_

        return w_size, b_size

    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        return self.term.out_shape(in_shape)

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, LayerGraph):
            raise ValueError(f"Can only add another 'LayerGraph'!")

        new_graph = LayerGraph()
        merged_graph: Dict[LayerNode, list] = {**self.graph}

        for node, edges in other.graph.items():
            if node in merged_graph:
                merged_graph[node].extend(edges)
            else:
                merged_graph[node] = edges

        new_graph.graph = merged_graph
        if self.term in new_graph.graph:
            new_graph[self.term].append(other.root)
        else:
            new_graph.graph[self.term] = list(other.root)

        new_graph.root = self.root
        new_graph.term = other.term

        new_graph._init_predefined_graph()
        new_graph.build()

        return new_graph

    def __bool__(self) -> bool:
        return self.built_

    def __str__(self) -> str:
        return type(self).__name__

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, key_node: LayerNode) -> List[LayerNode]:
        return self.graph[key_node]
