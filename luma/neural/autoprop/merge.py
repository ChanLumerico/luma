from enum import Enum
import numpy as np

from luma.interface.typing import TensorLike


class MergeMode(Enum):
    CHCAT = "chcat"
    SUM = "sum"
    HADAMARD = "hadamard"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    DOT = "dot"
    SUB = "sub"

    def forward(self, f_queue: list[TensorLike]) -> TensorLike:
        match self:
            case MergeMode.CHCAT:
                return np.concatenate(f_queue, axis=1)

            case MergeMode.SUM:
                return np.sum(f_queue, axis=0)

            case MergeMode.HADAMARD:
                X = np.ones_like(f_queue[0])
                for tensor in f_queue:
                    X *= tensor
                return X

            case MergeMode.AVG:
                return np.mean(f_queue, axis=0)

            case MergeMode.MAX:
                return np.maximum.reduce(f_queue)

            case MergeMode.MIN:
                return np.minimum.reduce(f_queue)

            case MergeMode.DOT:
                return np.dot(f_queue[0], f_queue[1])

            case MergeMode.SUB:
                result = f_queue[0]
                for tensor in f_queue[1:]:
                    result -= tensor
                return result

    def backward(
        self, f_queue: list[TensorLike], d_out: TensorLike, i: int
    ) -> TensorLike:
        match self:
            case MergeMode.CHCAT:
                cum_ch = [0]
                for tensor in f_queue:
                    cum_ch.append(cum_ch[-1] + tensor.shape[1])
                return d_out[:, cum_ch[i] : cum_ch[i + 1], ...]

            case MergeMode.SUM:
                return d_out

            case MergeMode.HADAMARD:
                total_prod = np.prod(f_queue, axis=0)
                prod_except_current = total_prod / f_queue[i]
                return d_out * prod_except_current

            case MergeMode.AVG:
                return d_out / len(f_queue)

            case MergeMode.MAX | MergeMode.MIN:
                return (
                    d_out * (f_queue[i] == getattr(np, self.value).reduce(f_queue))
                ).astype(d_out.dtype)

            case MergeMode.DOT:
                if i == 0:
                    return np.dot(d_out, f_queue[1].T)
                elif i == 1:
                    return np.dot(f_queue[0].T, d_out)

            case MergeMode.SUB:
                return d_out if i == 0 else -d_out
