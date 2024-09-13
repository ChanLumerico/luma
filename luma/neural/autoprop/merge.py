from enum import Enum
import numpy as np

from luma.interface.typing import TensorLike


class MergeMode(Enum):
    CHCAT: str = "chcat"
    SUM: str = "sum"
    HADAMARD: str = "hadamard"
    AVG: str = "avg"
    MAX: str = "max"
    MIN: str = "min"
    DOT: str = "dot"
    SUB: str = "sub"

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
                stacked = np.stack(f_queue, axis=0)
                return np.max(stacked, axis=0)

            case MergeMode.MIN:
                stacked = np.stack(f_queue, axis=0)
                return np.min(stacked, axis=0)

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

            case MergeMode.MIN | MergeMode.MAX:
                stacked = np.stack(f_queue, axis=0)
                merged = np.max(stacked, axis=0)
                mask = f_queue[i] == merged

                total_mask = np.sum(
                    [tensor == merged for tensor in f_queue],
                    axis=0,
                )
                total_mask = np.clip(total_mask, a_min=1, a_max=None)

                grad = (d_out * mask / total_mask).astype(d_out.dtype)
                return grad

            case MergeMode.DOT:
                if i == 0:
                    return np.dot(d_out, f_queue[1].T)
                elif i == 1:
                    return np.dot(f_queue[0].T, d_out)

            case MergeMode.SUB:
                return d_out if i == 0 else -d_out
