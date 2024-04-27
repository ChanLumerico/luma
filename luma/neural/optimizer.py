from typing import Tuple
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Tensor


__all__ = (
    "SGDOptimizer",
    "MomentumOptimizer",
    "RMSPropOptimizer",
    "AdamOptimizer",
    "AdaGradOptimizer",
    "AdaDeltaOptimizer",
    "AdaMaxOptimizer",
    "AdamWOptimizer",
    "NAdamOptimizer",
)


class SGDOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001) -> None:
        super().__init__()
        self.learning_rate = learning_rate

        self.set_param_ranges({"learning_rate": ("0<,+inf", None)})
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        updated = []
        for w, grad_w in zip(W, dW):
            new_weight = w - self.learning_rate * grad_w
            updated.append(new_weight)
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        updated = []
        for b, grad_b in zip(B, dB):
            new_bias = b - self.learning_rate * grad_b
            updated.append(new_bias)
        return updated


class MomentumOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vel_weights = None
        self.vel_biases = None

        self.set_param_ranges(
            {"learning_rate": ("0<,+inf", None), "momentum": ("0,1", None)}
        )
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.vel_weights is None:
            self.vel_weights = [np.zeros_like(w) for w in W]

        updated = []
        for w, v_w, grad_w in zip(W, self.vel_weights, dW):
            v_w = self.momentum * v_w - self.learning_rate * grad_w
            updated.append(w + v_w)

        self.vel_weights = [
            self.momentum * v_w - self.learning_rate * grad_w
            for v_w, grad_w in zip(self.vel_weights, dW)
        ]
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.vel_biases is None:
            self.vel_biases = [np.zeros_like(b) for b in B]

        updated = []
        for b, v_b, grad_b in zip(B, self.vel_biases, dB):
            v_b = self.momentum * v_b - self.learning_rate * grad_b
            updated.append(b + v_b)

        self.vel_biases = [
            self.momentum * v_b - self.learning_rate * grad_b
            for v_b, grad_b in zip(self.vel_biases, dB)
        ]
        return updated


class RMSPropOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.sq_grad_weights = None
        self.sq_grad_biases = None

        self.set_param_ranges(
            {"learning_rate": ("0<,+inf", None), "decay_rate": ("0,1", None)}
        )
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.sq_grad_weights is None:
            self.sq_grad_weights = [np.zeros_like(w) for w in W]

        updated = []
        for w, sq_g_w, grad_w in zip(W, self.sq_grad_weights, dW):
            sq_g_w *= self.decay_rate
            sq_g_w += (1 - self.decay_rate) * np.square(grad_w)
            w -= self.learning_rate * grad_w / (np.sqrt(sq_g_w) + 1e-8)
            updated.append(w)

        self.sq_grad_weights = [
            self.decay_rate * sq_g_w + (1 - self.decay_rate) * np.square(grad_w)
            for sq_g_w, grad_w in zip(self.sq_grad_weights, dW)
        ]
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.sq_grad_biases is None:
            self.sq_grad_biases = [np.zeros_like(b) for b in B]

        updated = []
        for b, sq_g_b, grad_b in zip(B, self.sq_grad_biases, dB):
            sq_g_b *= self.decay_rate
            sq_g_b += (1 - self.decay_rate) * np.square(grad_b)
            b -= self.learning_rate * grad_b / (np.sqrt(sq_g_b) + 1e-8)
            updated.append(b)

        self.sq_grad_biases = [
            self.decay_rate * sq_g_b + (1 - self.decay_rate) * np.square(grad_b)
            for sq_g_b, grad_b in zip(self.sq_grad_biases, dB)
        ]
        return updated


class AdamOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "beta_1": ("0,1", None),
                "beta_2": ("0,1", None),
                "epsilon": ("0<,1", None),
            }
        )
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not (weights is None and biases is None):
            self.t += 1

        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in W]
            self.v_weights = [np.zeros_like(w) for w in W]

        updated = []
        for i in range(len(W)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * dW[i]
            )
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (
                1 - self.beta_2
            ) * np.square(dW[i])
            m_hat_w = self.m_weights[i] / (1 - self.beta_1**self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta_2**self.t)
            updated.append(
                W[i] - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            )
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in B]
            self.v_biases = [np.zeros_like(b) for b in B]

        updated = []
        for i in range(len(B)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * dB[i]
            )
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (
                1 - self.beta_2
            ) * np.square(dB[i])
            m_hat_b = self.m_biases[i] / (1 - self.beta_1**self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta_2**self.t)
            updated.append(
                B[i] - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            )
        return updated


class AdaGradOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.grad_accum_weights = None
        self.grad_accum_biases = None

        self.set_param_ranges({"learning_rate": ("0<,+inf", None)})
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.grad_accum_weights is None:
            self.grad_accum_weights = [np.zeros_like(w) for w in W]

        updated = []
        for w, grad_w, accum_w in zip(W, dW, self.grad_accum_weights):
            accum_w += grad_w**2
            adjusted_lr = self.learning_rate / (np.sqrt(accum_w) + 1e-8)

            new_weight = w - adjusted_lr * grad_w
            updated.append(new_weight)
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.grad_accum_biases is None:
            self.grad_accum_biases = [np.zeros_like(b) for b in B]

        updated = []
        for b, grad_b, accum_b in zip(B, dB, self.grad_accum_biases):
            accum_b += grad_b**2
            adjusted_lr = self.learning_rate / (np.sqrt(accum_b) + 1e-8)

            new_bias = b - adjusted_lr * grad_b
            updated.append(new_bias)
        return updated


class AdaDeltaOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, rho: float = 0.95, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.rho = rho
        self.epsilon = epsilon

        self.accum_grads_w = None
        self.accum_updates_w = None
        self.accum_grads_b = None
        self.accum_updates_b = None

        self.set_param_ranges({"rho": ("0,+inf", None), "epsilon": ("0<,1", None)})
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.accum_grads_w is None:
            self.accum_grads_w = [np.zeros_like(w) for w in W]
            self.accum_updates_w = [np.zeros_like(w) for w in W]

        updated = []
        for i in range(len(W)):
            self.accum_grads_w[i] = self.rho * self.accum_grads_w[i] + (
                1 - self.rho
            ) * (dW[i] ** 2)
            update_w = (
                -np.sqrt(
                    (self.accum_updates_w[i] + self.epsilon)
                    / (self.accum_grads_w[i] + self.epsilon)
                )
                * dW[i]
            )
            self.accum_updates_w[i] = self.rho * self.accum_updates_w[i] + (
                1 - self.rho
            ) * (update_w**2)
            new_weight = W[i] + update_w
            updated.append(new_weight)

        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.accum_grads_b is None:
            self.accum_grads_b = [np.zeros_like(b) for b in B]
            self.accum_updates_b = [np.zeros_like(b) for b in B]

        updated = []
        for i in range(len(B)):
            self.accum_grads_b[i] = self.rho * self.accum_grads_b[i] + (
                1 - self.rho
            ) * (dB[i] ** 2)
            update_b = (
                -np.sqrt(
                    (self.accum_updates_b[i] + self.epsilon)
                    / (self.accum_grads_b[i] + self.epsilon)
                )
                * dB[i]
            )
            self.accum_updates_b[i] = self.rho * self.accum_updates_b[i] + (
                1 - self.rho
            ) * (update_b**2)
            new_bias = B[i] + update_b
            updated.append(new_bias)

        return updated


class AdaMaxOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "beta_1": ("0,1", None),
                "beta_2": ("0,1", None),
                "epsilon": ("0<,1", None),
            }
        )
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not (weights is None and biases is None):
            self.t += 1

        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in W]
            self.v_weights = [np.zeros_like(w) for w in W]

        updated = []
        for i in range(len(W)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * dW[i]
            )
            self.v_weights[i] = np.maximum(
                self.beta_2 * self.v_weights[i], np.abs(dW[i])
            )
            m_hat_w = self.m_weights[i] / (1 - self.beta_1**self.t)
            updated.append(
                W[i]
                - (self.learning_rate / (self.v_weights[i] + self.epsilon)) * m_hat_w
            )
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in B]
            self.v_biases = [np.zeros_like(b) for b in B]

        updated = []
        for i in range(len(B)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * dB[i]
            )
            self.v_biases[i] = np.maximum(self.beta_2 * self.v_biases[i], np.abs(dB[i]))
            m_hat_b = self.m_biases[i] / (1 - self.beta_1**self.t)
            updated.append(
                B[i]
                - (self.learning_rate / (self.v_biases[i] + self.epsilon)) * m_hat_b
            )
        return updated


class AdamWOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.001,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "beta_1": ("0,1", None),
                "beta_2": ("0,1", None),
                "epsilon": ("0<,1", None),
                "weight_decay": ("0,1", None),
            }
        )
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not (weights is None and biases is None):
            self.t += 1

        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in W]
            self.v_weights = [np.zeros_like(w) for w in W]

        updated = []
        for i in range(len(W)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * dW[i]
            )
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (
                1 - self.beta_2
            ) * np.square(dW[i])
            m_hat_w = self.m_weights[i] / (1 - self.beta_1**self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta_2**self.t)
            weight_decay_term = self.weight_decay * W[i]
            updated.append(
                W[i]
                - self.learning_rate
                * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + weight_decay_term)
            )
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in B]
            self.v_biases = [np.zeros_like(b) for b in B]

        updated = []
        for i in range(len(B)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * dB[i]
            )
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (
                1 - self.beta_2
            ) * np.square(dB[i])
            m_hat_b = self.m_biases[i] / (1 - self.beta_1**self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta_2**self.t)
            updated.append(
                B[i] - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            )
        return updated


class NAdamOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0

        self.set_param_ranges(
            {
                "learning_rate": ("0<,+inf", None),
                "beta_1": ("0,1", None),
                "beta_2": ("0,1", None),
                "epsilon": ("0<,1", None),
            }
        )
        self.check_param_ranges()

    def update(
        self,
        weights: Tensor,
        biases: Tensor,
        grad_weights: Tensor,
        grad_biases: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not (weights is None and biases is None):
            self.t += 1

        super().update(weights, biases, grad_weights, grad_biases)
        return self.updated_weights, self.updated_biases

    def _update_weights(self, W: Tensor, dW: Tensor) -> Tensor:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in W]
            self.v_weights = [np.zeros_like(w) for w in W]

        updated = []
        for i in range(len(W)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * dW[i]
            )
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (
                1 - self.beta_2
            ) * np.square(dW[i])

            m_hat_w = self.m_weights[i] / (1 - self.beta_1 ** (self.t + 1))
            v_hat_w = self.v_weights[i] / (1 - self.beta_2**self.t)

            nadam_update_w = self.beta_1 * m_hat_w + (1 - self.beta_1) * dW[i] / (
                1 - self.beta_1**self.t
            )
            updated.append(
                W[i]
                - self.learning_rate * nadam_update_w / np.sqrt(v_hat_w + self.epsilon)
            )
        return updated

    def _update_biases(self, B: Tensor, dB: Tensor) -> Tensor:
        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in B]
            self.v_biases = [np.zeros_like(b) for b in B]

        updated = []
        for i in range(len(B)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * dB[i]
            )
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (
                1 - self.beta_2
            ) * np.square(dB[i])

            m_hat_b = self.m_biases[i] / (1 - self.beta_1 ** (self.t + 1))
            v_hat_b = self.v_biases[i] / (1 - self.beta_2**self.t)

            nadam_update_b = self.beta_1 * m_hat_b + (1 - self.beta_1) * dB[i] / (
                1 - self.beta_1**self.t
            )
            updated.append(
                B[i]
                - self.learning_rate * nadam_update_b / np.sqrt(v_hat_b + self.epsilon)
            )
        return updated
