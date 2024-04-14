from typing import List, Tuple
import numpy as np

from luma.core.super import Optimizer
from luma.interface.util import Matrix


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
        self.learning_rate = learning_rate

        self.set_param_ranges({"learning_rate": ("0<,+inf", None)})
        self.check_param_ranges()

    def update(
        self,
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        updated_weights = []
        for w, grad_w in zip(weights, grad_weights):
            new_weight = w - self.learning_rate * grad_w
            updated_weights.append(new_weight)

        updated_biases = []
        for b, grad_b in zip(biases, grad_biases):
            new_bias = b - self.learning_rate * grad_b
            updated_biases.append(new_bias)

        return updated_weights, updated_biases


class MomentumOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.vel_weights is None:
            self.vel_weights = [np.zeros_like(w) for w in weights]
        if self.vel_biases is None:
            self.vel_biases = [np.zeros_like(b) for b in biases]

        updated_weights = []
        for w, v_w, grad_w in zip(weights, self.vel_weights, grad_weights):
            v_w = self.momentum * v_w - self.learning_rate * grad_w
            updated_weights.append(w + v_w)

        updated_biases = []
        for b, v_b, grad_b in zip(biases, self.vel_biases, grad_biases):
            v_b = self.momentum * v_b - self.learning_rate * grad_b
            updated_biases.append(b + v_b)

        self.vel_weights = [
            self.momentum * v_w - self.learning_rate * grad_w
            for v_w, grad_w in zip(self.vel_weights, grad_weights)
        ]

        self.vel_biases = [
            self.momentum * v_b - self.learning_rate * grad_b
            for v_b, grad_b in zip(self.vel_biases, grad_biases)
        ]

        return updated_weights, updated_biases


class RMSPropOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9) -> None:
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.sq_grad_weights is None:
            self.sq_grad_weights = [np.zeros_like(w) for w in weights]
        if self.sq_grad_biases is None:
            self.sq_grad_biases = [np.zeros_like(b) for b in biases]

        updated_weights = []
        for w, sq_g_w, grad_w in zip(weights, self.sq_grad_weights, grad_weights):
            sq_g_w *= self.decay_rate
            sq_g_w += (1 - self.decay_rate) * np.square(grad_w)
            w -= self.learning_rate * grad_w / (np.sqrt(sq_g_w) + 1e-8)
            updated_weights.append(w)

        updated_biases = []
        for b, sq_g_b, grad_b in zip(biases, self.sq_grad_biases, grad_biases):
            sq_g_b *= self.decay_rate
            sq_g_b += (1 - self.decay_rate) * np.square(grad_b)
            b -= self.learning_rate * grad_b / (np.sqrt(sq_g_b) + 1e-8)
            updated_biases.append(b)

        self.sq_grad_weights = [
            self.decay_rate * sq_g_w + (1 - self.decay_rate) * np.square(grad_w)
            for sq_g_w, grad_w in zip(self.sq_grad_weights, grad_weights)
        ]
        self.sq_grad_biases = [
            self.decay_rate * sq_g_b + (1 - self.decay_rate) * np.square(grad_b)
            for sq_g_b, grad_b in zip(self.sq_grad_biases, grad_biases)
        ]

        return updated_weights, updated_biases


class AdamOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]
        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]

        self.t += 1
        updated_weights, updated_biases = [], []

        for i in range(len(weights)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * grad_weights[i]
            )
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (
                1 - self.beta_2
            ) * np.square(grad_weights[i])
            m_hat_w = self.m_weights[i] / (1 - self.beta_1**self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta_2**self.t)
            updated_weights.append(
                weights[i]
                - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            )

        for i in range(len(biases)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * grad_biases[i]
            )
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (
                1 - self.beta_2
            ) * np.square(grad_biases[i])
            m_hat_b = self.m_biases[i] / (1 - self.beta_1**self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta_2**self.t)
            updated_biases.append(
                biases[i]
                - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            )

        return updated_weights, updated_biases


class AdaGradOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, learning_rate: float = 0.001) -> None:
        self.learning_rate = learning_rate
        self.grad_accum_weights = None
        self.grad_accum_biases = None

        self.set_param_ranges({"learning_rate": ("0<,+inf", None)})
        self.check_param_ranges()

    def update(
        self,
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.grad_accum_weights is None:
            self.grad_accum_weights = [np.zeros_like(w) for w in weights]
            self.grad_accum_biases = [np.zeros_like(b) for b in biases]

        updated_weights = []
        for w, grad_w, accum_w in zip(weights, grad_weights, self.grad_accum_weights):
            accum_w += grad_w**2
            adjusted_lr = self.learning_rate / (np.sqrt(accum_w) + 1e-8)
            new_weight = w - adjusted_lr * grad_w
            updated_weights.append(new_weight)

        updated_biases = []
        for b, grad_b, accum_b in zip(biases, grad_biases, self.grad_accum_biases):
            accum_b += grad_b**2
            adjusted_lr = self.learning_rate / (np.sqrt(accum_b) + 1e-8)
            new_bias = b - adjusted_lr * grad_b
            updated_biases.append(new_bias)

        return updated_weights, updated_biases


class AdaDeltaOptimizer(Optimizer, Optimizer.Neural):
    def __init__(self, rho: float = 0.95, epsilon: float = 1e-8) -> None:
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.accum_grads_w is None:
            self.accum_grads_w = [np.zeros_like(w) for w in weights]
            self.accum_updates_w = [np.zeros_like(w) for w in weights]
            self.accum_grads_b = [np.zeros_like(b) for b in biases]
            self.accum_updates_b = [np.zeros_like(b) for b in biases]

        updated_weights = []
        for i in range(len(weights)):
            self.accum_grads_w[i] = self.rho * self.accum_grads_w[i] + (
                1 - self.rho
            ) * (grad_weights[i] ** 2)
            update_w = (
                -np.sqrt(
                    (self.accum_updates_w[i] + self.epsilon)
                    / (self.accum_grads_w[i] + self.epsilon)
                )
                * grad_weights[i]
            )
            self.accum_updates_w[i] = self.rho * self.accum_updates_w[i] + (
                1 - self.rho
            ) * (update_w**2)
            new_weight = weights[i] + update_w
            updated_weights.append(new_weight)

        updated_biases = []
        for i in range(len(biases)):
            self.accum_grads_b[i] = self.rho * self.accum_grads_b[i] + (
                1 - self.rho
            ) * (grad_biases[i] ** 2)
            update_b = (
                -np.sqrt(
                    (self.accum_updates_b[i] + self.epsilon)
                    / (self.accum_grads_b[i] + self.epsilon)
                )
                * grad_biases[i]
            )
            self.accum_updates_b[i] = self.rho * self.accum_updates_b[i] + (
                1 - self.rho
            ) * (update_b**2)
            new_bias = biases[i] + update_b
            updated_biases.append(new_bias)

        return updated_weights, updated_biases


class AdaMaxOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]

        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]

        self.t += 1
        updated_weights, updated_biases = [], []

        for i in range(len(weights)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * grad_weights[i]
            )
            self.v_weights[i] = np.maximum(
                self.beta_2 * self.v_weights[i], np.abs(grad_weights[i])
            )
            m_hat_w = self.m_weights[i] / (1 - self.beta_1**self.t)
            updated_weights.append(
                weights[i]
                - (self.learning_rate / (self.v_weights[i] + self.epsilon)) * m_hat_w
            )

        for i in range(len(biases)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * grad_biases[i]
            )
            self.v_biases[i] = np.maximum(
                self.beta_2 * self.v_biases[i], np.abs(grad_biases[i])
            )
            m_hat_b = self.m_biases[i] / (1 - self.beta_1**self.t)
            updated_biases.append(
                biases[i]
                - (self.learning_rate / (self.v_biases[i] + self.epsilon)) * m_hat_b
            )

        return updated_weights, updated_biases


class AdamWOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.001,
    ):
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]

        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]

        self.t += 1
        updated_weights, updated_biases = [], []

        for i in range(len(weights)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * grad_weights[i]
            )
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (
                1 - self.beta_2
            ) * np.square(grad_weights[i])
            m_hat_w = self.m_weights[i] / (1 - self.beta_1**self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta_2**self.t)
            weight_decay_term = self.weight_decay * weights[i]
            updated_weights.append(
                weights[i]
                - self.learning_rate
                * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + weight_decay_term)
            )

        for i in range(len(biases)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * grad_biases[i]
            )
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (
                1 - self.beta_2
            ) * np.square(grad_biases[i])
            m_hat_b = self.m_biases[i] / (1 - self.beta_1**self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta_2**self.t)
            updated_biases.append(
                biases[i]
                - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            )

        return updated_weights, updated_biases


class NAdamOptimizer(Optimizer, Optimizer.Neural):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
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
        weights: List[Matrix],
        biases: List[Matrix],
        grad_weights: List[Matrix],
        grad_biases: List[Matrix],
    ) -> Tuple[List[Matrix], List[Matrix]]:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]

        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]

        self.t += 1
        updated_weights, updated_biases = [], []

        for i in range(len(weights)):
            self.m_weights[i] = (
                self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * grad_weights[i]
            )
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (
                1 - self.beta_2
            ) * np.square(grad_weights[i])

            m_hat_w = self.m_weights[i] / (1 - self.beta_1 ** (self.t + 1))
            v_hat_w = self.v_weights[i] / (1 - self.beta_2**self.t)

            nadam_update_w = self.beta_1 * m_hat_w + (1 - self.beta_1) * grad_weights[
                i
            ] / (1 - self.beta_1**self.t)
            updated_weights.append(
                weights[i]
                - self.learning_rate * nadam_update_w / np.sqrt(v_hat_w + self.epsilon)
            )

        for i in range(len(biases)):
            self.m_biases[i] = (
                self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * grad_biases[i]
            )
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (
                1 - self.beta_2
            ) * np.square(grad_biases[i])

            m_hat_b = self.m_biases[i] / (1 - self.beta_1 ** (self.t + 1))
            v_hat_b = self.v_biases[i] / (1 - self.beta_2**self.t)

            nadam_update_b = self.beta_1 * m_hat_b + (1 - self.beta_1) * grad_biases[
                i
            ] / (1 - self.beta_1**self.t)
            updated_biases.append(
                biases[i]
                - self.learning_rate * nadam_update_b / np.sqrt(v_hat_b + self.epsilon)
            )

        return updated_weights, updated_biases
