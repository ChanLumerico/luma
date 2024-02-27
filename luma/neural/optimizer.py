from typing import Tuple, List
import numpy as np

from luma.core.super import Optimizer
from luma.interface.util import Matrix


__all__ = (
    'SGDOptimizer',
    'MomentumOptimizer',
    'RMSPropOptimizer',
    'AdamOptimizer'
)


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    
    def update(self, 
               weights: List[Matrix], 
               biases: List[Matrix], 
               grad_weights: List[Matrix], 
               grad_biases: List[Matrix]) -> Tuple[List[Matrix]]:
        updated_weights = []
        for w, grad_w in zip(weights, grad_weights):
            new_weight = w - self.learning_rate * grad_w
            updated_weights.append(new_weight)
        
        updated_biases = []
        for b, grad_b in zip(biases, grad_biases):
            new_bias = b - self.learning_rate * grad_b
            updated_biases.append(new_bias)
        
        return updated_weights, updated_biases


class MomentumOptimizer(Optimizer):
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 momentum: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vel_weights = None
        self.vel_biases = None

    def update(self, 
               weights: List[Matrix], 
               biases: List[Matrix], 
               grad_weights: List[Matrix], 
               grad_biases: List[Matrix]) -> Tuple[List[Matrix]]:
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
        
        self.vel_weights = [self.momentum * v_w - self.learning_rate * grad_w 
                            for v_w, grad_w in zip(self.vel_weights, grad_weights)]
        
        self.vel_biases = [self.momentum * v_b - self.learning_rate * grad_b 
                           for v_b, grad_b in zip(self.vel_biases, grad_biases)]

        return updated_weights, updated_biases


class RMSPropOptimizer(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.01,
                 decay_rate: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.sq_grad_weights = None
        self.sq_grad_biases = None
    
    def update(self,
               weights: List[Matrix],
               biases: List[Matrix],
               grad_weights: List[Matrix],
               grad_biases: List[Matrix]) -> Tuple[List[Matrix]]:
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


class AdamOptimizer:
    def __init__(self, 
                 learning_rate: float = 0.001, 
                 beta_1: float = 0.9, 
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_weights = None  # First moment vector for weights
        self.v_weights = None  # Second moment vector for weights
        self.m_biases = None   # First moment vector for biases
        self.v_biases = None   # Second moment vector for biases
        self.t = 0  # Initialization of the timestep

    def update(self, 
               weights: Matrix, 
               biases: Matrix, 
               grad_weights: Matrix, 
               grad_biases: Matrix) -> Tuple[Matrix, Matrix]:
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]
        if self.m_biases is None:
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        updated_weights, updated_biases = [], []
        
        # Update parameters
        for i in range(len(weights)):
            self.m_weights[i] = self.beta_1 * self.m_weights[i] + (1 - self.beta_1) * grad_weights[i]
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (1 - self.beta_2) * np.square(grad_weights[i])
            m_hat_w = self.m_weights[i] / (1 - self.beta_1 ** self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta_2 ** self.t)
            updated_weights.append(weights[i] - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon))

        for i in range(len(biases)):
            self.m_biases[i] = self.beta_1 * self.m_biases[i] + (1 - self.beta_1) * grad_biases[i]
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (1 - self.beta_2) * np.square(grad_biases[i])
            m_hat_b = self.m_biases[i] / (1 - self.beta_1 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta_2 ** self.t)
            updated_biases.append(biases[i] - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))

        return updated_weights, updated_biases

