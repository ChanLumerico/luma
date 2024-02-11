import numpy as np


# TODO: Refine to fit in `luma`

class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, weights, biases, grad_weights, grad_biases):
        updated_weights = [w - self.learning_rate * grad_w for w, grad_w in zip(weights, grad_weights)]
        updated_biases = [b - self.learning_rate * grad_b for b, grad_b in zip(biases, grad_biases)]
        return updated_weights, updated_biases


class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_weights = None
        self.velocity_biases = None

    def update_params(self, weights, biases, grad_weights, grad_biases):
        if self.velocity_weights is None:
            self.velocity_weights = [np.zeros_like(w) for w in weights]
        if self.velocity_biases is None:
            self.velocity_biases = [np.zeros_like(b) for b in biases]
        
        updated_weights = []
        updated_biases = []
        for w, v_w, grad_w in zip(weights, self.velocity_weights, grad_weights):
            v_w = self.momentum * v_w - self.learning_rate * grad_w
            w += v_w
            updated_weights.append(w)
            self.velocity_weights = updated_weights

        for b, v_b, grad_b in zip(biases, self.velocity_biases, grad_biases):
            v_b = self.momentum * v_b - self.learning_rate * grad_b
            b += v_b
            updated_biases.append(b)
            self.velocity_biases = updated_biases

        return updated_weights, updated_biases

