import numpy as np
from modules.base import Module

class LinearLayer(Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        std = np.sqrt(2.0 / (input_dim + output_dim))  # Xavier normal initialization
        self.weights = np.random.normal(0, std, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)

        self.intermediate_vars = {}
        self.gradients = {}

    def forward(self, x):
        self.intermediate_vars['x'] = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, dLdY):
        # Handle both 2D and 3D (batched) inputs
        if self.intermediate_vars['x'].ndim == 3:  # Batched input: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = self.intermediate_vars['x'].shape
            
            # Reshape to 2D for weight gradient computation
            x_reshaped = self.intermediate_vars['x'].reshape(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)
            dLdY_reshaped = dLdY.reshape(-1, dLdY.shape[-1])  # (batch_size * seq_len, output_dim)
            
            # Compute gradients
            dLdW = x_reshaped.T @ dLdY_reshaped
            dLdb = np.sum(dLdY, axis=(0, 1))  # Sum over batch and sequence dimensions
            dLdX = dLdY @ self.weights.T  # Keep original shape for input gradient
            
        else:  # 2D input: (seq_len, hidden_dim) or (batch_size, hidden_dim)
            dLdW = self.intermediate_vars['x'].T @ dLdY
            dLdb = np.sum(dLdY, axis=0)
            dLdX = dLdY @ self.weights.T

        self.gradients['dLdW'] = dLdW
        self.gradients['dLdb'] = dLdb
        self.gradients['dLdX'] = dLdX

        return self.gradients['dLdX']

    def update(self, lr=1e-3):
        self.weights -= lr * self.gradients['dLdW']
        self.bias -= lr * self.gradients['dLdb']

        return self.weights, self.bias

    def zero_grad(self):
        """Clear gradients and intermediate variables"""
        self.gradients = {}
        self.intermediate_vars = {}

        return self.weights, self.bias
