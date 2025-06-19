import numpy as np
from modules.base import Module

class LayerNormalization(Module):
    def __init__(self, hidden_dim, eps=1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(hidden_dim)  # scale parameter
        self.beta = np.zeros(hidden_dim)  # shift parameter

        self.intermediate_vars = {}
        self.gradients = {}

    def forward(self, x):
        # Store intermediate values for backward pass
        self.intermediate_vars['x'] = x
        self.intermediate_vars['mean'] = np.mean(x, axis=-1, keepdims=True)
        self.intermediate_vars['var'] = np.var(x, axis=-1, keepdims=True)
        self.intermediate_vars['std'] = np.sqrt(self.intermediate_vars['var'] + self.eps)  # Add epsilon for numerical stability
        self.intermediate_vars['x_norm'] = (x - self.intermediate_vars['mean']) / self.intermediate_vars['std']
        
        # Apply learnable transformation
        return self.gamma * self.intermediate_vars['x_norm'] + self.beta
    
    def backward(self, dLdY):
        # Get batch size and feature dimension
        N = self.intermediate_vars['x'].shape[-1]  # Number of features being normalized
        
        x_norm = self.intermediate_vars['x_norm']
        std = self.intermediate_vars['std']
        
        # Compute parameter gradients
        self.gradients['dLdGamma'] = np.sum(dLdY * x_norm, axis=(0, 1))  # Sum over batch and seq_len dimensions
        self.gradients['dLdBeta'] = np.sum(dLdY, axis=(0, 1))  # Sum over batch and seq_len dimensions
        
        # Compute input gradients using the correct layer norm backward formula
        self.gradients['dLdX_norm'] = dLdY * self.gamma
        
        # The complete gradient formula for layer normalization:
        # dL/dx = (gamma/std) * [dL/dy - mean(dL/dy) - x_norm * mean(dL/dy * x_norm)]
        self.gradients['mean_dLdX_norm'] = np.mean(self.gradients['dLdX_norm'], axis=-1, keepdims=True)
        self.gradients['mean_dLdX_norm_x_norm'] = np.mean(self.gradients['dLdX_norm'] * x_norm, axis=-1, keepdims=True)
        
        self.gradients['dLdX'] = (1.0 / std) * (self.gradients['dLdX_norm'] - self.gradients['mean_dLdX_norm'] - x_norm * self.gradients['mean_dLdX_norm_x_norm'])
        
        return self.gradients['dLdX']
    
    def update(self, lr=1e-3):
        self.gamma -= lr * self.gradients['dLdGamma']
        self.beta -= lr * self.gradients['dLdBeta']

    def zero_grad(self):
        """Clear gradients and intermediate variables"""
        self.gradients = {}
        self.intermediate_vars = {}

        return self.gamma, self.beta