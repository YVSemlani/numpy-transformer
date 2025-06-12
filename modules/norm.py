import numpy as np
from base import Module

class LayerNormalization(Module):
    def __init__(self, hidden_dim, eps=1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(hidden_dim)  # scale parameter
        self.beta = np.zeros(hidden_dim)  # shift parameter

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)  # Add epsilon for numerical stability
        x_norm = (x - mean) / std
        
        # Apply learnable transformation
        return self.gamma * x_norm + self.beta