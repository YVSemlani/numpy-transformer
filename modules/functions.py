import numpy as np
from modules.base import Module

class Softmax(Module):
    def __init__(self):
        self.output = None
        self.intermediate_vars = {}

    def forward(self, x):
        # Numerically stable softmax
        self.intermediate_vars['x'] = x

        # shift x so that max(x) = 0 thus no overflow
        # doesn't address underflow but we don't care about that b/c very negative x should tend to 0 anyway
        x_shifted = x - np.max(x, axis=-1, keepdims=True)

        # everything else is the same
        exp_x = np.exp(x_shifted)
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, dLdY):
        batch_size, seq_len_i, seq_len_j = self.output.shape
        dLdX = np.zeros_like(self.output)
        
        for b in range(batch_size):
            for i in range(seq_len_i):
                # Update gradient for each row of the attention matrix
                y_i = self.output[b, i, :]  # (seq_len_j,)
                dLdY_i = dLdY[b, i, :]      # (seq_len_j,)
                
                # Compute jacobian: diag(y_i) - y_i @ y_i.T (y_i is a vector so this is a matrix)
                jacobian = np.diag(y_i) - np.outer(y_i, y_i)
                dLdX[b, i, :] = jacobian @ dLdY_i
        
        return dLdX

    def zero_grad(self):
        """Clear intermediate variables"""
        self.intermediate_vars = {}
        self.output = None