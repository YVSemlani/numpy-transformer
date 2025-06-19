import numpy as np
from modules.base import Module

class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x, dLdY):
        dLdX = dLdY * (x > 0)

        return dLdX

    def zero_grad(self):
        """ReLU has no parameters or gradients to clear"""
        pass