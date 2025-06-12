import numpy as np
from base import Module

class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)