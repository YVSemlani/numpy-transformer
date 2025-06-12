import numpy as np
from base import Module

class Softmax(Module):
    def __init__(self):
        pass

    def forward(self, x):
        x = np.exp(x)
        return x / np.sum(x, axis=-1)