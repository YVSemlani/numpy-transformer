import numpy as np
from base import Module

class LinearLayer(Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        std = np.sqrt(2.0 / (input_dim + output_dim))  # Xavier normal initialization
        self.weights = np.random.normal(0, std, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias