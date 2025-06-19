import numpy as np
from modules.base import Module
from modules.linear import LinearLayer
from modules.activations import ReLU
    
class FeedForward(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.linear2 = LinearLayer(hidden_dim, output_dim)
        self.relu = ReLU()

        self.intermediate_vars = {}
        self.gradients = {}

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = self.relu(z1)
        z2 = self.linear2(a1)
        
        self.intermediate_vars['x'] = x
        self.intermediate_vars['z1'] = z1
        self.intermediate_vars['a1'] = a1
        self.intermediate_vars['z2'] = z2

        return z2

    def backward(self, dLdY):
        dLda1 = self.linear2.backward(dLdY)
        dLdZ1 = self.relu.backward(self.intermediate_vars['z1'], dLda1)
        dLdX = self.linear1.backward(dLdZ1)

        self.gradients['dLdX'] = dLdX

        return self.gradients['dLdX']
    
    def update(self, lr=1e-3):
        self.linear1.update(lr)
        self.linear2.update(lr)

    def zero_grad(self):
        """Clear gradients for all FFN components"""
        self.linear1.zero_grad()
        self.linear2.zero_grad()

        return