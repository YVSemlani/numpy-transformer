import numpy as np
from base import Module
from linear import LinearLayer
from activations import ReLU
    
class FeedForward(Module):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        self.linear1 = LinearLayer(hidden_dim, hidden_dim)
        self.linear2 = LinearLayer(hidden_dim, hidden_dim)
        self.relu = ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x