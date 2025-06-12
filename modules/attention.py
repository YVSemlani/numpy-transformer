import numpy as np
from base import Module
from linear import LinearLayer
from functions import Softmax

class SingleHeadAttention(Module):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.w_q = LinearLayer(hidden_dim, hidden_dim)
        self.w_k = LinearLayer(hidden_dim, hidden_dim)
        self.w_v = LinearLayer(hidden_dim, hidden_dim)
        self.softmax = Softmax()

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        kv = q @ np.transpose(k, (0, 2, 1))
        kv = kv / np.sqrt(self.hidden_dim)
        attn = self.softmax(kv)
        attn_v = attn @ v
        return attn_v

class MultiHeadAttention(Module):
    def __init__(self, hidden_dim, num_heads):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, x):
        return x
    
    def backward(self, x):
        return x