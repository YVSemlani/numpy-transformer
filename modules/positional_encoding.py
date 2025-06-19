import numpy as np

from modules.base import Module

class PositionalEncoding(Module):
    def __init__(self, embedding_dim, max_length=5000):
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        
        # Create position and dimension grids
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(embedding_dim)[np.newaxis, :]
        
        # positional encoding formula sans trigonometric functions
        pre_trig = pos / np.power(10000, 2 * i / embedding_dim)
        
        # Apply sin/cos based on even/odd indices
        positional_encoding = np.where(
            i % 2 == 0,
            np.sin(pre_trig),
            np.cos(pre_trig)
        )
        
        # Expand to match batch dimension: (seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        positional_encoding = np.expand_dims(positional_encoding, axis=0)
        positional_encoding = np.repeat(positional_encoding, batch_size, axis=0)
        
        return x + positional_encoding