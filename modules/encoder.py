import numpy as np

from base import Module
from norm import LayerNormalization
from embedding import Embedding
from attention import SingleHeadAttention, MultiHeadAttention
from ffn import FeedForward

class Encoder(Module):
    def __init__(self, embedding_dim, num_layers, vocab_size, num_heads=1):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = Embedding(embedding_dim, vocab_size)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(embedding_dim, num_heads))

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderBlock(Module):
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        if num_heads == 1:
            self.attention = SingleHeadAttention(embedding_dim)
        else:
            self.attention = MultiHeadAttention(embedding_dim, num_heads)

        self.feed_forward = FeedForward(embedding_dim)
        self.norm = LayerNormalization(embedding_dim)

    def forward(self, x):
        # attention
        res = x
        x = self.attention(x)
        x = self.norm(x + res)

        # feed forward
        res = x
        x = self.feed_forward(x)
        x = self.norm(x + res)
        return x
    
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
    

test_encoder = Encoder(embedding_dim=4096, num_layers=1, vocab_size=100)
x = np.random.randint(0, 100, (10, 10))
x = test_encoder(x)
print(x.shape)
