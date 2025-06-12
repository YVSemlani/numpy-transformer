import numpy as np
from base import Module

class Embedding(Module):
    def __init__(self, embedding_dim, vocab_size):
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim)

    def forward(self, x):
        # takes tokens as input and looks up the embedding from the embedding matrix
        return self.embedding_matrix[x]