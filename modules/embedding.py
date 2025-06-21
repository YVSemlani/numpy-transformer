import numpy as np
from modules.base import Module

class Embedding(Module):
    def __init__(self, vocab_size, embedding_dim, debug=False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.debug = debug

        # Scale embeddings by sqrt of embedding dimension
        # prevents exploding gradients
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) / np.sqrt(embedding_dim)
        
        # Store intermediate variables and gradients
        self.intermediate_vars = {}
        self.gradients = {}

    def forward(self, x, is_decoder=False):
        # Store input tokens for backward pass
        # if we're in the decoder then store tokens as decoder tokens and vice versa for encoder
        if is_decoder:
            if self.debug:
                print(f"      🎯 Decoder tokens: {x}")
            self.intermediate_vars['decoder_tokens'] = x
        else:
            if self.debug:
                print(f"      📝 Encoder tokens: {x}")
            self.intermediate_vars['encoder_tokens'] = x

        # takes tokens as input and looks up the embedding from the embedding matrix
        # lookup table is functionally the same as a linear layer bc only the relevant token row is being used (thus only that row gets updated)
        return self.embedding_matrix[x]
    
    def backward(self, dLdY, is_decoder=False):
        if is_decoder:
            input_tokens = self.intermediate_vars['decoder_tokens']
            if self.debug:
                print(f"      ⬅️  Computing decoder embedding gradients")
        else:
            input_tokens = self.intermediate_vars['encoder_tokens']
            if self.debug:
                print(f"      ⬅️  Computing encoder embedding gradients")

        # Initialize gradients wrt encoder outputs (bc they're used in the decoder cross attention) if they don't exist
        # Otherwise accumulate current gradient with the previous gradients so all cross attention gradients eventually get back to the last encoder layer
        if 'dLdE' not in self.gradients:
            self.gradients['dLdE'] = np.zeros_like(self.embedding_matrix)
        
        # Handle different input dimensions
        if input_tokens.ndim == 1:  # Single sequence: (seq_len,)
            for i, token_id in enumerate(input_tokens):
                self.gradients['dLdE'][token_id] += dLdY[i]
        elif input_tokens.ndim == 2:  # Batched sequences: (batch_size, seq_len)
            for batch_idx in range(input_tokens.shape[0]):
                for seq_idx in range(input_tokens.shape[1]):
                    token_id = input_tokens[batch_idx, seq_idx]
                    self.gradients['dLdE'][token_id] += dLdY[batch_idx, seq_idx]

        return None
    
    def update(self, lr=1e-3):
        if self.debug:
            print(f"      🔄 Updating embeddings (lr={lr})")
        if 'dLdE' in self.gradients:
            self.embedding_matrix -= lr * self.gradients['dLdE']
        return self.embedding_matrix

    def zero_grad(self):
        """Clear gradients and intermediate variables"""
        self.gradients = {}
        self.intermediate_vars = {}

        return self.embedding_matrix