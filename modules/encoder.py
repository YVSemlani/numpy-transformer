import numpy as np

from modules.base import Module
from modules.norm import LayerNormalization
from modules.attention import SingleHeadAttention, MultiHeadAttention
from modules.ffn import FeedForward
from modules.positional_encoding import PositionalEncoding

class Encoder(Module):
    def __init__(self, embedding_dim, num_layers, vocab_size, num_heads=1, debug=False):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.debug = debug

        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(embedding_dim, num_heads, debug=debug))

    def forward(self, x):
        if self.debug:
            print(f"      📍 Positional encoding input: {x.shape}")
        
        x = self.positional_encoding(x)
        if self.debug:
            print(f"      ➕ After positional encoding: {x.shape}")
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.debug:
                print(f"      📦 Layer {i+1}/{self.num_layers}: {x.shape}")
            
        return x
    
    def backward(self, dLdY):
        if self.debug:
            print(f"\n      ⬅️  Encoder backward pass")
        for i, encoder_block in enumerate(reversed(self.layers)):
            layer_num = self.num_layers - i
            if self.debug:
                print(f"         📦 Layer {layer_num}: {dLdY.shape}")
            dLdY = encoder_block.backward(dLdY)
            
        return dLdY

    def zero_grad(self):
        """Clear gradients for all encoder layers"""
        for layer in self.layers:
            layer.zero_grad()

    def update(self, lr=1e-3):
        """Update all encoder layers"""
        for layer in self.layers:
            layer.update(lr)

class EncoderBlock(Module):
    def __init__(self, embedding_dim, num_heads, debug=False):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.debug = debug

        if num_heads == 1:
            self.attention = SingleHeadAttention(embedding_dim)
        else:
            self.attention = MultiHeadAttention(embedding_dim, num_heads)

        self.feed_forward = FeedForward(embedding_dim, embedding_dim, embedding_dim)
        # Create separate norm layers for attention and FFN
        self.norm1 = LayerNormalization(embedding_dim)  # After attention
        self.norm2 = LayerNormalization(embedding_dim)  # After FFN
       
        # Store layers for update
        self.layers = [self.attention, self.feed_forward, self.norm1, self.norm2]

    def forward(self, x):
        if self.debug:
            print(f"         🔍 Attention input: {x.shape}")
        
        # attention
        res = x
        x = self.attention(x)
        if self.debug:
            print(f"         🧠 After attention: {x.shape}")
        
        x = self.norm1(x + res)
        if self.debug:
            print(f"         🔄 After norm1 + residual: {x.shape}")

        # feed forward
        res = x
        x = self.feed_forward(x)
        if self.debug:
            print(f"         ⚡ After FFN: {x.shape}")
        
        x = self.norm2(x + res)
        if self.debug:
            print(f"         ✅ Block output: {x.shape}")
        
        return x
    
    def backward(self, dLdY):
        # Backward pass through the layers in reverse order
        if self.debug:
            print(f"            ⬅️  Block backward: {dLdY.shape}")
        
        # Last norm layer
        dLdY = self.norm2.backward(dLdY)
        if self.debug:
            print(f"            🔄 After norm2: {dLdY.shape}")
        
        # Feed forward layer
        dLdY = self.feed_forward.backward(dLdY)
        if self.debug:
            print(f"            ⚡ After FFN: {dLdY.shape}")
        
        # First norm layer
        dLdY = self.norm1.backward(dLdY)
        if self.debug:
            print(f"            🔄 After norm1: {dLdY.shape}")
        
        # Attention layer
        dLdY = self.attention.backward(dLdY)
        if self.debug:
            print(f"            🧠 After attention: {dLdY.shape}")
        
        return dLdY

    def zero_grad(self):
        """Clear gradients for all block components"""
        self.attention.zero_grad()
        self.feed_forward.zero_grad()
        self.norm1.zero_grad()
        self.norm2.zero_grad()

    def update(self, lr=1e-3):
        """Update all block components"""
        self.attention.update(lr)
        self.feed_forward.update(lr)
        self.norm1.update(lr)
        self.norm2.update(lr)
