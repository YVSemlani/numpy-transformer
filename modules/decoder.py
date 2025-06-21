import numpy as np

from modules.base import Module
from modules.norm import LayerNormalization
from modules.attention import SingleHeadAttention, MultiHeadAttention, SingleHeadCrossAttention
from modules.ffn import FeedForward
from modules.positional_encoding import PositionalEncoding

class Decoder(Module):
    def __init__(self, embedding_dim, num_layers, vocab_size, num_heads=1, debug=False, use_mask=False):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.debug = debug
        self.use_mask = use_mask
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(DecoderBlock(embedding_dim, num_heads, debug=debug, use_mask=use_mask))

    def forward(self, x, encoder_output):
        if self.debug:
            print(f"      📍 Positional encoding input: {x.shape}")
        
        x = self.positional_encoding(x)
        if self.debug:
            print(f"      ➕ After positional encoding: {x.shape}")
        
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_output)
            if self.debug:
                print(f"      📦 Layer {i+1}/{self.num_layers}: {x.shape}")
            
        return x
    
    def backward(self, dLdY):
        if self.debug:
            print(f"\n      ⬅️  Decoder backward pass")
        dLdenc_out_accumulated = 0
        for i, decoder_block in enumerate(reversed(self.layers)):
            layer_num = self.num_layers - i
            if self.debug:
                print(f"         📦 Layer {layer_num}: {dLdY.shape}")
            dLdenc_out, dLdY = decoder_block.backward(dLdY)
            if self.debug:
                print(f"            🔗 Encoder grad: {dLdenc_out.shape}")
            dLdenc_out_accumulated += dLdenc_out
        return dLdenc_out_accumulated, dLdY
    
    def zero_grad(self):
        """Clear gradients for all decoder layers"""
        for layer in self.layers:
            layer.zero_grad()

    def update(self, lr=1e-3):
        """Update all decoder layers"""
        for layer in self.layers:
            layer.update(lr)

class DecoderBlock(Module):
    def __init__(self, embedding_dim, num_heads, debug=False, use_mask=False):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.debug = debug
        self.use_mask = use_mask

        if num_heads == 1:
            self.masked_attention = SingleHeadAttention(embedding_dim)
            self.cross_attention = SingleHeadCrossAttention(embedding_dim)
        else:
            pass
            # TO-DO: add multi-head masked attention
            # TO-DO: add multi-head cross attention

        # Create separate norm layers for each component
        self.norm1 = LayerNormalization(embedding_dim)  # After masked attention
        self.norm2 = LayerNormalization(embedding_dim)  # After cross attention
        self.norm3 = LayerNormalization(embedding_dim)  # After FFN
        self.feed_forward = FeedForward(embedding_dim, embedding_dim, embedding_dim)

        # Store layers for update
        self.layers = [self.masked_attention, self.cross_attention, self.norm1, self.norm2, self.norm3, self.feed_forward]

    def forward(self, x, encoder_output):
        # create mask for masked attention
        seq_len = x.shape[1]
        if self.use_mask:
            mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        else:
            mask = np.zeros((seq_len, seq_len))

        if self.debug:
            print(f"         🎭 Masked attention input: {x.shape}")
        
        # Masked self-attention
        res = x 
        x = self.masked_attention(x, mask=mask)
        if self.debug:
            print(f"         🧠 After masked attention: {x.shape}")
        
        x = self.norm1(x + res)
        if self.debug:
            print(f"         🔄 After norm1 + residual: {x.shape}")

        # Cross attention
        res = x
        x = self.cross_attention(x, encoder_output) # Q = x (decoder), K,V = encoder_output
        if self.debug:
            print(f"         🔗 After cross attention: {x.shape}")
        
        x = self.norm2(x + res)
        if self.debug:
            print(f"         🔄 After norm2 + residual: {x.shape}")

        # Feed forward
        res = x
        x = self.feed_forward(x)
        if self.debug:
            print(f"         ⚡ After FFN: {x.shape}")
        
        x = self.norm3(x + res)
        if self.debug:
            print(f"         ✅ Block output: {x.shape}")
        
        return x
    
    def backward(self, dLdY):
        # process in reverse order
        if self.debug:
            print(f"            ⬅️  Block backward: {dLdY.shape}")
        
        dLdY = self.norm3.backward(dLdY)
        if self.debug:
            print(f"            🔄 After norm3: {dLdY.shape}")
        
        dLdY = self.feed_forward.backward(dLdY)
        if self.debug:
            print(f"            ⚡ After FFN: {dLdY.shape}")
        
        dLdY = self.norm2.backward(dLdY)
        if self.debug:
            print(f"            🔄 After norm2: {dLdY.shape}")
        
        dLdenc_out, dLdY = self.cross_attention.backward(dLdY)
        if self.debug:
            print(f"            🔗 After cross attention - enc: {dLdenc_out.shape}, dec: {dLdY.shape}")
        
        dLdY = self.norm1.backward(dLdY)
        if self.debug:
            print(f"            🔄 After norm1: {dLdY.shape}")
        
        dLdY = self.masked_attention.backward(dLdY)
        if self.debug:
            print(f"            🧠 After masked attention: {dLdY.shape}")
        
        return dLdenc_out, dLdY

    def zero_grad(self):
        """Clear gradients for all block components"""
        self.masked_attention.zero_grad()
        self.cross_attention.zero_grad()
        self.norm1.zero_grad()
        self.norm2.zero_grad()
        self.norm3.zero_grad()
        self.feed_forward.zero_grad()

    def update(self, lr=1e-3):
        """Update all block components"""
        self.masked_attention.update(lr)
        self.cross_attention.update(lr)
        self.norm1.update(lr)
        self.norm2.update(lr)
        self.norm3.update(lr)
        self.feed_forward.update(lr)