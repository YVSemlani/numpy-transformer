import numpy as np

from modules.base import Module
from modules.embedding import Embedding
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.linear import LinearLayer
from modules.functions import Softmax


class Transformer(Module):
    def __init__(self, embedding_dim, num_layers, vocab_size, num_heads=1, debug=False):
        self.debug = debug
        
        self.embedding = Embedding(vocab_size, embedding_dim, debug=debug)
        self.encoder = Encoder(embedding_dim, num_layers, vocab_size, num_heads, debug=debug)
        self.decoder = Decoder(embedding_dim, num_layers, vocab_size, num_heads, debug=debug)
        self.output_projection = LinearLayer(embedding_dim, vocab_size)
        self.softmax = Softmax()
        
        self.layers = [self.embedding, self.encoder, self.decoder, self.output_projection]


    def forward(self, x, output_tokens):
        if self.debug:
            print("\n" + "="*60)
            print("🔄 TRANSFORMER FORWARD PASS")
            print("="*60)
            print(f"📥 Input shapes - encoder: {x.shape}, decoder: {output_tokens.shape}")
        
        # Encoder path
        if self.debug:
            print("\n📊 Encoder Pipeline:")
        x = self.embedding(x, is_decoder=False)
        if self.debug:
            print(f"   ├─ After embedding: {x.shape}")
        
        encoder_output = self.encoder(x)
        if self.debug:
            print(f"   └─ Encoder output: {encoder_output.shape}")

        # Decoder path  
        if self.debug:
            print("\n🔄 Decoder Pipeline:")
        output_tokens = self.embedding(output_tokens, is_decoder=True)
        if self.debug:
            print(f"   ├─ After embedding: {output_tokens.shape}")
        
        predicted_tokens = self.decoder(output_tokens, encoder_output)
        if self.debug:
            print(f"   ├─ Decoder output: {predicted_tokens.shape}")
        
        predicted_tokens = self.output_projection(predicted_tokens)
        if self.debug:
            print(f"   ├─ After output projection: {predicted_tokens.shape}")
        
        predicted_tokens = self.softmax(predicted_tokens)
        if self.debug:
            print(f"   └─ Final output (post-softmax): {predicted_tokens.shape}")
            print("="*60 + "\n")
        
        return predicted_tokens
    
    def backward(self, dLdYpred):
        if self.debug:
            print("\n" + "="*60)
            print("⬅️  TRANSFORMER BACKWARD PASS")
            print("="*60)
            print(f"📥 Input gradient: {dLdYpred.shape}")
            print("\n🔄 Backward Pipeline:")
        
        dLdY = self.output_projection.backward(dLdYpred)
        if self.debug:
            print(f"   ├─ After projection: {dLdY.shape}")
        
        dLdenc_out, dLdY_decoder = self.decoder.backward(dLdY)
        if self.debug:
            print(f"   ├─ Decoder grads: {dLdY_decoder.shape}")

        # Accumulate decoder embedding gradients
        self.embedding.backward(dLdY_decoder, is_decoder=True)
        if self.debug:
            print(f"   ├─ ✅ Decoder embeddings updated")
        
        dLdY_encoder = self.encoder.backward(dLdenc_out)
        if self.debug:
            print(f"   ├─ Encoder grads: {dLdY_encoder.shape}")
        
        # Accumulate encoder embedding gradients  
        self.embedding.backward(dLdY_encoder, is_decoder=False)
        if self.debug:
            print(f"   └─ ✅ Encoder embeddings updated")
            print("="*60 + "\n")
        
        return dLdY_encoder

    def zero_grad(self):
        """Clear gradients for all components"""
        self.embedding.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.output_projection.zero_grad()
        self.softmax.zero_grad()

    def update(self, lr=0.001):
        """Update all trainable parameters in the transformer"""
        self.embedding.update(lr)
        self.encoder.update(lr)
        self.decoder.update(lr)
        self.output_projection.update(lr)
        # Note: softmax has no trainable parameters, so no update needed



