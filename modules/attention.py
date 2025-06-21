import numpy as np
from modules.base import Module
from modules.linear import LinearLayer
from modules.functions import Softmax

class SingleHeadAttention(Module):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.w_q = LinearLayer(hidden_dim, hidden_dim)
        self.w_k = LinearLayer(hidden_dim, hidden_dim)
        self.w_v = LinearLayer(hidden_dim, hidden_dim)
        self.softmax = Softmax()

        self.intermediate_vars = {}
        self.gradients = {}

    def forward(self, x, mask=None):
        # Store input for backward pass
        self.intermediate_vars['x'] = x
        
        self.intermediate_vars['q'] = self.w_q.forward(x)
        self.intermediate_vars['k'] = self.w_k.forward(x)
        self.intermediate_vars['v'] = self.w_v.forward(x)

        self.intermediate_vars['kv'] = self.intermediate_vars['q'] @ np.transpose(self.intermediate_vars['k'], (0, 2, 1))
        self.intermediate_vars['kv'] = self.intermediate_vars['kv'] / np.sqrt(self.hidden_dim)

        if mask is not None:
            self.intermediate_vars['kv'] = self.intermediate_vars['kv'] + mask

        self.intermediate_vars['attn'] = self.softmax.forward(self.intermediate_vars['kv'])
        self.intermediate_vars['attn_v'] = self.intermediate_vars['attn'] @ self.intermediate_vars['v']
        return self.intermediate_vars['attn_v']
    
    def backward(self, dLdY):
                
        # dL/dV = attn.T @ dLdY
        dLdV = self.intermediate_vars['attn'].transpose(0, 2, 1) @ dLdY
        
        # dL/dAttn = dLdY @ v.T
        dLdAttn = dLdY @ self.intermediate_vars['v'].transpose(0, 2, 1)
        
        # softmax backward pass
        dLdScores = self.softmax.backward(dLdAttn)
        
        # apply d_k scaling factor
        dLdScores = dLdScores / np.sqrt(self.hidden_dim)
        
        # dL/dQ = dL/dScores @ K
        dLdQ = dLdScores @ self.intermediate_vars['k']
        
        # dL/dK = dL/dScores.T @ Q
        dLdK = dLdScores.transpose(0, 2, 1) @ self.intermediate_vars['q']
        
        # use linear layer backward passes for weights and biases
        # Get gradients from linear layers
        dLdX_q = self.w_q.backward(dLdQ)
        dLdX_k = self.w_k.backward(dLdK) 
        dLdX_v = self.w_v.backward(dLdV)
        
        # sum gradients from all three pathways to get gradient w.r.t. input X
        # feed this into the next layer (in backwards order)
        self.gradients['dLdX'] = dLdX_q + dLdX_k + dLdX_v
        
        return self.gradients['dLdX']

    def update(self, lr=1e-3):
        self.w_q.update(lr)
        self.w_k.update(lr)
        self.w_v.update(lr)

    def zero_grad(self):
        """Clear gradients and intermediate variables"""
        self.gradients = {}
        self.intermediate_vars = {}
        self.w_q.zero_grad()
        self.w_k.zero_grad()
        self.w_v.zero_grad()

# TODO: implement this
class MultiHeadAttention(Module):
    def __init__(self, hidden_dim, num_heads):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        return x
    
    def backward(self, x):
        return x

class SingleHeadCrossAttention(Module):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.w_q = LinearLayer(hidden_dim, hidden_dim)
        self.w_k = LinearLayer(hidden_dim, hidden_dim)
        self.w_v = LinearLayer(hidden_dim, hidden_dim)
        self.softmax = Softmax()
        
        self.intermediate_vars = {}
        self.gradients = {}

    def forward(self, decoder_input, encoder_output, mask=None):
        # Store inputs for backward pass
        self.intermediate_vars['decoder_input'] = decoder_input
        self.intermediate_vars['encoder_output'] = encoder_output
        
        # FIXED: Q from decoder (x_1), K and V from encoder (x_2)
        self.intermediate_vars['q'] = self.w_q.forward(decoder_input) # decoder queries
        self.intermediate_vars['k'] = self.w_k.forward(encoder_output) # encoder keys
        self.intermediate_vars['v'] = self.w_v.forward(encoder_output) # encoder values
        
        self.intermediate_vars['kv'] = self.intermediate_vars['q'] @ np.transpose(self.intermediate_vars['k'], (0, 2, 1))
        self.intermediate_vars['kv'] = self.intermediate_vars['kv'] / np.sqrt(self.hidden_dim)

        if mask is not None:
            self.intermediate_vars['kv'] = self.intermediate_vars['kv'] + mask

        self.intermediate_vars['attn'] = self.softmax.forward(self.intermediate_vars['kv'])
        self.intermediate_vars['attn_v'] = self.intermediate_vars['attn'] @ self.intermediate_vars['v']
        return self.intermediate_vars['attn_v']
    
    def backward(self, dLdY):
        # Similar to self-attention, but gradients flow to two different inputs
        # Q gradients go to x_1 (decoder), K/V gradients go to x_2 (encoder)
        
        # dL/dV = attn.T @ dLdY
        dLdV = self.intermediate_vars['attn'].transpose(0, 2, 1) @ dLdY
        
        # dL/dAttn = dLdY @ v.T
        dLdAttn = dLdY @ self.intermediate_vars['v'].transpose(0, 2, 1)
        
        # softmax backward pass
        dLdScores = self.softmax.backward(dLdAttn)
        
        # apply d_k scaling factor
        dLdScores = dLdScores / np.sqrt(self.hidden_dim)
        
        # dL/dQ = dL/dScores @ K
        dLdQ = dLdScores @ self.intermediate_vars['k']
        
        # dL/dK = dL/dScores.T @ Q
        dLdK = dLdScores.transpose(0, 2, 1) @ self.intermediate_vars['q']
        
        # Get gradients from linear layers
        # Q gradients flow back to decoder input
        dLdX_1_q = self.w_q.backward(dLdQ)
        
        # K and V gradients flow back to encoder output
        dLdX_2_k = self.w_k.backward(dLdK)
        dLdX_2_v = self.w_v.backward(dLdV)
        
        # Combine gradients for each input
        # decoder input receives gradients only from Q pathway (decoder)
        self.gradients['dLdX_1'] = dLdX_1_q
        
        # encoder output receives gradients from both K and V pathways (encoder)
        self.gradients['dLdX_2'] = dLdX_2_k + dLdX_2_v
        
        # Return gradients for both inputs
        return self.gradients['dLdX_1'], self.gradients['dLdX_2']
    
    
    def update(self, lr=1e-3):
        self.w_q.update(lr)
        self.w_k.update(lr)
        self.w_v.update(lr)

    def zero_grad(self):
        """Clear gradients and intermediate variables"""
        self.gradients = {}
        self.intermediate_vars = {}
        self.w_q.zero_grad()
        self.w_k.zero_grad()
        self.w_v.zero_grad()

# TODO: implement this
class MultiHeadCrossAttention(Module):
    def __init__(self, hidden_dim, num_heads):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, decoder_input, encoder_output, mask=None):
        return decoder_input
    
    def backward(self, decoder_input, encoder_output):
        return decoder_input, encoder_output