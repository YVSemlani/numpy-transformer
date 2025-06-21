import numpy as np
from modules.base import Module

class CrossEntropyLoss(Module):
    def __init__(self):
        self.loss = 0.0
        self.epsilon = 1e-15  # prevents undefined log(0)
    
    def forward(self, y_pred, y_true):
        # Ensure predictions are clipped to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Handle different input shapes
        if len(y_pred.shape) == 3:  # (batch_size, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = y_pred.shape
            


            # Reshape for easier computation
            y_pred_flat = y_pred_clipped.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
            y_true_flat = y_true.reshape(-1)  # (batch_size * seq_len,)
            
            # Get log probabilities for each "token"
            log_probs = np.log(y_pred_flat)
            
            # Extract the log probabilities of the correct classes (for one hot encoded targets only the correct class is non-zero in the sum)
            correct_log_probs = log_probs[np.arange(len(y_true_flat)), y_true_flat.astype(int)]
            
            # Average over all positions and samples
            # We're computing per token loss so loss in invariant w.r.t seq_len
            self.loss = -np.mean(correct_log_probs)
            
        else:  # (batch_size, vocab_size)
            batch_size, vocab_size = y_pred.shape
            
            # Compute cross-entropy
            log_probs = np.log(y_pred_clipped)
            
            # Select the log probability of the correct class for each sample
            correct_log_probs = log_probs[np.arange(batch_size), y_true.astype(int)]
            
            # Average over batch
            self.loss = -np.mean(correct_log_probs)
        
        return self.loss
    
    def backward(self, y_pred, y_true):
        # Handle different input shapes
        if len(y_pred.shape) == 3:  # (batch_size, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = y_pred.shape
            
            # Create one-hot encoded targets
            y_true_flat = y_true.reshape(-1).astype(int)
            y_true_onehot = np.zeros((batch_size * seq_len, vocab_size))
            y_true_onehot[np.arange(len(y_true_flat)), y_true_flat] = 1
            y_true_onehot = y_true_onehot.reshape(batch_size, seq_len, vocab_size)
            
            # Gradient is (predicted - true) / (batch_size * seq_len)
            # / by batch_size * seq_len because we're computing per token loss and num tokens is diff for 3D vs 2D
            gradient = (y_pred - y_true_onehot) / (batch_size * seq_len)
            
        else:  # (batch_size, vocab_size)
            batch_size, vocab_size = y_pred.shape
            
            # Create one-hot encoded targets
            y_true_onehot = np.zeros((batch_size, vocab_size))
            y_true_onehot[np.arange(batch_size), y_true.astype(int)] = 1
            
            # Gradient is (predicted - true) / batch_size
            # / by batch_size because we're computing per sample loss and num samples is diff for 3D vs 2D
            gradient = (y_pred - y_true_onehot) / batch_size
        
        return gradient