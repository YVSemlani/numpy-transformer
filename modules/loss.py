import numpy as np
from modules.base import Module

class MSE(Module):
    def __init__(self):
        self.loss = 0.0

    def forward(self, y_pred, y_true):
        self.loss = 0.0
        self.loss = np.mean((y_pred - y_true) ** 2)

        return self.loss

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size

class CrossEntropyLoss(Module):
    def __init__(self):
        self.loss = 0.0
        self.epsilon = 1e-15  # Small value to prevent log(0)
    
    def forward(self, y_pred, y_true):
        # Ensure predictions are clipped to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Handle different input shapes
        if len(y_pred.shape) == 3:  # (batch_size, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = y_pred.shape
            
            # Reshape for easier computation
            y_pred_flat = y_pred_clipped.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
            y_true_flat = y_true.reshape(-1)  # (batch_size * seq_len,)
            
            # Compute cross-entropy for each position
            log_probs = np.log(y_pred_flat)
            
            # Select the log probability of the correct class for each sample
            correct_log_probs = log_probs[np.arange(len(y_true_flat)), y_true_flat.astype(int)]
            
            # Average over all positions and samples
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
        """
        Compute gradient of cross-entropy loss with respect to predictions
        
        Args:
            y_pred: Predicted probabilities, same shape as forward pass
            y_true: True labels, same shape as forward pass
            
        Returns:
            Gradient with respect to y_pred
        """
        # Handle different input shapes
        if len(y_pred.shape) == 3:  # (batch_size, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = y_pred.shape
            
            # Create one-hot encoded targets
            y_true_flat = y_true.reshape(-1).astype(int)
            y_true_onehot = np.zeros((batch_size * seq_len, vocab_size))
            y_true_onehot[np.arange(len(y_true_flat)), y_true_flat] = 1
            y_true_onehot = y_true_onehot.reshape(batch_size, seq_len, vocab_size)
            
            # Gradient is (predicted - true) / (batch_size * seq_len)
            gradient = (y_pred - y_true_onehot) / (batch_size * seq_len)
            
        else:  # (batch_size, vocab_size)
            batch_size, vocab_size = y_pred.shape
            
            # Create one-hot encoded targets
            y_true_onehot = np.zeros((batch_size, vocab_size))
            y_true_onehot[np.arange(batch_size), y_true.astype(int)] = 1
            
            # Gradient is (predicted - true) / batch_size
            gradient = (y_pred - y_true_onehot) / batch_size
        
        return gradient

class SparseCrossEntropyLoss(Module):
    """
    Optimized version for when targets are sparse (integer labels)
    More memory efficient than one-hot encoding
    """
    def __init__(self):
        self.loss = 0.0
        self.epsilon = 1e-15
    
    def forward(self, y_pred, y_true):
        """Sparse cross-entropy - more efficient for large vocabularies"""
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        if len(y_pred.shape) == 3:
            batch_size, seq_len, vocab_size = y_pred.shape
            
            # Flatten predictions and targets
            y_pred_flat = y_pred_clipped.reshape(-1, vocab_size)
            y_true_flat = y_true.reshape(-1).astype(int)
            
            # Compute log probabilities and select correct classes
            log_probs = np.log(y_pred_flat[np.arange(len(y_true_flat)), y_true_flat])
            self.loss = -np.mean(log_probs)
            
        else:
            batch_size, vocab_size = y_pred.shape
            y_true_int = y_true.astype(int)
            
            log_probs = np.log(y_pred_clipped[np.arange(batch_size), y_true_int])
            self.loss = -np.mean(log_probs)
        
        return self.loss
    
    def backward(self, y_pred, y_true):
        """Compute gradients without creating one-hot matrices"""
        if len(y_pred.shape) == 3:
            batch_size, seq_len, vocab_size = y_pred.shape
            total_samples = batch_size * seq_len
            
            gradient = y_pred.copy()
            y_true_flat = y_true.reshape(-1).astype(int)
            
            # Subtract 1 from the correct class probabilities
            for i in range(len(y_true_flat)):
                batch_idx = i // seq_len
                seq_idx = i % seq_len
                gradient[batch_idx, seq_idx, y_true_flat[i]] -= 1
            
            gradient = gradient / total_samples
            
        else:
            batch_size, vocab_size = y_pred.shape
            gradient = y_pred.copy()
            y_true_int = y_true.astype(int)
            
            # Subtract 1 from the correct class probabilities
            gradient[np.arange(batch_size), y_true_int] -= 1
            gradient = gradient / batch_size
        
        return gradient