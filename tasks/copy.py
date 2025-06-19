import numpy as np
from transformer import Transformer

def generate_copy_dataset(seq_length=8, vocab_size=50, num_samples=1000):
    """Generate copy task dataset"""
    # Generate random sequences using FULL vocab range [0, vocab_size-1]
    # This ensures all embedding indices are used during training
    inputs = np.random.randint(0, vocab_size, (num_samples, seq_length))
    targets = inputs.copy()
    
    return np.array(inputs), np.array(targets)

def evaluate_copy_task(model, test_inputs, test_targets):
    """Evaluate model on copy task"""
    batch_size = len(test_inputs)
    # Use proper decoder inputs: zeros for autoregressive generation
    decoder_inputs = np.zeros((batch_size, test_targets.shape[1]), dtype=np.int32)
    
    predictions = model.forward(test_inputs, decoder_inputs)
    predicted_tokens = np.argmax(predictions, axis=-1)
    accuracy = np.mean(predicted_tokens == test_targets)
    return accuracy