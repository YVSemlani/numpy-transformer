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
    """Evaluate model on copy task using autoregressive generation"""
    batch_size = len(test_inputs)
    max_length = test_targets.shape[1]
    
    # Use autoregressive generation with proper start token
    # Note: START_TOKEN should be vocab_size (e.g., 3 when vocab is 0,1,2)
    START_TOKEN = 3  # Assuming vocab_size=3, so valid tokens are 0,1,2 and start token is 3
    predicted_tokens = generate_sequence_for_copy(model, test_inputs, max_length, start_token=START_TOKEN)
    accuracy = np.mean(predicted_tokens == test_targets)
    return accuracy

def generate_sequence_for_copy(model, encoder_input, max_length, start_token=3):
    """Generate sequences autoregressively for copy task"""
    batch_size = encoder_input.shape[0]
    
    # Start with just the start token
    decoder_input = np.full((batch_size, 1), start_token, dtype=np.int32)
    
    for _ in range(max_length):
        # Get predictions for current sequence
        predictions = model.forward(encoder_input, decoder_input)
        
        # Get the last predicted token
        next_token = np.argmax(predictions[:, -1, :], axis=-1)
        
        # Append to decoder input for next iteration
        decoder_input = np.concatenate([
            decoder_input, 
            next_token.reshape(-1, 1)
        ], axis=1)
    
    return decoder_input[:, 1:]  # Remove start token