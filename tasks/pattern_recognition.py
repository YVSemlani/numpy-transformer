import numpy as np
from transformer import Transformer

def generate_pattern_dataset(seq_length=12, vocab_size=10, num_samples=1000, pattern_lengths=[2, 3, 4]):
    """Generate pattern recognition dataset
    
    Creates sequences with repeating patterns and asks model to predict the next token.
    For example: [1, 2, 1, 2, 1, 2] -> predict 1
                 [3, 1, 4, 3, 1, 4] -> predict 3
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Randomly choose pattern length
        pattern_len = np.random.choice(pattern_lengths)
        
        # Generate base pattern (avoiding 0 which might be padding)
        base_pattern = np.random.randint(1, vocab_size, pattern_len)
        
        # Calculate how many full repetitions we can fit
        num_reps = (seq_length - 1) // pattern_len
        remainder = (seq_length - 1) % pattern_len
        
        # Create sequence with repeated pattern
        sequence = np.tile(base_pattern, num_reps)
        if remainder > 0:
            sequence = np.concatenate([sequence, base_pattern[:remainder]])
        
        # Pad if necessary
        if len(sequence) < seq_length - 1:
            sequence = np.pad(sequence, (0, seq_length - 1 - len(sequence)), 
                            constant_values=0)
        
        # The target is the next token in the pattern
        next_pos_in_pattern = len(sequence) % pattern_len
        target = base_pattern[next_pos_in_pattern]
        
        inputs.append(sequence)
        targets.append(target)
    
    return np.array(inputs), np.array(targets)

def evaluate_pattern_task(model, test_inputs, test_targets):
    """Evaluate model on pattern recognition task"""
    predictions = model(test_inputs)
    
    # Get the prediction for the last position (next token prediction)
    if len(predictions.shape) == 3:  # (batch, seq_len, vocab_size)
        last_predictions = predictions[:, -1, :]
    else:  # (batch, vocab_size)
        last_predictions = predictions
    
    predicted_tokens = np.argmax(last_predictions, axis=-1)
    accuracy = np.mean(predicted_tokens == test_targets)
    return accuracy

def generate_sequence_pattern_dataset(seq_length=8, vocab_size=50, num_samples=1000):
    """Generate sequence-to-sequence pattern recognition dataset
    
    Creates input sequences with patterns and expects the model to output the complete pattern.
    Similar to copy task but with implicit pattern structure.
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generate pattern length (2-4 tokens)
        pattern_len = np.random.randint(2, 5)
        
        # Generate base pattern
        base_pattern = np.random.randint(1, vocab_size, pattern_len)
        
        # Create sequence with 1-2 full repetitions
        num_reps = np.random.randint(1, 3)
        sequence = np.tile(base_pattern, num_reps)
        
        # Truncate or pad to seq_length
        if len(sequence) > seq_length:
            sequence = sequence[:seq_length]
        else:
            sequence = np.pad(sequence, (0, seq_length - len(sequence)), 
                            constant_values=0)
        
        inputs.append(sequence)
        targets.append(sequence.copy())  # Target is to reproduce the input (like copy task)
    
    return np.array(inputs), np.array(targets)
