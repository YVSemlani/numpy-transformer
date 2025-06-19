import numpy as np
from transformer import Transformer

hyperparameters = {
    'embedding_dim': 512,
    'num_layers': 6,
    'vocab_size': 100,
    'num_heads': 1,
    'lr': 1e-3,
}

def generate_addition_dataset(max_num=99, num_samples=1000, use_symbols=True):
    """Generate addition task dataset
    
    Creates addition problems in format: [num1, +, num2, =] -> sum
    For example: [3, +, 5, =] -> 8
                 [12, +, 7, =] -> 19
    """
    inputs = []
    targets = []
    
    # Define special tokens
    PLUS_TOKEN = 100  # Token for '+'
    EQUALS_TOKEN = 101  # Token for '='
    
    for _ in range(num_samples):
        # Generate two random numbers
        num1 = np.random.randint(0, max_num + 1)
        num2 = np.random.randint(0, max_num + 1)
        result = num1 + num2
        
        if use_symbols:
            # Format: [num1, +, num2, =]
            sequence = [num1, PLUS_TOKEN, num2, EQUALS_TOKEN]
        else:
            # Format: [num1, num2] (implicit addition)
            sequence = [num1, num2]
        
        inputs.append(sequence)
        targets.append(result)
    
    return np.array(inputs), np.array(targets)

def generate_digit_addition_dataset(max_digits=2, num_samples=1000):
    """Generate digit-by-digit addition dataset
    
    Breaks numbers into individual digits for sequence-to-sequence learning.
    For example: 23 + 45 -> [2, 3, +, 4, 5, =] -> [6, 8]
    """
    inputs = []
    targets = []
    
    PLUS_TOKEN = 10  # Token for '+'
    EQUALS_TOKEN = 11  # Token for '='
    
    for _ in range(num_samples):
        # Generate numbers with specified max digits
        max_val = 10**max_digits - 1
        num1 = np.random.randint(1, max_val + 1)
        num2 = np.random.randint(1, max_val + 1)
        result = num1 + num2
        
        # Convert numbers to digit sequences
        digits1 = [int(d) for d in str(num1)]
        digits2 = [int(d) for d in str(num2)]
        result_digits = [int(d) for d in str(result)]
        
        # Create input sequence: [d1, d2, +, d3, d4, =]
        sequence = digits1 + [PLUS_TOKEN] + digits2 + [EQUALS_TOKEN]
        
        # Pad sequences to consistent length
        max_result_len = max_digits + 1  # Account for potential carry
        if len(result_digits) < max_result_len:
            result_digits = [0] * (max_result_len - len(result_digits)) + result_digits
        
        inputs.append(sequence)
        targets.append(result_digits)
    
    return np.array(inputs, dtype=object), np.array(targets, dtype=object)

def evaluate_addition_task(model, test_inputs, test_targets):
    """Evaluate model on addition task"""
    predictions = model(test_inputs)
    
    # Handle different prediction formats
    if len(predictions.shape) == 3:  # (batch, seq_len, vocab_size)
        # For next-token prediction, use last position
        last_predictions = predictions[:, -1, :]
        predicted_tokens = np.argmax(last_predictions, axis=-1)
    else:  # (batch, vocab_size)
        predicted_tokens = np.argmax(predictions, axis=-1)
    
    # Calculate accuracy
    if len(test_targets.shape) == 1:  # Single number targets
        accuracy = np.mean(predicted_tokens == test_targets)
    else:  # Sequence targets (digit-by-digit)
        # For sequence targets, check if entire sequence matches
        correct = 0
        for i in range(len(test_targets)):
            if len(predictions.shape) == 3:
                pred_seq = np.argmax(predictions[i], axis=-1)
            else:
                pred_seq = predicted_tokens[i:i+1]
            
            if np.array_equal(pred_seq[:len(test_targets[i])], test_targets[i]):
                correct += 1
        accuracy = correct / len(test_targets)
    
    return accuracy

def generate_simple_addition_dataset(seq_length=5, max_num=10, num_samples=1000):
    """Generate simple addition dataset for sequence-to-sequence learning
    
    Format: [num1, num2, 0, 0, 0] -> [0, 0, 0, 0, sum]
    This treats addition as a sequence transformation task.
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generate two small numbers
        num1 = np.random.randint(1, max_num + 1)
        num2 = np.random.randint(1, max_num + 1)
        result = num1 + num2
        
        # Create input: [num1, num2, padding...]
        input_seq = [num1, num2] + [0] * (seq_length - 2)
        
        # Create target: [padding..., result]
        target_seq = [0] * (seq_length - 1) + [result]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return np.array(inputs), np.array(targets)
