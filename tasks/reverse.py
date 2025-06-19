import numpy as np
from transformer import Transformer

def generate_reverse_dataset(seq_length=8, vocab_size=50, num_samples=1000):
    """Generate sequence reversal dataset
    
    Creates sequences and their reversed versions.
    For example: [1, 2, 3, 4] -> [4, 3, 2, 1]
                 [5, 1, 8, 2] -> [2, 8, 1, 5]
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random sequence (avoiding 0 which might be padding)
        sequence = np.random.randint(1, vocab_size, seq_length)
        
        # Create reversed sequence
        reversed_sequence = sequence[::-1]
        
        inputs.append(sequence)
        targets.append(reversed_sequence)
    
    return np.array(inputs), np.array(targets)

def generate_partial_reverse_dataset(seq_length=10, vocab_size=30, num_samples=1000, reverse_start=3):
    """Generate partial sequence reversal dataset
    
    Reverses only a portion of the sequence.
    For example: [1, 2, 3, 4, 5, 6] with reverse_start=2 -> [1, 2, 6, 5, 4, 3]
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random sequence
        sequence = np.random.randint(1, vocab_size, seq_length)
        
        # Create partially reversed sequence
        target_sequence = sequence.copy()
        if reverse_start < len(sequence):
            target_sequence[reverse_start:] = target_sequence[reverse_start:][::-1]
        
        inputs.append(sequence)
        targets.append(target_sequence)
    
    return np.array(inputs), np.array(targets)

def generate_word_reverse_dataset(seq_length=12, vocab_size=20, num_samples=1000, separator_token=0):
    """Generate word-level reversal dataset
    
    Reverses words in a sequence separated by a separator token.
    For example: [1, 2, 0, 3, 4, 0, 5, 6] -> [5, 6, 0, 3, 4, 0, 1, 2]
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        sequence = []
        words = []
        current_word = []
        
        # Generate sequence with random "words" separated by separator_token
        for i in range(seq_length):
            if len(current_word) > 0 and (np.random.random() < 0.3 or i == seq_length - 1):
                # End current word
                words.append(current_word.copy())
                sequence.extend(current_word)
                current_word = []
                
                # Add separator if not at end
                if i < seq_length - 1 and len(sequence) < seq_length - 1:
                    sequence.append(separator_token)
            else:
                # Continue current word
                if len(sequence) < seq_length:
                    token = np.random.randint(1, vocab_size)
                    current_word.append(token)
        
        # Add remaining word if any
        if current_word and len(sequence) < seq_length:
            words.append(current_word)
            sequence.extend(current_word[:seq_length - len(sequence)])
        
        # Pad sequence if needed
        while len(sequence) < seq_length:
            sequence.append(separator_token)
        
        # Create reversed sequence by reversing word order
        if words:
            reversed_sequence = []
            for word in reversed(words):
                reversed_sequence.extend(word)
                if len(reversed_sequence) < seq_length - len(word):
                    reversed_sequence.append(separator_token)
            
            # Pad if necessary
            while len(reversed_sequence) < seq_length:
                reversed_sequence.append(separator_token)
            
            # Truncate if necessary
            reversed_sequence = reversed_sequence[:seq_length]
        else:
            reversed_sequence = sequence[::-1]
        
        inputs.append(sequence[:seq_length])
        targets.append(reversed_sequence[:seq_length])
    
    return np.array(inputs), np.array(targets)

def evaluate_reverse_task(model, test_inputs, test_targets):
    """Evaluate model on sequence reversal task"""
    predictions = model(test_inputs)
    predicted_tokens = np.argmax(predictions, axis=-1)
    
    # Calculate sequence-level accuracy (entire sequence must be correct)
    sequence_accuracy = np.mean([
        np.array_equal(pred, target) 
        for pred, target in zip(predicted_tokens, test_targets)
    ])
    
    # Calculate token-level accuracy
    token_accuracy = np.mean(predicted_tokens == test_targets)
    
    return {
        'sequence_accuracy': sequence_accuracy,
        'token_accuracy': token_accuracy
    }

def generate_conditional_reverse_dataset(seq_length=8, vocab_size=30, num_samples=1000, condition_token=99):
    """Generate conditional reversal dataset
    
    Reverses sequence only if condition token is present.
    For example: [1, 2, 3, 99] -> [99, 3, 2, 1] (reverse because 99 is present)
                 [1, 2, 3, 4] -> [1, 2, 3, 4] (no reverse because 99 is absent)
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random sequence
        sequence = np.random.randint(1, vocab_size, seq_length)
        
        # Randomly decide whether to include condition token
        should_reverse = np.random.random() < 0.5
        
        if should_reverse:
            # Replace random position with condition token
            pos = np.random.randint(0, seq_length)
            sequence[pos] = condition_token
            # Reverse the sequence
            target_sequence = sequence[::-1]
        else:
            # Keep sequence as is
            target_sequence = sequence.copy()
        
        inputs.append(sequence)
        targets.append(target_sequence)
    
    return np.array(inputs), np.array(targets)
