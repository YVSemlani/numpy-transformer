import numpy as np
from transformer import Transformer

class CopyTask:
    def __init__(self, model, seq_length, vocab_size, num_samples, batch_size, num_snapshots=2):
        self.model = model
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.START_TOKEN = self.vocab_size # start token is the last token in the vocab

        self.num_snapshots = num_snapshots

        self.current_batch_idx = 0

        self.train_inputs, self.train_targets = self.generate_copy_dataset(seed=42)
        self.test_inputs, self.test_targets = self.generate_copy_dataset(seed=24)
    

    def generate_copy_dataset(self, seed=None):
        """Generate copy task dataset"""
        # Generate random sequences using FULL vocab range [0, vocab_size-1]
        # This ensures all embedding indices are used during training
        if seed is not None:
            np.random.seed(seed)
        inputs = np.random.randint(0, self.vocab_size, (self.num_samples, self.seq_length))
        targets = inputs.copy()
        
        return np.array(inputs), np.array(targets)
    
    def shuffle_dataset(self):
        """Shuffle the dataset"""
        indices = np.random.permutation(len(self.train_inputs))
        self.train_inputs_shuffled = self.train_inputs[indices]
        self.train_targets_shuffled = self.train_targets[indices]

        return self.train_inputs_shuffled, self.train_targets_shuffled
    
    def get_batch(self):
        """Get a batch of data"""

        # Get batch indices
        # Stateful bc we need to of where we're at in the dataset
        start_idx = self.current_batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        # Get batch 
        # if the batch size is larger than the dataset we wrap around to the beginning of the dataset
        # technically this double trains some things but it should be fine..?
        batch_inputs = self.train_inputs_shuffled[start_idx:end_idx]
        batch_targets = self.train_targets_shuffled[start_idx:end_idx]
        
        # Update batch index
        # If we've reached the end of the dataset, shuffle dataset and reset batch index
        self.current_batch_idx += 1
        if self.current_batch_idx >= len(self.train_inputs_shuffled) // self.batch_size:
            self.current_batch_idx = 0
            self.shuffle_dataset()
        
        return batch_inputs, batch_targets
    
    def create_decoder_inputs(self, inputs):
        """Create decoder inputs for copy task"""

        decoder_inputs = np.full((inputs.shape[0], inputs.shape[1]), self.START_TOKEN, dtype=np.int32) # create array of all start tokens
        decoder_inputs[:, 1:] = inputs[:, :-1]
        return decoder_inputs
    
    def snapshot_eval(self):
        """Evaluate model on some random samples from the test set"""

        n = 2  # Number of random samples to evaluate
        random_indices = np.random.choice(len(self.test_inputs), n, replace=False)

        test_predictions = self.generate_sequence_for_copy(self.test_inputs[random_indices])

        truth_predictions = self.test_targets[random_indices]
        
        accuracy = np.mean(test_predictions == truth_predictions)
        return test_predictions, truth_predictions, accuracy

    def evaluate_copy_task(self):
        """Evaluate model on copy task using autoregressive generation"""

        # Use autoregressive generation with proper start token (vocab_size)
        predicted_tokens = self.generate_sequence_for_copy(self.test_inputs)
        accuracy = np.mean(predicted_tokens == self.test_targets)
        return accuracy

    def generate_sequence_for_copy(self, inputs):
        """Generate sequences autoregressively for copy task"""

        # decoder input is just the start token repeated over the sequence length
        # this is for autoregressive generation only bc the decoder input is the output of the previous step
        decoder_input = np.full((inputs.shape[0], 1), self.START_TOKEN, dtype=np.int32)

        for _ in range(self.seq_length):
            # Get predictions for current sequence
            predictions = self.model.forward(inputs, decoder_input)
            
            # Get the last predicted token
            next_token = np.argmax(predictions[:, -1, :], axis=-1)
        
            # store the next token in the decoder input
            decoder_input = np.concatenate([decoder_input, next_token.reshape(-1, 1)], axis=1)
        
        return decoder_input[:, 1:]  # Remove start token