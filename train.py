import numpy as np
from transformer import Transformer
from modules.loss import CrossEntropyLoss, SparseCrossEntropyLoss, MSE
import matplotlib.pyplot as plt

# Import task datasets
from tasks.copy import generate_copy_dataset, evaluate_copy_task
from tasks.addition import generate_addition_dataset, evaluate_addition_task
from tasks.reverse import generate_reverse_dataset, evaluate_reverse_task
from tasks.pattern_recognition import generate_pattern_dataset, evaluate_pattern_task

# Hyperparameters - Optimized for demonstration
embedding_dim = 128 # 128 also works reasonably well with this config
num_layers = 1
vocab_size = 3
seq_length = 8
START_TOKEN = vocab_size  # Use vocab_size as start token (e.g., 3 when vocab_size=3)
batch_size = 16         # Smaller batches for more frequent updates
# Learning rate schedule - start high and decay
initial_learning_rate = 5e-3
final_learning_rate = 1e-4
learning_rate = initial_learning_rate  # Will be updated during training
num_epochs = 100
debug = False           # Set to True to enable debug output
report_batch_freq = 200   # More frequent reporting for better demo visibility

# Initialize the transformer with debug parameter - need extra vocab slot for start token
model = Transformer(embedding_dim=embedding_dim, num_layers=num_layers, vocab_size=vocab_size+1, debug=debug)

# Initialize loss function - USE THIS INSTEAD OF MSE
loss_fn = CrossEntropyLoss()  # or SparseCrossEntropyLoss() for better memory efficiency

def generate_sequence(model, encoder_input, max_length, start_token=START_TOKEN):
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

def train_model(task_name="copy", enable_debug=False, batch_report_freq=10):
    """Train the model on a specific task
    
    Args:
        task_name: Task to train on ('copy', 'addition', 'reverse', 'pattern')
        enable_debug: Enable debug output from the model
        batch_report_freq: Report progress every N batches (0 to disable)
    """
    # Update model debug setting
    model.debug = enable_debug
    
    print(f"\n🚀 Training on {task_name} task..." + (" [DEBUG MODE]" if enable_debug else ""))
    
    # Generate dataset based on task
    if task_name == "copy":
        train_inputs, train_targets = generate_copy_dataset(seq_length, vocab_size, 1000)
        test_inputs, test_targets = generate_copy_dataset(seq_length, vocab_size, 100)
        eval_fn = evaluate_copy_task
    elif task_name == "addition":
        train_inputs, train_targets = generate_addition_dataset(max_num=50, num_samples=1000)
        test_inputs, test_targets = generate_addition_dataset(max_num=50, num_samples=200)
        eval_fn = evaluate_addition_task
    elif task_name == "reverse":
        train_inputs, train_targets = generate_reverse_dataset(seq_length, vocab_size, 1000)
        test_inputs, test_targets = generate_reverse_dataset(seq_length, vocab_size, 200)
        eval_fn = evaluate_reverse_task
    elif task_name == "pattern":
        train_inputs, train_targets = generate_pattern_dataset(seq_length, vocab_size//10, 1000)
        test_inputs, test_targets = generate_pattern_dataset(seq_length, vocab_size//10, 200)
        eval_fn = evaluate_pattern_task
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    print(f"📊 Dataset shapes: inputs {train_inputs.shape}, targets {train_targets.shape}")
    
    losses = []
    accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(train_inputs) // batch_size
        
        # Shuffle data
        indices = np.random.permutation(len(train_inputs))
        train_inputs_shuffled = train_inputs[indices]
        train_targets_shuffled = train_targets[indices]
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = train_inputs_shuffled[start_idx:end_idx]
            batch_targets = train_targets_shuffled[start_idx:end_idx]
            
            # For sequence-to-sequence tasks, decoder input setup
            if len(batch_targets.shape) == 2:  # Sequence outputs
                # Create decoder inputs by shifting targets right and adding start token
                decoder_inputs = np.full((batch_size, batch_targets.shape[1]), START_TOKEN, dtype=np.int32)
                decoder_inputs[:, 1:] = batch_targets[:, :-1]  # Shift right
                # decoder_inputs[:, 0] is START_TOKEN
            else:  # Single token outputs (like addition)
                decoder_inputs = np.zeros((batch_size, 1))  # Dummy input

            model.zero_grad()
            
            # Forward pass
            predictions = model.forward(batch_inputs, decoder_inputs)

            # one hot encode the targets
            #batch_targets = np.eye(vocab_size)[batch_targets]
            
            # Compute loss
            loss = loss_fn.forward(predictions, batch_targets)
            total_loss += loss
            
            # Backward pass
            grad = loss_fn.backward(predictions, batch_targets)
            model.backward(grad)
            
            # Update learning rate
            learning_rate = initial_learning_rate * (1 - (epoch + batch_idx / num_batches) / num_epochs)
            model.update(lr=learning_rate)
            
            # Batch-level reporting
            if batch_report_freq > 0 and (batch_idx + 1) % batch_report_freq == 0:
                running_avg_loss = total_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  📈 Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1:3d}/{num_batches} ({progress:5.1f}%) | "
                      f"Batch Loss: {loss:.4f} | Running Avg: {running_avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Evaluate at the end of every epoch - pick n random samples
        n = 2  # Number of random samples to evaluate
        random_indices = np.random.choice(len(test_inputs), n, replace=False)
        
        # Use autoregressive generation for sequence tasks
        if len(test_targets.shape) == 2:  # Sequence outputs
            test_predictions = generate_sequence(model, test_inputs[random_indices], 
                                               test_targets.shape[1], start_token=START_TOKEN)
        else:  # Single token outputs (like addition)
            test_predictions = model.forward(test_inputs[random_indices], 
                                           np.zeros((n, 1), dtype=np.int32))
            test_predictions = np.argmax(test_predictions, axis=-1)

        print(f"Test Predictions: {test_predictions}")
        print(f"Truth: {test_targets[random_indices]}")
        if task_name == "reverse":
            accuracy = eval_fn(model, test_inputs[:10], test_targets[:10])
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy['token_accuracy']:.3f}")
            acc_val = accuracy['token_accuracy']
        else:
            accuracy = eval_fn(model, test_inputs[:10], test_targets[:10])
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.3f}")
            acc_val = accuracy * 100
        losses.append(avg_loss)
        accuracies.append(acc_val)

    # Plot losses and accuracies after training
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', color='orange')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Train on different tasks with batch-level reporting
    # batch_report_freq: 1=every batch, 10=every 10 batches, 0=disabled
    
    # Train with frequent batch reporting for detailed monitoring
    train_model("copy", enable_debug=False, batch_report_freq=report_batch_freq)
    
    # Uncomment to try other tasks:
    # train_model("addition", enable_debug=True, batch_report_freq=1)  # Debug mode + every batch
    # train_model("reverse", enable_debug=False, batch_report_freq=10)  # Standard reporting
    # train_model("pattern", enable_debug=False, batch_report_freq=0)  # No batch reporting