import numpy as np
from transformer import Transformer
from modules.loss import CrossEntropyLoss

from tasks.copy import CopyTask

import matplotlib.pyplot as plt

# Hyperparameters - Optimized for demonstration
embedding_dim = 64 
num_layers = 1
vocab_size = 3
seq_length = 8
START_TOKEN = vocab_size 
batch_size = 16

# Learning rate schedule - start high and decay
initial_learning_rate = 5e-3
final_learning_rate = 1e-3
learning_rate = initial_learning_rate  # Will be updated during training loop

num_epochs = 100

report_batch_freq = 200

# Initialize the transformer with debug parameter - need extra vocab slot for start token
model = Transformer(embedding_dim=embedding_dim, num_layers=num_layers, vocab_size=vocab_size+1)

# Set debug mode
model.debug = False

# Initialize loss function
loss_fn = CrossEntropyLoss()

# No optimizer bc the update step handles it natively
# This was a poor decision choice b/c we can't hotswap easily between optimizers

# Initialize task
copy_task = CopyTask(model, seq_length, vocab_size, 1000, batch_size)

if __name__ == "__main__":
    print(f"\n🚀 Training on copy task..." + (" [DEBUG MODE]" if model.debug else ""))

    print(f"📊 Dataset shapes: inputs {copy_task.train_inputs.shape}, targets {copy_task.train_targets.shape}")

    # Initialize lists to store losses and accuracies
    losses = []
    accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(copy_task.train_inputs) // batch_size
        
        # Shuffle data
        copy_task.shuffle_dataset()
        
        for batch_idx in range(num_batches):
            # Get batch
            batch_inputs, batch_targets = copy_task.get_batch()
            
            # Shift targets right and add start token
            decoder_inputs = copy_task.create_decoder_inputs(batch_targets)

            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            predictions = model.forward(batch_inputs, decoder_inputs)
            
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
            if report_batch_freq > 0 and (batch_idx + 1) % report_batch_freq == 0:
                running_avg_loss = total_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  📈 Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1:3d}/{num_batches} ({progress:5.1f}%) | "
                      f"Batch Loss: {loss:.4f} | Running Avg: {running_avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Evaluate at the end of every epoch - pick n random samples
        test_predictions, truth_predictions, snapshot_accuracy = copy_task.snapshot_eval()

        print(f"Test Predictions: {test_predictions}")
        print(f"Truth: {truth_predictions}")
        print(f"Snapshot Accuracy: {snapshot_accuracy*100:.3f}")

        accuracy = copy_task.evaluate_copy_task()
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.3f}")
        losses.append(avg_loss)
        accuracies.append(accuracy)

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