# Transformer Implementation in Pure NumPy

A basic but complete transformer architecture implementation built from scratch using only NumPy. I built this project to develop a deeper understanding of the transformer architecture by implementing every component without relying on deep learning frameworks and autograd libraries as a crutch.

## 🧠 Core Modules
- **Transformer**: Main model class tying together encoder-decoder architecture
- **Encoder/Decoder**: Multi-layer encoder and decoder stacks and their smaller encoder/decoder block components
- **Attention Mechanisms**:
  - Single-head self-attention
  - Single-head cross-attention (encoder-decoder attention)
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Layer normalization with learnable beta and gamma parameters
- **Embeddings**: Token embeddings with positional encoding
- **Linear Layers**: Fully connected layers with weight matrices and biases
- **Activations**: ReLU activation function
- **Loss**: Cross-entropy loss with optimized processing

## 📁 Project Structure

```
transformer-mlx/
├── transformer.py          # Main transformer class
├── train.py               # Training script with copy task
├── modules/               # Core transformer components
│   ├── attention.py       # Self and cross-attention
│   ├── encoder.py         # Encoder layers
│   ├── decoder.py         # Decoder layers
│   ├── embedding.py       # Token embeddings
│   ├── ffn.py            # Feed-forward networks
│   ├── norm.py           # Layer normalization
│   ├── linear.py         # Linear layers
│   ├── loss.py           # Cross-entropy loss
│   └── ...
├── tasks/
│   └── copy.py           # Copy task implementation
├── tests/                # Don't use these tests, they were only used during development
│   ├── test_transformer.py
│   ├── test_attention_gradients.py
│   ├── test_ffn_grads.py
│   └── test_layernorm_grads.py
└── img/                  # Training result visualizations
```

## ⚡ Optimal Model Hyper Parameters

**Note**: These hyperparameters were chosen through trial and error. A Grid Search may yield better results.

- **Embedding Dimension**: 64
- **Number of Layers**: 1 (configurable)
- **Vocabulary Size**: 3 tokens + START token
- **Sequence Length**: 8 tokens
- **Batch Size**: 16

## 🔧 Implementation Details

- **Pure NumPy Implementation**: No PyTorch, TensorFlow, or other ML frameworks
- **Complete Backpropagation**: Derived and manually implemented gradients for all components
- **Manual Forward Pass**: Manually implemented forward pass for each component
- **Mini-Batch Gradient Descent**: Parameters updated and gradients + intermediate values cleared every mini-batch
- **Learning Rate Scheduler**: Implemented learning rate scheduler with decay from 5e-3 to 1e-3
- **Generation Modes**: Autoregressive and teacher-forcing (greedy) generation

## 🏋️ Training Task

Trained and evaluated on a simplistic **copy task** where the model learns to reproduce input sequences:
- Input: Random sequences of tokens
- Target: Identical sequences (copy the input)
- Autoregressive generation during evaluation
- Start token mechanism for proper decoder initialization

## 🚀 Usage

```python

# Train for 100 epochs
python train.py
```


## 📊 Results

The model successfully learns the copy task, demonstrating:
- Proper gradient flow through all components
- Learning on a simple but non-trivial task
- Correct implementation of attention mechanisms
- Semi-stable training dynamics

Training visualizations show convergence of both loss and accuracy metrics over epochs.

## 🎯 Future Improvements

- [ ] Multi-head attention implementation
- [ ] KV Caching
- [ ] Efficient attention implementation
- [ ] GPU acceleration
- [ ] More complex training tasks
- [ ] Optimization algorithms (Adam, etc.)
- [ ] Model scaling experiments
- [ ] New components (e.g. activations, losses, normalization, etc.)

## 📧 Contact Information

For any questions, feedback, or suggestions, reach out to me on X [@YVSemlani](https://x.com/YVSemlani).


