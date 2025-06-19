import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.linear import LinearLayer
from modules.activations import ReLU
from modules.ffn import FeedForward
from modules.norm import LayerNormalization



class MLPWrapper:
    """Wrapper for FeedForward to provide consistent interface for testing gradients"""
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Use the FeedForward module we want to test
        self.ffn = FeedForward(input_dim, hidden_dim, output_dim)
        
        # Store intermediate values for analysis
        self.cache = {}
    
    def forward(self, x):
        """Forward pass through the network"""
        self.cache['x'] = x
        output = self.ffn.forward(x)
        self.cache['output'] = output
        return output
    
    def backward(self, dLdY):
        """Backward pass through the network"""
        gradients = self.ffn.backward(dLdY)
        return gradients
    
    def update(self, lr=1e-3):
        """Update all parameters"""
        self.ffn.update()

def generate_semi_linear_data(n_samples=500, noise_level=0.0):
    """Generate training data for a semi-linear function f(x) = x + sin(x/2)"""
    x = np.random.uniform(-2*np.pi, 2*np.pi, (n_samples, 1))
    y = x + np.sin(x/2) + noise_level * np.random.randn(n_samples, 1)
    return x, y

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_grad(y_true, y_pred):
    """Gradient of MSE loss"""
    return 2 * (y_pred - y_true) / y_pred.shape[0]

def test_ffn_gradients():
    """Test FeedForward gradients by training to learn sin(x)"""
    print("\n" + "="*70)
    print("⚡ FEEDFORWARD GRADIENT TEST")
    print("="*70)
    print("🎯 Task: Learning f(x) = x + sin(x/2)")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    n_samples = 500
    x_train, y_train = generate_semi_linear_data(n_samples, noise_level=0.0)
    
    # Generate test data for evaluation
    x_test = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
    y_test = x_test + np.sin(x_test/2)
    
    # Initialize model
    model = MLPWrapper(input_dim=1, hidden_dim=32, output_dim=1)
    
    # Training parameters
    num_epochs = 2000
    learning_rate = 1e-3
    print_interval = 100
    
    print(f"\n📊 Training Setup:")
    print(f"   ├─ Epochs: {num_epochs}")
    print(f"   ├─ Learning rate: {learning_rate}")
    print(f"   ├─ Hidden units: 32")
    print(f"   ├─ Training samples: {n_samples}")
    print(f"   └─ Test samples: {len(x_test)}")
    
    # Storage for training history
    train_losses = []
    test_losses = []
    
    print(f"\n🚀 Starting training...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model.forward(x_train)
        
        # Compute loss
        train_loss = mse_loss(y_train, y_pred)
        train_losses.append(train_loss)
        
        # Backward pass
        loss_grad = mse_loss_grad(y_train, y_pred)
        gradients = model.backward(loss_grad)
        
        # Update parameters
        model.update(learning_rate)
        
        # Evaluate on test set
        if epoch % print_interval == 0:
            y_test_pred = model.forward(x_test)
            test_loss = mse_loss(y_test, y_test_pred)
            test_losses.append(test_loss)
            
            print(f"📈 Epoch {epoch:4d}: Train={train_loss:.6f}, Test={test_loss:.6f}")
            
            # Check FFN layer parameters
            w1_norm = np.linalg.norm(model.ffn.linear1.weights)
            b1_norm = np.linalg.norm(model.ffn.linear1.bias)
            w2_norm = np.linalg.norm(model.ffn.linear2.weights)
            b2_norm = np.linalg.norm(model.ffn.linear2.bias)
            
            print(f"    ├─ FC1: ||W||={w1_norm:.4f}, ||b||={b1_norm:.4f}")
            print(f"    └─ FC2: ||W||={w2_norm:.4f}, ||b||={b2_norm:.4f}")
    
    # Final evaluation
    y_train_final = model.forward(x_train)
    y_test_final = model.forward(x_test)
    
    final_train_loss = mse_loss(y_train, y_train_final)
    final_test_loss = mse_loss(y_test, y_test_final)
    
    print(f"\n🎯 Training Results:")
    print(f"   ├─ Final train loss: {final_train_loss:.6f}")
    print(f"   └─ Final test loss: {final_test_loss:.6f}")
    
    # Print some sample predictions
    print(f"\n📊 Sample Predictions:")
    test_indices = [0, 25, 50, 75, 99]
    for i in test_indices:
        x_val = x_test[i, 0]
        y_true = y_test[i, 0]
        y_pred = y_test_final[i, 0]
        error = abs(y_true - y_pred)
        print(f"   x={x_val:6.3f}: true={y_true:6.3f}, pred={y_pred:6.3f}, err={error:6.3f}")
    
    # Test gradient computation specifically for FeedForward
    print(f"\n" + "="*60)
    print("🔬 GRADIENT COMPUTATION TEST")
    print("="*60)
    
    # Create small test case for gradient checking
    test_x = np.random.randn(5, 3)  # Small batch for easy verification
    test_ffn = FeedForward(input_dim=3, hidden_dim=4, output_dim=2)
    
    # Forward pass
    test_output = test_ffn.forward(test_x)
    
    # Backward pass with dummy gradient
    dummy_grad = np.ones_like(test_output)
    gradients = test_ffn.backward(dummy_grad)
    
    print(f"📊 Test Setup:")
    print(f"   ├─ Input shape: {test_x.shape}")
    print(f"   ├─ Output shape: {test_output.shape}")
    print(f"   └─ Gradient type: {type(gradients)}")
    
    # Check gradient information from the FFN layers
    print(f"\n🔍 Gradient Analysis:")
    print(f"   ├─ FC1 weight grad norm: {np.linalg.norm(test_ffn.gradients['dLdW1']):.6f}")
    print(f"   ├─ FC1 bias grad norm: {np.linalg.norm(test_ffn.gradients['dLdb1']):.6f}")
    print(f"   ├─ FC2 weight grad norm: {np.linalg.norm(test_ffn.gradients['dLdW2']):.6f}")
    print(f"   └─ FC2 bias grad norm: {np.linalg.norm(test_ffn.gradients['dLdb2']):.6f}")
    
    # Check for any NaN or infinite values in gradients
    def check_gradients_health(gradients_dict, name):
        print(f"\n🩺 {name} Gradient Health:")
        for grad_name, grad_value in gradients_dict.items():
            is_healthy = np.isfinite(grad_value).all()
            grad_norm = np.linalg.norm(grad_value)
            status = "✅" if is_healthy else "❌"
            print(f"   {status} {grad_name}: norm={grad_norm:.6f}, shape={grad_value.shape}")
    
    check_gradients_health(test_ffn.gradients, "FFN")
    print("="*70)
    
    return model, train_losses, test_losses, x_test, y_test, y_test_final

if __name__ == "__main__":
    # Run the test
    model, train_losses, test_losses, x_test, y_test, y_test_final = test_ffn_gradients()
    
    # Optional: Create a simple plot if matplotlib is available
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # Plot 2: Function approximation
        plt.subplot(1, 2, 2)
        plt.plot(x_test, y_test, 'b-', label='True f(x)', linewidth=2)
        plt.plot(x_test, y_test_final, 'r--', label='FFN prediction', linewidth=2)
        plt.title('Function Approximation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('img/ffn_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n📈 Plots saved as 'img/ffn_test_results.png'")
        
    except ImportError:
        print("\n📊 Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"\n❌ Error creating plots: {e}")



