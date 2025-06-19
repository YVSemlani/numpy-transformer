import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from modules.linear import LinearLayer
from modules.activations import ReLU
from modules.norm import LayerNormalization

class MLP:
    """Simple Multi-Layer Perceptron with LayerNorm for testing gradients"""
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Network layers
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.layernorm1 = LayerNormalization(hidden_dim)
        self.relu1 = ReLU()
        
        self.linear2 = LinearLayer(hidden_dim, hidden_dim)
        self.layernorm2 = LayerNormalization(hidden_dim)
        self.relu2 = ReLU()
        
        self.linear3 = LinearLayer(hidden_dim, output_dim)
        
        # Store intermediate values for backward pass
        self.cache = {}
    
    def forward(self, x):
        """Forward pass through the network"""
        # First hidden layer
        self.cache['x'] = x
        z1 = self.linear1.forward(x)
        self.cache['z1'] = z1
        
        n1 = self.layernorm1.forward(z1)
        self.cache['n1'] = n1
        
        a1 = self.relu1.forward(n1)
        self.cache['a1'] = a1
        
        # Second hidden layer
        z2 = self.linear2.forward(a1)
        self.cache['z2'] = z2
        
        n2 = self.layernorm2.forward(z2)
        self.cache['n2'] = n2
        
        a2 = self.relu2.forward(n2)
        self.cache['a2'] = a2
        
        # Output layer
        output = self.linear3.forward(a2)
        self.cache['output'] = output
        
        return output
    
    def backward(self, dLdY):
        """Backward pass through the network"""
        # Backward through output layer
        dLdW3, dLdb3, dLdA2 = self.linear3.backward(self.cache['a2'], dLdY)
        
        # Backward through second ReLU
        dLdN2 = self.relu2.backward(self.cache['n2'], dLdA2)
        
        # Backward through second LayerNorm
        dLdZ2, dLdGamma2, dLdBeta2 = self.layernorm2.backward(self.cache['z2'], dLdN2)
        
        # Backward through second linear layer
        dLdW2, dLdb2, dLdA1 = self.linear2.backward(self.cache['a1'], dLdZ2)
        
        # Backward through first ReLU
        dLdN1 = self.relu1.backward(self.cache['n1'], dLdA1)
        
        # Backward through first LayerNorm
        dLdZ1, dLdGamma1, dLdBeta1 = self.layernorm1.backward(self.cache['z1'], dLdN1)
        
        # Backward through first linear layer
        dLdW1, dLdb1, dLdX = self.linear1.backward(self.cache['x'], dLdZ1)
        
        return {
            'linear1': (dLdW1, dLdb1),
            'layernorm1': (dLdGamma1, dLdBeta1),
            'linear2': (dLdW2, dLdb2),
            'layernorm2': (dLdGamma2, dLdBeta2),
            'linear3': (dLdW3, dLdb3)
        }
    
    def update(self, lr=1e-3):
        """Update all parameters"""
        self.linear1.update(lr)
        self.layernorm1.update(lr)
        self.linear2.update(lr)
        self.layernorm2.update(lr)
        self.linear3.update(lr)

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

def test_layernorm_gradients():
    """Test LayerNorm gradients by training MLP to learn sin(x)"""
    print("Testing LayerNorm gradients by learning f(x) = sin(x)")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    n_samples = 500
    x_train, y_train = generate_semi_linear_data(n_samples, noise_level=0.0)
    
    # Generate test data for evaluation
    x_test = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
    y_test = x_test + np.sin(x_test/2)
    
    # Initialize model
    model = MLP(input_dim=1, hidden_dim=32, output_dim=1)
    
    # Training parameters
    num_epochs = 2000
    learning_rate = 1e-3
    print_interval = 100
    
    # Storage for training history
    train_losses = []
    test_losses = []
    
    print(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
    print(f"Training samples: {n_samples}, Test samples: {len(x_test)}")
    print()
    
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
            
            print(f"Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
            
            # Check LayerNorm parameters
            gamma1_norm = np.linalg.norm(model.layernorm1.gamma)
            beta1_norm = np.linalg.norm(model.layernorm1.beta)
            gamma2_norm = np.linalg.norm(model.layernorm2.gamma)
            beta2_norm = np.linalg.norm(model.layernorm2.beta)
            
            print(f"         LayerNorm1 - Gamma norm: {gamma1_norm:.4f}, Beta norm: {beta1_norm:.4f}")
            print(f"         LayerNorm2 - Gamma norm: {gamma2_norm:.4f}, Beta norm: {beta2_norm:.4f}")
            print()
    
    # Final evaluation
    y_train_final = model.forward(x_train)
    y_test_final = model.forward(x_test)
    
    final_train_loss = mse_loss(y_train, y_train_final)
    final_test_loss = mse_loss(y_test, y_test_final)
    
    print("=" * 50)
    print("Training completed!")
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss: {final_test_loss:.6f}")
    
    # Print some sample predictions
    print("\nSample predictions vs. true values:")
    test_indices = [0, 25, 50, 75, 99]
    for i in test_indices:
        x_val = x_test[i, 0]
        y_true = y_test[i, 0]
        y_pred = y_test_final[i, 0]
        print(f"x = {x_val:6.3f}: sin(x) = {y_true:6.3f}, pred = {y_pred:6.3f}, error = {abs(y_true - y_pred):6.3f}")
    
    # Test gradient computation specifically for LayerNorm
    print("\n" + "=" * 50)
    print("Testing LayerNorm gradient computation:")
    
    # Create small test case for gradient checking
    test_x = np.random.randn(5, 4)  # Small batch for easy verification
    test_layernorm = LayerNormalization(4)
    
    # Forward pass
    test_output = test_layernorm.forward(test_x)
    
    # Backward pass with dummy gradient
    dummy_grad = np.ones_like(test_output)
    dLdX, dLdGamma, dLdBeta = test_layernorm.backward(test_x, dummy_grad)
    
    print(f"Test input shape: {test_x.shape}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Gradient w.r.t. input shape: {dLdX.shape}")
    print(f"Gradient w.r.t. gamma shape: {dLdGamma.shape}")
    print(f"Gradient w.r.t. beta shape: {dLdBeta.shape}")
    
    # Check that gradients have reasonable magnitudes
    print(f"Input gradient norm: {np.linalg.norm(dLdX):.6f}")
    print(f"Gamma gradient norm: {np.linalg.norm(dLdGamma):.6f}")
    print(f"Beta gradient norm: {np.linalg.norm(dLdBeta):.6f}")
    
    return model, train_losses, test_losses, x_test, y_test, y_test_final

if __name__ == "__main__":
    # Run the test
    model, train_losses, test_losses, x_test, y_test, y_test_final = test_layernorm_gradients()
    
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
        plt.plot(x_test, y_test, 'b-', label='True sin(x)', linewidth=2)
        plt.plot(x_test, y_test_final, 'r--', label='MLP prediction', linewidth=2)
        plt.title('Function Approximation: f(x) = sin(x)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('img/layernorm_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nPlots saved as 'layernorm_test_results.png'")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    except Exception as e:
        print(f"\nError creating plots: {e}")
