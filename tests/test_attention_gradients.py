import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.attention import SingleHeadAttention

def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient of function f at point x
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + h
        fxh_pos = f(x)
        
        x[idx] = old_value - h
        fxh_neg = f(x)
        
        grad[idx] = (fxh_pos - fxh_neg) / (2 * h)
        x[idx] = old_value
        it.iternext()
    
    return grad

def test_attention_gradients():
    """
    Test attention gradients using numerical gradient checking
    """
    print("\n" + "="*60)
    print("🧠 ATTENTION GRADIENT TEST")
    print("="*60)
    
    # Set up test parameters
    batch_size, seq_len, hidden_dim = 2, 4, 8
    np.random.seed(42)
    print(f"📊 Test parameters:")
    print(f"   ├─ Batch size: {batch_size}")
    print(f"   ├─ Sequence length: {seq_len}")
    print(f"   └─ Hidden dimension: {hidden_dim}")
    
    # Create attention layer
    attention = SingleHeadAttention(hidden_dim)
    
    # Create test input
    x = np.random.randn(batch_size, seq_len, hidden_dim) * 0.1
    
    # Forward pass
    y = attention.forward(x)
    
    # Create upstream gradient
    dLdY = np.random.randn(*y.shape) * 0.1
    
    # Compute analytical gradients
    dLdX_analytical = attention.backward(dLdY)
    
    # Get the analytical weight gradients from the stored gradients
    dLdW_q = attention.w_q.gradients['dLdW']
    dLdW_k = attention.w_k.gradients['dLdW']
    dLdW_v = attention.w_v.gradients['dLdW']
    
    # Define loss function for numerical gradient checking
    def loss_fn_input(x_test):
        y_test = attention.forward(x_test)
        return np.sum(dLdY * y_test)
    
    def loss_fn_wq(w_test):
        original_w = attention.w_q.weights.copy()
        attention.w_q.weights = w_test
        y_test = attention.forward(x)
        loss = np.sum(dLdY * y_test)
        attention.w_q.weights = original_w
        return loss
    
    # Compute numerical gradients
    print(f"\n🔍 Computing numerical gradients...")
    dLdX_numerical = numerical_gradient(loss_fn_input, x)
    dLdW_q_numerical = numerical_gradient(loss_fn_wq, attention.w_q.weights)
    
    # Compare gradients
    print(f"\n📈 Gradient comparison:")
    print(f"   ├─ dL/dX max diff: {np.max(np.abs(dLdX_analytical - dLdX_numerical)):.6f}")
    print(f"   ├─ dL/dX rel error: {np.max(np.abs(dLdX_analytical - dLdX_numerical) / (np.abs(dLdX_numerical) + 1e-8)):.6f}")
    print(f"   ├─ dL/dW_q max diff: {np.max(np.abs(dLdW_q - dLdW_q_numerical)):.6f}")
    print(f"   └─ dL/dW_q rel error: {np.max(np.abs(dLdW_q - dLdW_q_numerical) / (np.abs(dLdW_q_numerical) + 1e-8)):.6f}")
    
    # Check if gradients are close
    tol = 1e-3
    input_grad_close = np.allclose(dLdX_analytical, dLdX_numerical, atol=tol, rtol=tol)
    weight_grad_close = np.allclose(dLdW_q, dLdW_q_numerical, atol=tol, rtol=tol)
    
    print(f"\n🎯 Gradient check (tolerance={tol}):")
    print(f"   ├─ Input gradients close: {input_grad_close}")
    print(f"   └─ Weight gradients close: {weight_grad_close}")
    
    if input_grad_close and weight_grad_close:
        print("\n✅ Attention backward pass implementation is CORRECT!")
    else:
        print("\n❌ Attention backward pass implementation has errors!")
        print(f"   ├─ Input grad shapes - Analytical: {dLdX_analytical.shape}, Numerical: {dLdX_numerical.shape}")
        print(f"   └─ Weight grad shapes - Analytical: {dLdW_q.shape}, Numerical: {dLdW_q_numerical.shape}")
    
    print("="*60 + "\n")
    return input_grad_close and weight_grad_close

if __name__ == "__main__":
    test_attention_gradients() 