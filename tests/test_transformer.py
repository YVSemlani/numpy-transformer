import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformer import Transformer

# hyperparameters
embedding_dim = 512
num_layers = 6
vocab_size = 100

# dummy input to verify the forward pass
x = np.random.randint(0, 100, (10, 10))
output_tokens = np.random.randint(0, 100, (10, 10))

print("="*100)
print("🚀 TRANSFORMER TRAINING VALIDATION")
print("="*100)
print(f"📊 Hyperparameters:")
print(f"   ├─ Embedding dim: {embedding_dim}")
print(f"   ├─ Num layers: {num_layers}")
print(f"   ├─ Vocab size: {vocab_size}")
print(f"   ├─ Input shape: {x.shape}")
print(f"   └─ Output tokens: {output_tokens.shape}")

dummy_transformer = Transformer(embedding_dim=embedding_dim, num_layers=num_layers, vocab_size=vocab_size)
prediction = dummy_transformer(x, output_tokens)

print("\n" + "="*80)
print("✅ FORWARD PASS VALIDATION")
print("="*80)

print(f"📤 Final prediction shape: {prediction.shape}")
expected_shape = (10, 10, vocab_size)
if prediction.shape == expected_shape:
    print(f"✅ Shape correct! Expected {expected_shape}, got {prediction.shape}")
else:
    print(f"❌ Shape incorrect! Expected {expected_shape}, got {prediction.shape}")

# dummy gradient to verify the backward pass
dummy_gradient = np.random.randn(10, 10, vocab_size)

print(f"\n📥 Testing backward pass with gradient: {dummy_gradient.shape}")
dLdY = dummy_transformer.backward(dummy_gradient)

print("\n" + "="*80)
print("⬅️  BACKWARD PASS VALIDATION")
print("="*80)

print(f"📤 Final gradient shape: {dLdY.shape}")
expected_grad_shape = (10, 10, embedding_dim)
if dLdY.shape == expected_grad_shape:
    print(f"✅ Gradient shape correct! Expected {expected_grad_shape}, got {dLdY.shape}")
else:
    print(f"❌ Gradient shape incorrect! Expected {expected_grad_shape}, got {dLdY.shape}")

print("\n" + "="*80)
print("🔄 PARAMETER UPDATE VALIDATION")
print("="*80)

# verify the update
def collect_all_weights(obj, weights_dict, metadata_dict, path="root"):
    """Recursively collect all weights from a transformer object with metadata"""
    # Check if this object has weights directly
    if hasattr(obj, 'weights'):
        key = id(obj)
        weights_dict[key] = obj.weights.copy()
        metadata_dict[key] = {
            'class': obj.__class__.__name__,
            'path': path,
            'attr': 'weights',
            'shape': obj.weights.shape
        }
    
    # Check for embedding matrix (named differently)
    if hasattr(obj, 'embedding_matrix'):
        key = id(obj)
        weights_dict[key] = obj.embedding_matrix.copy()
        metadata_dict[key] = {
            'class': obj.__class__.__name__,
            'path': path,
            'attr': 'embedding_matrix',
            'shape': obj.embedding_matrix.shape
        }
    
    # Check for bias (some layers might have separate bias)
    if hasattr(obj, 'bias'):
        key = f"{id(obj)}_bias"
        weights_dict[key] = obj.bias.copy()
        metadata_dict[key] = {
            'class': obj.__class__.__name__,
            'path': path,
            'attr': 'bias',
            'shape': obj.bias.shape
        }
    
    # Recursively check layers attribute
    if hasattr(obj, 'layers'):
        for i, layer in enumerate(obj.layers):
            collect_all_weights(layer, weights_dict, metadata_dict, f"{path}.layers[{i}]")
    
    # Check for specific named attributes that might contain weights
    for attr_name in ['encoder', 'decoder', 'output_projection', 'embedding', 'positional_encoding',
                      'attention', 'feed_forward', 'norm', 'masked_attention', 'cross_attention',
                      'w_q', 'w_k', 'w_v', 'linear1', 'linear2', 'softmax']:
        if hasattr(obj, attr_name):
            attr_obj = getattr(obj, attr_name)
            collect_all_weights(attr_obj, weights_dict, metadata_dict, f"{path}.{attr_name}")

# Store weights before update
print("🔍 Collecting weights before update...")
weights_before = {}
metadata = {}
collect_all_weights(dummy_transformer, weights_before, metadata)
print(f"📊 Found {len(weights_before)} parameter groups")

print("\n🔄 Performing parameter update...")
dummy_transformer.update()

# Store weights after update
weights_after = {}
metadata_after = {}  # We don't really need this, but keeping for consistency
collect_all_weights(dummy_transformer, weights_after, metadata_after)

# Check if ALL weights changed
print("\n📈 Analyzing parameter changes...")
all_weights_changed = True
any_weights_changed = False
unchanged_count = 0
unchanged_layers = []
changed_layers = []

for layer_id in weights_before:
    max_diff = np.max(np.abs(weights_before[layer_id] - weights_after[layer_id]))
    if not np.array_equal(weights_before[layer_id], weights_after[layer_id]):
        any_weights_changed = True
        changed_layers.append((layer_id, max_diff))
    else:
        all_weights_changed = False
        unchanged_count += 1
        unchanged_layers.append(layer_id)

print(f"\n📋 Update Summary:")
print(f"   ├─ Total parameter groups: {len(weights_before)}")
print(f"   ├─ Changed: {len(changed_layers)}")
print(f"   └─ Unchanged: {unchanged_count}")

if changed_layers:
    print(f"\n✅ Changed parameters:")
    for layer_id, max_diff in changed_layers[:5]:  # Show first 5
        meta = metadata[layer_id]
        print(f"   ├─ {meta['class']}.{meta['attr']} (max Δ: {max_diff:.6f})")
    if len(changed_layers) > 5:
        print(f"   └─ ... and {len(changed_layers) - 5} more")

if unchanged_layers:
    print(f"\n❌ Unchanged parameters:")
    for layer_id in unchanged_layers[:5]:  # Show first 5
        meta = metadata[layer_id]
        print(f"   ├─ {meta['class']}.{meta['attr']} at {meta['path']}")
    if len(unchanged_layers) > 5:
        print(f"   └─ ... and {len(unchanged_layers) - 5} more")

print(f"\n🎯 Final Status:")
if all_weights_changed:
    print("✅ ALL parameters were successfully updated!")
elif any_weights_changed:
    print("⚠️  SOME parameters were updated, but not all")
else:
    print("❌ NO parameters changed during update")

print("="*100)