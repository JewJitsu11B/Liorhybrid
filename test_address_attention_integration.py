"""
Integration test for Address-based geometric attention (Option 6 Extended)
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn

# Import address and attention modules
exec(open('inference/address.py').read())
# For attention, we'll just test the integration concept

print("=" * 60)
print("Testing Address-Based Attention Integration")
print("=" * 60)

# Test 1: Build addresses with AddressBuilder
print("\n1. Building addresses with neighbors...")
config = AddressConfig(d=128, enable_address_probing=True)
builder = AddressBuilder(config)

batch_size = 2
embedding = torch.randn(batch_size, 128)
neighbor_embeddings = torch.randn(batch_size, 100, 128)

# Build query addresses
Q_addresses = builder(embedding, neighbor_embeddings=neighbor_embeddings)
print(f"   Q_addresses built: shape={Q_addresses.shape}, total_dim={config.total_dim}")

# Test 2: Verify neighbor structure
print("\n2. Verifying neighbor structure...")
neighbor_scores = Q_addresses.all_neighbor_scores
neighbor_values = Q_addresses.all_neighbor_values
neighbor_metrics = Q_addresses.all_neighbor_metrics
neighbor_transports = Q_addresses.all_neighbor_transports

print(f"   Neighbor scores: {neighbor_scores.shape} (expected: ({batch_size}, 64, 6))")
print(f"   Neighbor values: {neighbor_values.shape} (expected: ({batch_size}, 64, 64))")
print(f"   Neighbor metrics: {neighbor_metrics.shape} (expected: ({batch_size}, 64, 16))")
print(f"   Neighbor transports: {neighbor_transports.shape} (expected: ({batch_size}, 64, 16))")

assert neighbor_scores.shape == (batch_size, 64, 6)
assert neighbor_values.shape == (batch_size, 64, 64)
assert neighbor_metrics.shape == (batch_size, 64, 16)
assert neighbor_transports.shape == (batch_size, 64, 16)
print("   ✓ All neighbor components present")

# Test 3: Verify role-typed access
print("\n3. Verifying role-typed neighbor access...")
nearest = Q_addresses.nearest_neighbors
attractors = Q_addresses.attractor_neighbors
repulsors = Q_addresses.repulsor_neighbors

print(f"   Nearest: {nearest.shape} (32 neighbors)")
print(f"   Attractors: {attractors.shape} (16 neighbors)")
print(f"   Repulsors: {repulsors.shape} (16 neighbors)")

assert nearest.shape[1] == 32
assert attractors.shape[1] == 16
assert repulsors.shape[1] == 16
print("   ✓ Role-typed partitions correct")

# Test 4: Simulate attention weighting using scores
print("\n4. Simulating attention with neighbor scores...")

# Extract primary scores (cosine similarity)
primary_scores = neighbor_scores[..., 0]  # (batch, 64)
learned_scores = neighbor_scores[..., 1:].mean(dim=-1)  # (batch, 64)
combined_scores = primary_scores + 0.1 * learned_scores

print(f"   Combined scores shape: {combined_scores.shape}")
print(f"   Score range: [{combined_scores.min():.3f}, {combined_scores.max():.3f}]")

# Apply role-typed weighting
role_weights = torch.ones(64)
role_weights[32:48] = 1.5   # Attractors
role_weights[48:64] = -0.5  # Repulsors

weighted_scores = combined_scores * role_weights.unsqueeze(0)
print(f"   Weighted scores shape: {weighted_scores.shape}")

# Compute attention weights (softmax)
attention_weights = torch.softmax(weighted_scores, dim=-1)
print(f"   Attention weights shape: {attention_weights.shape}")
print(f"   Attention weights sum: {attention_weights.sum(dim=-1)}")  # Should be ~1.0 per batch

assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
print("   ✓ Attention weights normalized")

# Test 5: Weighted neighbor value aggregation
print("\n5. Testing weighted aggregation of neighbor values...")

# Aggregate neighbor values using attention weights
# attention_weights: (batch, 64)
# neighbor_values: (batch, 64, 64)
aggregated = torch.bmm(
    attention_weights.unsqueeze(1),  # (batch, 1, 64)
    neighbor_values                   # (batch, 64, 64)
).squeeze(1)  # (batch, 64)

print(f"   Aggregated output shape: {aggregated.shape}")
print(f"   Output range: [{aggregated.min():.3f}, {aggregated.max():.3f}]")
assert aggregated.shape == (batch_size, 64)
print("   ✓ Weighted aggregation successful")

# Test 6: Multi-token batch
print("\n6. Testing multi-token scenario...")
batch_size = 4
seq_len = 8
embeddings = torch.randn(batch_size * seq_len, 128)
neighbor_embeddings_all = torch.randn(batch_size * seq_len, 100, 128)

Q_addresses_multi = builder(embeddings, neighbor_embeddings=neighbor_embeddings_all)
print(f"   Multi-token addresses: shape={Q_addresses_multi.shape}")
print(f"   Total tokens: {batch_size * seq_len}")

# Reshape for per-token attention
neighbor_scores_multi = Q_addresses_multi.all_neighbor_scores
print(f"   Neighbor scores for all tokens: {neighbor_scores_multi.shape}")
assert neighbor_scores_multi.shape == (batch_size * seq_len, 64, 6)
print("   ✓ Multi-token probing works")

print("\n" + "=" * 60)
print("Address-Based Attention Integration Tests Passed! ✓")
print("=" * 60)
print("\nIntegration Summary:")
print("- AddressBuilder creates addresses with 64 role-typed neighbors")
print("- Each neighbor has 6 similarity scores (cosine + 5 learned)")
print("- Neighbor values, metrics, and transports are accessible")
print("- Attention weights can be computed from similarity scores")
print("- Weighted aggregation produces attended output")
print("- Ready for full integration with GeometricAttention.forward()")
