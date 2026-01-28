"""
Standalone test for address builder (no pytest dependencies)
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import address module
exec(open('inference/address.py').read())

print("=" * 60)
print("Testing Address Builder Implementation")
print("=" * 60)

# Test 1: Config dimensions
print("\n1. Testing AddressConfig dimensions...")
config = AddressConfig(d=512)
print(f"   n_neighbors: {config.n_neighbors} (expected: 64)")
print(f"   m (scores): {config.m} (expected: 6)")
print(f"   d_block: {config.d_block}")
print(f"   total_dim: {config.total_dim}")
assert config.n_neighbors == 64, "Should have 64 neighbors"
assert config.m == 6, "Should have 6 similarity scores"
print("   ✓ Config dimensions correct")

# Test 2: AddressBuilder shape
print("\n2. Testing AddressBuilder shape...")
config = AddressConfig(d=128)
builder = AddressBuilder(config)
batch_size = 4
embedding = torch.randn(batch_size, 128)
addr = builder(embedding)
print(f"   Address shape: {addr.data.shape}")
print(f"   Expected: ({batch_size}, {config.total_dim})")
assert addr.data.shape == (batch_size, config.total_dim)
print("   ✓ Address shape correct")

# Test 3: 64 neighbors populated
print("\n3. Testing 64 neighbors populated...")
config = AddressConfig(d=128, enable_address_probing=True)
builder = AddressBuilder(config)
batch_size = 2
embedding = torch.randn(batch_size, 128)
neighbor_embeddings = torch.randn(batch_size, 100, 128)
addr = builder(embedding, neighbor_embeddings=neighbor_embeddings)
neighbors_blocked = addr.neighbors_blocked
print(f"   Neighbors shape: {neighbors_blocked.shape}")
print(f"   Expected: ({batch_size}, 64, {config.d_block})")
assert neighbors_blocked.shape == (batch_size, 64, config.d_block)
assert neighbors_blocked.abs().sum() > 0, "Neighbors should be populated"
print("   ✓ 64 neighbors populated")

# Test 4: 6 similarity scores per neighbor
print("\n4. Testing 6 similarity scores...")
scores = addr.all_neighbor_scores
print(f"   Scores shape: {scores.shape}")
print(f"   Expected: ({batch_size}, 64, 6)")
assert scores.shape == (batch_size, 64, 6)
assert scores.abs().sum() > 0, "Scores should be populated"
print("   ✓ 6 similarity scores per neighbor")

# Test 5: Role-typed partitions
print("\n5. Testing role-typed partitions...")
nearest = addr.nearest_neighbors
attractors = addr.attractor_neighbors
repulsors = addr.repulsor_neighbors
print(f"   Nearest shape: {nearest.shape} (expected: ({batch_size}, 32, {config.d_block}))")
print(f"   Attractors shape: {attractors.shape} (expected: ({batch_size}, 16, {config.d_block}))")
print(f"   Repulsors shape: {repulsors.shape} (expected: ({batch_size}, 16, {config.d_block}))")
assert nearest.shape == (batch_size, 32, config.d_block)
assert attractors.shape == (batch_size, 16, config.d_block)
assert repulsors.shape == (batch_size, 16, config.d_block)
print("   ✓ Role-typed partitions correct")

# Test 6: Metric and transport per neighbor
print("\n6. Testing metric/transport per neighbor...")
neighbor_metrics = addr.all_neighbor_metrics
neighbor_transports = addr.all_neighbor_transports
print(f"   Neighbor metrics shape: {neighbor_metrics.shape}")
print(f"   Neighbor transports shape: {neighbor_transports.shape}")
assert neighbor_metrics.shape == (batch_size, 64, config.d_neighbor_metric)
assert neighbor_transports.shape == (batch_size, 64, config.d_neighbor_transport)
assert neighbor_metrics.abs().sum() > 0
assert neighbor_transports.abs().sum() > 0
print("   ✓ Metric/transport per neighbor populated")

# Test 7: ECC and timestamps
print("\n7. Testing ECC and timestamps...")
ecc = addr.ecc
timestamps = addr.timestamps
print(f"   ECC shape: {ecc.shape} (expected: ({batch_size}, 32))")
print(f"   Timestamps shape: {timestamps.shape} (expected: ({batch_size}, 2))")
assert ecc.shape == (batch_size, 32)
assert timestamps.shape == (batch_size, 2)
assert ecc.abs().sum() > 0, "ECC should be populated"
assert (timestamps > 0).all(), "Timestamps should be positive"
print("   ✓ ECC and timestamps present")

# Test 8: Collision checking
print("\n8. Testing collision checking...")
batch_size = 10
embedding = torch.randn(batch_size, 128)
addr = builder(embedding)
n_collisions, collision_matrix = check_address_collisions(addr, threshold=0.99)
print(f"   Collisions detected: {n_collisions}")
print(f"   Collision matrix shape: {collision_matrix.shape}")
assert n_collisions >= 0
assert collision_matrix.shape == (batch_size, batch_size)
print("   ✓ Collision checking works")

# Test 9: Uniqueness score
print("\n9. Testing uniqueness score...")
uniqueness = compute_address_uniqueness_score(addr)
print(f"   Uniqueness score: {uniqueness:.4f} (0.0-1.0, higher is better)")
assert 0.0 <= uniqueness <= 1.0
print("   ✓ Uniqueness score computed")

# Test 10: Individual neighbor access
print("\n10. Testing individual neighbor access...")
for i in [0, 31, 32, 47, 48, 63]:
    value = addr.neighbor_value(i)
    metric = addr.neighbor_metric(i)
    transport = addr.neighbor_transport(i)
    scores = addr.neighbor_scores(i)
    coords = addr.neighbor_coords(i)
    print(f"   Neighbor {i}: value={value.shape}, metric={metric.shape}, transport={transport.shape}, scores={scores.shape}, coords={coords.shape}")
print("   ✓ Individual neighbor access works")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
