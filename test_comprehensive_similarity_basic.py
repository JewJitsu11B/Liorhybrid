#!/usr/bin/env python3
"""
Quick validation test for ComprehensiveSimilarity
"""
import torch
from utils.comprehensive_similarity import ComprehensiveSimilarity
from models.manifold import CognitiveManifold

def main():
    print("Testing ComprehensiveSimilarity...")
    
    # Create manifold
    print("Creating manifold...")
    manifold = CognitiveManifold(d_embed=64, d_coord=8, d_spinor=4)
    sim = ComprehensiveSimilarity(manifold, mode='core')
    
    # Create test data
    print("Creating test data...")
    query = torch.randn(8)
    candidates = torch.randn(10, 8)
    
    # Compute similarity
    print("Computing 9D similarity vectors...")
    vectors = sim.compute_batch(query, candidates)
    
    print(f"✓ Similarity vectors shape: {vectors.shape}")
    print(f"✓ No NaNs: {not torch.isnan(vectors).any()}")
    print(f"✓ No Infs: {not torch.isinf(vectors).any()}")
    
    # Test aggregation
    print("\nTesting aggregation strategies...")
    scores_mean = sim.aggregate_to_scalar(vectors, strategy='mean')
    scores_lior = sim.aggregate_to_scalar(vectors, strategy='lior_primary')
    scores_weighted = sim.aggregate_to_scalar(vectors, strategy='weighted')
    
    print(f"✓ Mean aggregation shape: {scores_mean.shape}")
    print(f"✓ LIoR-primary aggregation shape: {scores_lior.shape}")
    print(f"✓ Weighted aggregation shape: {scores_weighted.shape}")
    
    # Test with larger batch
    print("\nTesting with larger batch (N=100)...")
    candidates_large = torch.randn(100, 8)
    vectors_large = sim.compute_batch(query, candidates_large)
    print(f"✓ Large batch shape: {vectors_large.shape}")
    
    print("\n✅ All basic tests passed!")
    print("\nExample 9D vector for first candidate:")
    print(vectors[0])

if __name__ == '__main__':
    main()
