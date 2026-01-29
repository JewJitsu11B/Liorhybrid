"""
Unit Tests for Comprehensive Similarity (15D Vector)

Tests the 9D core similarity computation including:
- Individual dimension calculations
- Batch computation
- Aggregation strategies
- Integration with manifold geometry

Phase 1: 9D core measures
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except:
    pass

import torch
import pytest
from ..utils.comprehensive_similarity import (
    ComprehensiveSimilarity,
    compute_cosine_similarity,
    compute_wedge_magnitude,
    compute_tensor_trace,
)
from ..models.manifold import CognitiveManifold


class TestHelperFunctions:
    """Test standalone helper functions."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        query = torch.tensor([1.0, 0.0, 0.0])
        candidates = torch.tensor([
            [1.0, 0.0, 0.0],  # Same direction: cosine = 1
            [0.0, 1.0, 0.0],  # Orthogonal: cosine = 0
            [-1.0, 0.0, 0.0], # Opposite: cosine = -1
        ])
        
        cosines = compute_cosine_similarity(query, candidates)
        
        assert cosines.shape == (3,)
        assert torch.allclose(cosines[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(cosines[1], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(cosines[2], torch.tensor(-1.0), atol=1e-5)
    
    def test_wedge_magnitude(self):
        """Test wedge product magnitude."""
        # Parallel vectors: wedge = 0
        query = torch.tensor([1.0, 0.0])
        candidate = torch.tensor([2.0, 0.0])
        wedge = compute_wedge_magnitude(query, candidate)
        assert torch.allclose(wedge, torch.tensor(0.0), atol=1e-5)
        
        # Orthogonal vectors: maximum wedge
        query = torch.tensor([1.0, 0.0])
        candidate = torch.tensor([0.0, 1.0])
        wedge = compute_wedge_magnitude(query, candidate)
        assert wedge > 0.0
    
    def test_tensor_trace(self):
        """Test tensor product trace (inner product)."""
        query = torch.tensor([1.0, 2.0, 3.0])
        candidates = torch.tensor([
            [1.0, 0.0, 0.0],  # Trace = 1
            [0.0, 1.0, 0.0],  # Trace = 2
            [0.0, 0.0, 1.0],  # Trace = 3
        ])
        
        traces = compute_tensor_trace(query, candidates)
        
        assert traces.shape == (3,)
        assert torch.allclose(traces[0], torch.tensor(1.0))
        assert torch.allclose(traces[1], torch.tensor(2.0))
        assert torch.allclose(traces[2], torch.tensor(3.0))


class TestComprehensiveSimilarity:
    """Test ComprehensiveSimilarity class."""
    
    @pytest.fixture
    def manifold(self):
        """Create a test manifold."""
        return CognitiveManifold(d_embed=64, d_coord=8, d_spinor=4)
    
    @pytest.fixture
    def similarity_computer(self, manifold):
        """Create a similarity computer."""
        return ComprehensiveSimilarity(manifold, mode='core')
    
    def test_initialization(self, manifold):
        """Test proper initialization."""
        sim = ComprehensiveSimilarity(manifold, mode='core')
        assert sim.mode == 'core'
        assert sim.manifold is manifold
        assert len(sim.context_cache) == 0
    
    def test_invalid_mode(self, manifold):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ComprehensiveSimilarity(manifold, mode='invalid')
    
    def test_compute_batch_shape(self, similarity_computer):
        """Test that compute_batch returns correct shape."""
        query_embedding = torch.randn(64)  # d_embed = 64
        candidate_embeddings = torch.randn(100, 64)  # N = 100
        
        vectors = similarity_computer.compute_batch(query_embedding, candidate_embeddings)
        
        assert vectors.shape == (100, 9), f"Expected (100, 9), got {vectors.shape}"
    
    def test_compute_batch_dimensions(self, similarity_computer):
        """Test that all 9 dimensions are computed."""
        query_embedding = torch.randn(64)
        candidate_embeddings = torch.randn(10, 64)
        
        vectors = similarity_computer.compute_batch(query_embedding, candidate_embeddings)
        
        # Check that no dimension is all zeros (all should be computed)
        for dim in range(9):
            # At least some values should be non-zero for most dimensions
            # (except possibly phase which can be zero)
            if dim != 4:  # Skip spinor_phase which might be zero
                assert not torch.allclose(vectors[:, dim], torch.zeros(10))
    
    def test_compute_batch_cosine_dimension(self, similarity_computer):
        """Test that dimension 0 is cosine similarity."""
        # Create query and candidates with known relationships
        # For embeddings, we need to ensure the projected coordinates have known cosine
        query_embedding = torch.randn(64)
        candidate_embeddings = torch.randn(2, 64)
        
        vectors = similarity_computer.compute_batch(query_embedding, candidate_embeddings)
        
        # Dimension 0 should be cosine (values between -1 and 1)
        assert -1.0 <= vectors[0, 0] <= 1.0
        assert -1.0 <= vectors[1, 0] <= 1.0
    
    def test_compute_batch_no_nans(self, similarity_computer):
        """Test that computation doesn't produce NaNs."""
        query_embedding = torch.randn(64)
        candidate_embeddings = torch.randn(50, 64)
        
        vectors = similarity_computer.compute_batch(query_embedding, candidate_embeddings)
        
        assert not torch.isnan(vectors).any(), "Similarity vectors contain NaN"
        assert not torch.isinf(vectors).any(), "Similarity vectors contain Inf"
    
    def test_aggregate_mean(self, similarity_computer):
        """Test mean aggregation strategy."""
        vectors = torch.randn(100, 9)
        scores = similarity_computer.aggregate_to_scalar(vectors, strategy='mean')
        
        assert scores.shape == (100,)
        # Mean should be close to average of dimensions
        expected = vectors.mean(dim=1)
        assert torch.allclose(scores, expected)
    
    def test_aggregate_lior_primary(self, similarity_computer):
        """Test LIoR-primary aggregation strategy."""
        vectors = torch.randn(100, 9)
        # Set known values for dimension 8 (LIoR distance)
        vectors[:, 8] = torch.arange(100, dtype=torch.float32)
        
        scores = similarity_computer.aggregate_to_scalar(vectors, strategy='lior_primary')
        
        assert scores.shape == (100,)
        # Should be negative of dimension 8
        expected = -torch.arange(100, dtype=torch.float32)
        assert torch.allclose(scores, expected)
    
    def test_aggregate_weighted(self, similarity_computer):
        """Test weighted aggregation strategy."""
        vectors = torch.randn(100, 9)
        scores = similarity_computer.aggregate_to_scalar(vectors, strategy='weighted')
        
        assert scores.shape == (100,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
    
    def test_aggregate_invalid_strategy(self, similarity_computer):
        """Test that invalid strategy raises error."""
        vectors = torch.randn(100, 9)
        
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            similarity_computer.aggregate_to_scalar(vectors, strategy='invalid')
    
    def test_clear_cache(self, similarity_computer):
        """Test cache clearing."""
        # Add something to cache
        similarity_computer.context_cache[123] = {'test': 'data'}
        assert len(similarity_computer.context_cache) > 0
        
        # Clear cache
        similarity_computer.clear_cache()
        assert len(similarity_computer.context_cache) == 0
    
    def test_extended_mode_not_implemented(self, manifold):
        """Test that extended mode raises NotImplementedError."""
        sim = ComprehensiveSimilarity(manifold, mode='extended')
        query_embedding = torch.randn(64)
        candidate_embeddings = torch.randn(10, 64)
        
        with pytest.raises(NotImplementedError, match="Extended.*not yet implemented"):
            sim.compute_batch(query_embedding, candidate_embeddings)
    
    def test_full_mode_not_implemented(self, manifold):
        """Test that full mode raises NotImplementedError."""
        sim = ComprehensiveSimilarity(manifold, mode='full')
        query_embedding = torch.randn(64)
        candidate_embeddings = torch.randn(10, 64)
        
        with pytest.raises(NotImplementedError, match="Full.*not yet implemented"):
            sim.compute_batch(query_embedding, candidate_embeddings)


class TestIntegrationWithManifold:
    """Test integration with CognitiveManifold."""
    
    def test_with_real_manifold(self):
        """Test with actual manifold instance."""
        # Create manifold
        manifold = CognitiveManifold(d_embed=128, d_coord=8, d_spinor=4)
        sim = ComprehensiveSimilarity(manifold, mode='core')
        
        # Create query and candidates (embeddings)
        query_embedding = torch.randn(128)
        candidate_embeddings = torch.randn(20, 128)
        
        # Compute similarity
        vectors = sim.compute_batch(query_embedding, candidate_embeddings)
        
        assert vectors.shape == (20, 9)
        assert not torch.isnan(vectors).any()
    
    def test_different_coord_dimensions(self):
        """Test with different coordinate dimensions."""
        for d_coord in [4, 8, 16]:
            d_embed = 64
            manifold = CognitiveManifold(d_embed=d_embed, d_coord=d_coord, d_spinor=4)
            sim = ComprehensiveSimilarity(manifold, mode='core')
            
            query_embedding = torch.randn(d_embed)
            candidate_embeddings = torch.randn(10, d_embed)
            
            vectors = sim.compute_batch(query_embedding, candidate_embeddings)
            
            assert vectors.shape == (10, 9)


class TestPerformance:
    """Test computational performance."""
    
    def test_batch_efficiency(self):
        """Test that batched computation is efficient."""
        manifold = CognitiveManifold(d_embed=512, d_coord=8, d_spinor=4)
        sim = ComprehensiveSimilarity(manifold, mode='core')
        
        query_embedding = torch.randn(512)
        candidate_embeddings = torch.randn(1000, 512)  # Large batch
        
        # Should complete without hanging
        vectors = sim.compute_batch(query_embedding, candidate_embeddings)
        
        assert vectors.shape == (1000, 9)
        assert not torch.isnan(vectors).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test that computation works on CUDA."""
        manifold = CognitiveManifold(d_embed=128, d_coord=8, d_spinor=4).cuda()
        sim = ComprehensiveSimilarity(manifold, mode='core')
        
        query_embedding = torch.randn(128, device='cuda')
        candidate_embeddings = torch.randn(100, 128, device='cuda')
        
        vectors = sim.compute_batch(query_embedding, candidate_embeddings)
        
        assert vectors.device.type == 'cuda'
        assert vectors.shape == (100, 9)
        assert not torch.isnan(vectors).any()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_candidate(self):
        """Test with single candidate."""
        manifold = CognitiveManifold(d_embed=64, d_coord=8, d_spinor=4)
        sim = ComprehensiveSimilarity(manifold, mode='core')
        
        query_embedding = torch.randn(64)
        candidate_embeddings = torch.randn(1, 64)
        
        vectors = sim.compute_batch(query_embedding, candidate_embeddings)
        
        assert vectors.shape == (1, 9)
    
    def test_identical_query_candidate(self):
        """Test when query equals candidate."""
        manifold = CognitiveManifold(d_embed=64, d_coord=8, d_spinor=4)
        sim = ComprehensiveSimilarity(manifold, mode='core')
        
        query_embedding = torch.randn(64)
        candidate_embeddings = query_embedding.unsqueeze(0)  # Same as query
        
        vectors = sim.compute_batch(query_embedding, candidate_embeddings)
        
        # Cosine should be 1.0
        assert torch.allclose(vectors[0, 0], torch.tensor(1.0), atol=1e-4)
        # Wedge should be 0 (no rotation with self)
        assert torch.allclose(vectors[0, 1], torch.tensor(0.0), atol=1e-4)
    
    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        manifold = CognitiveManifold(d_embed=64, d_coord=8, d_spinor=4)
        sim = ComprehensiveSimilarity(manifold, mode='core')
        
        query_embedding = torch.zeros(64)
        candidate_embeddings = torch.randn(10, 64)
        
        vectors = sim.compute_batch(query_embedding, candidate_embeddings)
        
        # Should not produce NaN even with zero query
        assert not torch.isnan(vectors).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
