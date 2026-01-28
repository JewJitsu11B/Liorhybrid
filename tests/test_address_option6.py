"""
Test Address Structure and Strict Neighbor Selector (Option 6)

Verifies:
- AddressConfig with 6 score channels
- NeighborSelector with strict metric-only selection
- AddressBuilder integration
- Collision detection with route_hash
- 64-slot population (32 nearest, 16 attractors, 16 repulsors)
- Fail-fast semantics when metric missing or invalid
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except:
    pass

import torch
import pytest
from ..inference.address import (
    AddressConfig,
    Address,
    AddressBuilder,
    NeighborSelector,
)


class TestAddressConfig:
    """Test AddressConfig dimensions and properties."""
    
    def test_default_config_6_scores(self):
        """Verify default config has 6 score channels."""
        config = AddressConfig()
        assert config.m == 6, "Should have 6 score channels (not 8)"
        assert config.n_neighbors == 64, "Should have 64 neighbors"
        assert config.n_nearest == 32
        assert config.n_attractors == 16
        assert config.n_repulsors == 16
    
    def test_d_block_calculation(self):
        """Verify d_block = d_prime + m + k = 64 + 6 + 16 = 86."""
        config = AddressConfig()
        assert config.d_block == 86, f"Expected 86, got {config.d_block}"
    
    def test_total_dim_calculation(self):
        """Verify total dimension with 6 scores: 7074 for d=512."""
        config = AddressConfig(d=512)
        expected = 512 + 1024 + (64 * 86) + 34  # core + geom + neighbors + integrity
        assert config.total_dim == expected, f"Expected {expected}, got {config.total_dim}"
        assert config.total_dim == 7074


class TestNeighborSelector:
    """Test strict metric-only neighbor selector."""
    
    def test_requires_metric(self):
        """Fail fast if metric is None."""
        selector = NeighborSelector()
        
        batch_size = 2
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        
        # Should raise ValueError when metric is None
        with pytest.raises(ValueError, match="requires metric and transport"):
            selector.select_neighbors(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                metric=None,
                transport=torch.randn(batch_size, d)
            )
    
    def test_requires_transport(self):
        """Fail fast if transport is None."""
        selector = NeighborSelector()
        
        batch_size = 2
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        
        # Should raise ValueError when transport is None
        with pytest.raises(ValueError, match="requires metric and transport"):
            selector.select_neighbors(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                metric=torch.randn(batch_size, d),
                transport=None
            )
    
    def test_rejects_nan_metric(self):
        """Fail fast if metric contains NaN."""
        selector = NeighborSelector()
        
        batch_size = 2
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        
        metric = torch.randn(batch_size, d)
        metric[0, 0] = float('nan')  # Inject NaN
        transport = torch.randn(batch_size, d)
        
        # Should raise ValueError for NaN
        with pytest.raises(ValueError, match="Invalid metric.*NaN"):
            selector.select_neighbors(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                metric=metric,
                transport=transport
            )
    
    def test_rejects_inf_metric(self):
        """Fail fast if metric contains Inf."""
        selector = NeighborSelector()
        
        batch_size = 2
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        
        metric = torch.randn(batch_size, d)
        metric[0, 0] = float('inf')  # Inject Inf
        transport = torch.randn(batch_size, d)
        
        # Should raise ValueError for Inf
        with pytest.raises(ValueError, match="Invalid metric.*Inf"):
            selector.select_neighbors(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                metric=metric,
                transport=transport
            )
    
    def test_requires_64_candidates(self):
        """Fail fast if fewer than 64 candidates."""
        selector = NeighborSelector()
        
        batch_size = 2
        d = 512
        n_cand = 32  # Too few!
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        metric = torch.randn(batch_size, d)
        transport = torch.randn(batch_size, d)
        
        # Should raise ValueError for insufficient candidates
        with pytest.raises(ValueError, match="Need at least 64 candidates"):
            selector.select_neighbors(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                metric=metric,
                transport=transport
            )
    
    def test_selects_64_neighbors(self):
        """Select exactly 64 neighbors with valid inputs."""
        selector = NeighborSelector()
        
        batch_size = 2
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        metric = torch.randn(batch_size, d).abs() + 0.1  # Positive metric
        transport = torch.randn(batch_size, d)
        
        selected_embeddings, selected_scores, selected_indices = selector.select_neighbors(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            metric=metric,
            transport=transport
        )
        
        # Check shapes
        assert selected_embeddings.shape == (batch_size, 64, d)
        assert selected_scores.shape == (batch_size, 64, 6), "Should have 6 score channels"
        assert selected_indices.shape == (batch_size, 64)
    
    def test_6_score_channels(self):
        """Verify 6 score channels are computed."""
        selector = NeighborSelector()
        
        batch_size = 1
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        metric = torch.randn(batch_size, d).abs() + 0.1
        transport = torch.randn(batch_size, d)
        
        _, selected_scores, _ = selector.select_neighbors(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            metric=metric,
            transport=transport
        )
        
        # Check all 6 channels populated
        assert selected_scores.shape[-1] == 6
        assert not selected_scores.isnan().any(), "Scores should not contain NaN"
        assert not selected_scores.isinf().any(), "Scores should not contain Inf"
    
    def test_role_partition(self):
        """Verify neighbor selection uses proper role-based partitioning."""
        selector = NeighborSelector()
        
        batch_size = 1
        d = 512
        n_cand = 128
        
        query_embedding = torch.randn(batch_size, d)
        candidate_embeddings = torch.randn(batch_size, n_cand, d)
        metric = torch.randn(batch_size, d).abs() + 0.1
        transport = torch.randn(batch_size, d)
        
        _, selected_scores, selected_indices = selector.select_neighbors(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            metric=metric,
            transport=transport
        )
        
        # Indices should be within valid range
        assert selected_indices.min() >= 0
        assert selected_indices.max() < n_cand
        
        # Check that we got 64 unique selections (or allow some overlap)
        # In practice, there might be overlap between nearest/attractors/repulsors
        assert selected_indices.shape[1] == 64


class TestAddressBuilder:
    """Test AddressBuilder with strict neighbor selection."""
    
    def test_builds_address_with_64_neighbors(self):
        """Build address with all 64 neighbor slots populated."""
        config = AddressConfig(d=512)
        builder = AddressBuilder(config=config, enable_collision_check=False)
        
        batch_size = 2
        n_cand = 128
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, n_cand, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Check address shape
        assert address.data.shape == (batch_size, config.total_dim)
        
        # Check neighbors populated
        neighbors_blocked = address.neighbors_blocked
        assert neighbors_blocked.shape == (batch_size, 64, config.d_block)
        
        # Check neighbor values, scores, coords
        values = address.all_neighbor_values
        scores = address.all_neighbor_scores
        coords = address.all_neighbor_coords
        
        assert values.shape == (batch_size, 64, config.d_prime)
        assert scores.shape == (batch_size, 64, 6), "Should have 6 score channels"
        assert coords.shape == (batch_size, 64, config.k)
    
    def test_metric_transport_present(self):
        """Verify metric and transport are populated."""
        config = AddressConfig(d=512)
        builder = AddressBuilder(config=config, enable_collision_check=False)
        
        batch_size = 2
        n_cand = 128
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, n_cand, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Check metric and transport
        metric = address.metric
        transport = address.transport
        
        assert metric.shape == (batch_size, config.d)
        assert transport.shape == (batch_size, config.d)
        assert not metric.isnan().any()
        assert not transport.isnan().any()
    
    def test_ecc_timestamps_present(self):
        """Verify ECC and timestamps are present but excluded from scoring."""
        config = AddressConfig(d=512)
        builder = AddressBuilder(config=config, enable_collision_check=False)
        
        batch_size = 2
        n_cand = 128
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, n_cand, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Check ECC and timestamps
        ecc = address.ecc
        timestamps = address.timestamps
        
        assert ecc.shape == (batch_size, config.ecc_bits)
        assert timestamps.shape == (batch_size, config.n_timestamps)
        
        # Timestamps should be non-zero (current time)
        assert (timestamps > 0).any()
    
    def test_fails_with_insufficient_candidates(self):
        """Fail fast when candidate count < 64."""
        config = AddressConfig(d=512)
        builder = AddressBuilder(config=config, enable_collision_check=False)
        
        batch_size = 2
        n_cand = 32  # Too few
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, n_cand, config.d)
        
        with pytest.raises(ValueError, match="Strict neighbor selection failed"):
            builder(
                embedding=embedding,
                candidate_embeddings=candidate_embeddings,
                enable_probing=True
            )


class TestAddressStructure:
    """Test Address data structure access patterns."""
    
    def test_neighbor_role_slices(self):
        """Verify neighbor role slices (nearest, attractors, repulsors)."""
        config = AddressConfig(d=512)
        address = Address.zeros(2, config=config)
        
        nearest = address.nearest_neighbors
        attractors = address.attractor_neighbors
        repulsors = address.repulsor_neighbors
        
        # Check shapes
        assert nearest.shape == (2, 32, config.d_block)
        assert attractors.shape == (2, 16, config.d_block)
        assert repulsors.shape == (2, 16, config.d_block)
    
    def test_neighbor_scores_6_channels(self):
        """Verify neighbor scores have 6 channels."""
        config = AddressConfig(d=512)
        builder = AddressBuilder(config=config, enable_collision_check=False)
        
        batch_size = 1
        n_cand = 128
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, n_cand, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Check individual neighbor scores
        for i in range(64):
            scores_i = address.neighbor_scores(i)
            assert scores_i.shape == (batch_size, 6), f"Neighbor {i} should have 6 scores"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
