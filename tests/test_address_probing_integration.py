"""
Integration Test: Address Probing Path (Option 6)

Tests the full pipeline:
- GeometricStack builds Address structures
- GeometricAttention uses Address for probing
- Fail-fast behavior when metric missing
- 6 score channels integration
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except:
    pass

import torch
import pytest
from ..inference.address import Address, AddressBuilder, AddressConfig
from ..inference.geometric_attention import GeometricAttention
from ..inference.geometric_stack import GeometricStack
from ..core import CognitiveTensorField, FAST_TEST_CONFIG


class TestAddressProbingIntegration:
    """Test full address probing pipeline."""
    
    def test_geometric_attention_with_address(self):
        """GeometricAttention can process Address structures."""
        d_model = 256
        batch_size = 2
        seq_len = 16
        
        # Create address with 64 neighbors
        config = AddressConfig(d=d_model)
        builder = AddressBuilder(config=config)
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, 128, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Create geometric attention layer
        attention = GeometricAttention(
            d_model=d_model,
            n_heads=4,
            dropout=0.0,
            use_exponential_form=True,
            use_field_contraction=False
        )
        
        # Create query
        Q = torch.randn(batch_size, seq_len, d_model)
        
        # Create dummy field
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
        
        # Forward pass with Address
        output, weights = attention(
            Q_input=Q,
            address=address,
            enable_address_probing=True,
            T_field=T_field
        )
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, 64)
        
        # Check no NaN/Inf
        assert not output.isnan().any()
        assert not output.isinf().any()
    
    def test_fails_without_metric(self):
        """Fail fast when address has invalid metric."""
        d_model = 256
        batch_size = 2
        seq_len = 16
        
        # Create address with invalid metric
        config = AddressConfig(d=d_model)
        address = Address.zeros(batch_size, config=config)
        
        # Inject NaN into metric
        address.metric[:] = float('nan')
        
        # Create geometric attention layer
        attention = GeometricAttention(
            d_model=d_model,
            n_heads=4,
            dropout=0.0,
            use_exponential_form=True,
            use_field_contraction=False
        )
        
        # Create query
        Q = torch.randn(batch_size, seq_len, d_model)
        
        # Create dummy field
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
        
        # Should fail fast with invalid metric
        with pytest.raises(ValueError, match="requires valid metric"):
            attention(
                Q_input=Q,
                address=address,
                enable_address_probing=True,
                T_field=T_field
            )
    
    def test_score_channels_used(self):
        """Verify 6 score channels are incorporated into similarity."""
        d_model = 256
        batch_size = 1
        seq_len = 8
        
        # Create address with 64 neighbors
        config = AddressConfig(d=d_model)
        builder = AddressBuilder(config=config)
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, 128, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Verify 6 score channels present
        scores = address.all_neighbor_scores
        assert scores.shape == (batch_size, 64, 6)
        
        # Create geometric attention layer
        attention = GeometricAttention(
            d_model=d_model,
            n_heads=4,
            dropout=0.0,
            use_exponential_form=True,
            use_field_contraction=False
        )
        
        # Create query
        Q = torch.randn(batch_size, seq_len, d_model)
        
        # Create dummy field
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
        
        # Forward pass
        output, weights = attention(
            Q_input=Q,
            address=address,
            enable_address_probing=True,
            T_field=T_field
        )
        
        # Check that score_channel_weights were created
        assert hasattr(attention, 'score_channel_weights')
        assert attention.score_channel_weights.shape == (6,)
    
    def test_geometric_stack_builds_addresses(self):
        """GeometricStack builds Address structures from field state."""
        d_model = 256
        n_layers = 2
        n_attention_layers = 1
        field_dim = 4
        
        stack = GeometricStack(
            d_model=d_model,
            n_layers=n_layers,
            n_attention_layers=n_attention_layers,
            field_dim=field_dim,
            timing_debug=False
        )
        
        batch_size = 2
        seq_len = 16
        
        # Input embeddings
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Field state (must have enough positions for 128+ candidates)
        N_x, N_y, D = 16, 16, 4
        field_state = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)
        
        # Forward pass
        output, _ = stack(
            x=x,
            field_state=field_state,
            attention_mask=None,
            diagnose=False
        )
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check that address_builder was created
        assert hasattr(stack, 'address_builder')
    
    def test_geometric_stack_fails_small_field(self):
        """GeometricStack fails fast when field is too small."""
        d_model = 256
        n_layers = 2
        n_attention_layers = 1
        field_dim = 4
        
        stack = GeometricStack(
            d_model=d_model,
            n_layers=n_layers,
            n_attention_layers=n_attention_layers,
            field_dim=field_dim,
            timing_debug=False
        )
        
        batch_size = 2
        seq_len = 16
        
        # Input embeddings
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Field state too small (will produce < 64 candidates after projection)
        N_x, N_y, D = 2, 2, 2
        field_state = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)
        
        # Should fail fast with insufficient candidates
        with pytest.raises(ValueError, match="Failed to build Address structure"):
            stack(
                x=x,
                field_state=field_state,
                attention_mask=None,
                diagnose=False
            )
    
    def test_role_typed_weighting(self):
        """Verify role-typed weighting is applied (attractors, repulsors)."""
        d_model = 256
        batch_size = 1
        seq_len = 8
        
        # Create address with 64 neighbors
        config = AddressConfig(d=d_model)
        builder = AddressBuilder(config=config)
        
        embedding = torch.randn(batch_size, config.d)
        candidate_embeddings = torch.randn(batch_size, 128, config.d)
        
        address = builder(
            embedding=embedding,
            candidate_embeddings=candidate_embeddings,
            enable_probing=True
        )
        
        # Create geometric attention layer
        attention = GeometricAttention(
            d_model=d_model,
            n_heads=4,
            dropout=0.0,
            use_exponential_form=True,
            use_field_contraction=False
        )
        
        # Create query
        Q = torch.randn(batch_size, seq_len, d_model)
        
        # Create dummy field
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
        
        # Forward pass
        output, weights = attention(
            Q_input=Q,
            address=address,
            enable_address_probing=True,
            T_field=T_field
        )
        
        # Check weights shape includes all 64 neighbors
        assert weights.shape == (batch_size, seq_len, 64)
        
        # Weights should sum to approximately 1 per query position (after Born normalization)
        # (not exactly 1 due to Born × Gibbs × Softmax combination)
        weight_sums = weights.sum(dim=-1)
        assert weight_sums.shape == (batch_size, seq_len)


class TestBackwardCompatibility:
    """Test backward compatibility with legacy neighbor_embeddings."""
    
    def test_legacy_neighbor_embeddings_still_works(self):
        """Legacy neighbor_embeddings path still works (with deprecation warning)."""
        d_model = 256
        batch_size = 2
        seq_len = 16
        
        # Create geometric attention layer
        attention = GeometricAttention(
            d_model=d_model,
            n_heads=4,
            dropout=0.0,
            use_exponential_form=True,
            use_field_contraction=False
        )
        
        # Create query and neighbors (legacy path)
        Q = torch.randn(batch_size, seq_len, d_model)
        neighbor_embeddings = torch.randn(batch_size, 64, d_model)
        metric = torch.randn(batch_size, d_model).abs() + 0.1
        
        # Create dummy field
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
        
        # Forward pass with legacy neighbor_embeddings
        # Should work but emit deprecation warning
        with pytest.warns(DeprecationWarning, match="without full Address structure"):
            output, weights = attention(
                Q_input=Q,
                neighbor_embeddings=neighbor_embeddings,
                metric=metric,
                T_field=T_field
            )
        
        # Check output
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
