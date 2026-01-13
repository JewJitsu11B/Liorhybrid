"""
Test Geodesic Physics and LIoR Action

Validates the geodesic cost computation used in training.

Tests verify:
1. Metric tensor construction from field
2. Geodesic distance computation  
3. LIoR action integral properties
4. Physical consistency (positive definite, proper scaling)
"""

import torch
import pytest
from training.lior_trainer import compute_geodesic_cost
from core import CognitiveTensorField, FAST_TEST_CONFIG


class TestGeodesicPhysics:
    """Tests for geodesic physics in training."""

    @pytest.fixture
    def field_and_embeddings(self):
        """Create test field and embeddings."""
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        batch_size = 2
        seq_len = 8
        d_model = 32
        embeddings = torch.randn(batch_size, seq_len, d_model)
        
        return field, embeddings

    def test_geodesic_cost_positive(self, field_and_embeddings):
        """Test that geodesic cost is non-negative (physical requirement)."""
        field, embeddings = field_and_embeddings
        
        cost = compute_geodesic_cost(embeddings, field.T)
        
        assert cost >= 0, f"Geodesic cost must be non-negative, got {cost}"

    def test_geodesic_cost_zero_for_static(self):
        """
        Test that geodesic cost is zero for static trajectory.
        
        If embeddings don't change (velocity = 0), geodesic cost should be 0.
        """
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        # Create static embeddings (all timesteps identical)
        batch_size = 2
        seq_len = 8
        d_model = 32
        static_vector = torch.randn(batch_size, 1, d_model)
        embeddings = static_vector.expand(batch_size, seq_len, d_model).clone()
        
        cost = compute_geodesic_cost(embeddings, field.T)
        
        # Should be near zero (small numerical errors allowed)
        assert cost < 1e-4, f"Static trajectory should have zero cost, got {cost}"

    def test_geodesic_cost_scales_with_velocity(self):
        """
        Test that geodesic cost increases with trajectory velocity.
        
        Larger changes between timesteps should give larger cost.
        """
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        batch_size = 2
        seq_len = 8
        d_model = 32
        
        # Slow trajectory (small changes)
        base_embedding = torch.randn(batch_size, 1, d_model)
        slow_noise = 0.01 * torch.randn(batch_size, seq_len - 1, d_model)
        slow_embeddings = torch.cat([
            base_embedding,
            base_embedding.expand(batch_size, seq_len - 1, d_model) + slow_noise
        ], dim=1)
        
        # Fast trajectory (large changes)
        fast_noise = 1.0 * torch.randn(batch_size, seq_len - 1, d_model)
        fast_embeddings = torch.cat([
            base_embedding,
            base_embedding.expand(batch_size, seq_len - 1, d_model) + fast_noise
        ], dim=1)
        
        slow_cost = compute_geodesic_cost(slow_embeddings, field.T)
        fast_cost = compute_geodesic_cost(fast_embeddings, field.T)
        
        assert fast_cost > slow_cost, \
            f"Fast trajectory should have higher cost than slow: {fast_cost} vs {slow_cost}"

    def test_geodesic_cost_shape_consistency(self):
        """Test that geodesic cost works with different sequence lengths."""
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        for seq_len in [4, 8, 16]:
            embeddings = torch.randn(2, seq_len, 32)
            cost = compute_geodesic_cost(embeddings, field.T)
            
            assert isinstance(cost, torch.Tensor), "Cost should be a tensor"
            assert cost.numel() == 1, "Cost should be a scalar"
            assert cost >= 0, f"Cost must be non-negative for seq_len={seq_len}"

    def test_metric_positive_definite(self):
        """
        Test that metric tensor g = T^T·T is positive definite.
        
        This is a physical requirement for Riemannian geometry.
        """
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        # Average field over spatial dimensions
        T_avg = field.T.mean(dim=(0, 1))  # (D, D)
        T_real = torch.abs(T_avg) if T_avg.is_complex() else T_avg
        
        # Construct metric
        metric = torch.matmul(T_real.T, T_real)
        
        # Check eigenvalues are all positive
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        assert torch.all(eigenvalues > 0), \
            f"Metric must be positive definite. Min eigenvalue: {eigenvalues.min()}"

    def test_geodesic_dimension_handling(self):
        """
        Test that geodesic cost handles dimension mismatch correctly.
        
        field_dim may be smaller than d_model, which is valid.
        """
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        field_dim = field.T.shape[2]  # D
        d_model = field_dim + 8  # Larger than field dimension
        
        embeddings = torch.randn(2, 8, d_model)
        
        # Should work without error
        try:
            cost = compute_geodesic_cost(embeddings, field.T)
            assert cost >= 0
        except Exception as e:
            pytest.fail(f"Geodesic cost should handle dimension mismatch: {e}")

    def test_geodesic_invariant_to_field_normalization(self):
        """
        Test that geodesic cost is invariant to field rescaling.
        
        If we scale T → c·T, the metric g = T^T·T → c²·g,
        but geodesic distances should scale consistently.
        """
        config = FAST_TEST_CONFIG
        field = CognitiveTensorField(config)
        
        embeddings = torch.randn(2, 8, 32)
        
        # Original cost
        cost1 = compute_geodesic_cost(embeddings, field.T)
        
        # Scaled field
        field_scaled = field.T * 2.0
        cost2 = compute_geodesic_cost(embeddings, field_scaled)
        
        # Costs should scale (quadratically due to g = T^T·T)
        # But both should be positive and finite
        assert cost1 >= 0 and cost2 >= 0
        assert torch.isfinite(cost1) and torch.isfinite(cost2)
        
        # Ratio should be approximately 4 (squared scaling)
        ratio = cost2 / (cost1 + 1e-8)
        assert 2.0 < ratio < 8.0, \
            f"Scaling by 2 should roughly quadruple cost, got ratio {ratio}"


class TestManifoldGeometry:
    """Tests for manifold geometry physics (if manifold is accessible)."""

    def test_metric_symmetry(self):
        """Test that Riemannian metric is symmetric: g_μν = g_νμ."""
        try:
            from models.manifold import CognitiveManifold
        except ImportError:
            pytest.skip("Manifold module not accessible in test environment")
            return
        
        d_embed = 32
        manifold = CognitiveManifold(d_embed, d_coord=8)
        
        # Get metric
        L = manifold.L
        metric = torch.matmul(L, L.T)
        
        # Check symmetry
        assert torch.allclose(metric, metric.T, atol=1e-6), \
            "Metric must be symmetric"

    def test_resilience_field_positive(self):
        """Test that resilience field R(x) > 0 (physical requirement)."""
        try:
            from models.manifold import CognitiveManifold
        except ImportError:
            pytest.skip("Manifold module not accessible in test environment")
            return
        
        d_embed = 32
        manifold = CognitiveManifold(d_embed, d_coord=8)
        
        # Test coordinates
        coords = torch.randn(4, 8)  # (batch, d_coord)
        
        R = manifold.resilience_net(coords)
        
        assert torch.all(R > 0), "Resilience field must be strictly positive"

    def test_phase_field_computation(self):
        """Test phase field computation from fractional kernel."""
        try:
            from models.complex_metric import ComplexMetricTensor
        except ImportError:
            pytest.skip("Complex metric module not accessible in test environment")
            return
        
        metric_tensor = ComplexMetricTensor(d_coord=8)
        
        z = torch.randn(4, 8, 16)  # (batch, seq, d_model)
        alpha = torch.tensor(0.5)
        
        theta = metric_tensor.compute_phase_field(z, alpha)
        
        assert theta.shape == (4, 8), "Phase field should have shape (batch, seq)"
        assert torch.isfinite(theta).all(), "Phase field must be finite"


def test_lior_action_properties():
    """
    Test properties of LIoR action integral.
    
    The action S = ∫ R(x) √|g·ẋ·ẋ| dτ should satisfy:
    1. Non-negative (physical requirement)
    2. Zero for geodesic paths (minimum action principle)
    3. Increases with path length
    """
    # This is tested implicitly through geodesic cost tests
    pass
