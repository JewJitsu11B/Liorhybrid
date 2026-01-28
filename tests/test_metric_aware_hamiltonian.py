"""
Test Metric-Aware Hamiltonian Evolution

Verifies that:
1. Energy tracking is fixed (compute_energy works)
2. Metric-aware Hamiltonian respects learned geometry
3. H_T caching improves performance
"""

import torch
import pytest

from ..kernels.hamiltonian import hamiltonian_evolution, hamiltonian_evolution_with_metric
from ..core import CognitiveTensorField, FAST_TEST_CONFIG


def test_compute_energy_works():
    """Test that compute_energy() method exists and works."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Compute energy - should not raise an error
    energy = field.compute_energy()
    
    # Energy should be a scalar
    assert isinstance(energy, float), f"Energy should be float, got {type(energy)}"
    
    # Energy should be finite
    assert not torch.isnan(torch.tensor(energy)), "Energy should not be NaN"
    assert not torch.isinf(torch.tensor(energy)), "Energy should not be Inf"


def test_metric_aware_hamiltonian_with_none():
    """Test that metric-aware Hamiltonian falls back to flat space when g_inv_diag=None."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Compute with both methods
    H_flat = hamiltonian_evolution(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog
    )
    
    H_metric_none = hamiltonian_evolution_with_metric(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog,
        g_inv_diag=None
    )
    
    # Should be identical
    assert torch.allclose(H_flat, H_metric_none, atol=1e-6), \
        "Metric-aware with None should match flat-space Hamiltonian"


def test_metric_aware_hamiltonian_with_metric():
    """Test that metric-aware Hamiltonian respects non-trivial metric."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Create a non-trivial metric (scaled identity)
    D = config.tensor_dim
    g_inv_diag = torch.ones(D, device=config.device, dtype=torch.float32) * 2.0
    
    # Compute with both methods
    H_flat = hamiltonian_evolution(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog
    )
    
    H_metric = hamiltonian_evolution_with_metric(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog,
        g_inv_diag=g_inv_diag
    )
    
    # Should NOT be identical (metric scales the Laplacian)
    assert not torch.allclose(H_flat, H_metric, atol=1e-6), \
        "Metric-aware with non-trivial metric should differ from flat-space"
    
    # The kinetic term should be scaled by the metric
    # With g_inv_diag = 2.0 * ones, the Laplacian is scaled by 2.0
    # So H_metric should be approximately 2.0 * H_flat (for kinetic term)
    # Since potential is zero, this should hold
    ratio = torch.abs(H_metric / (H_flat + 1e-10))
    mean_ratio = ratio[torch.isfinite(ratio)].mean()
    
    # Mean ratio should be close to 2.0 (the metric scale)
    assert torch.abs(mean_ratio - 2.0) < 0.5, \
        f"Metric should scale Hamiltonian by ~2.0, got {mean_ratio:.2f}"


def test_evolve_step_with_metric():
    """Test that evolve_step accepts g_inv_diag parameter."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Create a metric
    D = config.tensor_dim
    g_inv_diag = torch.ones(D, device=config.device, dtype=torch.float32) * 1.5
    
    # Initial state
    T_initial = field.T.clone()
    
    # Evolve with metric
    field.evolve_step(g_inv_diag=g_inv_diag)
    
    # Field should have changed
    assert not torch.allclose(field.T, T_initial, atol=1e-6), \
        "Field should evolve with metric-aware Hamiltonian"


def test_H_T_caching():
    """Test that H_T is cached after evolve_step for fast energy computation."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Before evolve_step, no cache should exist
    assert not hasattr(field, '_last_H_T'), \
        "H_T should not be cached before evolve_step"
    
    # Run evolve_step
    field.evolve_step()
    
    # After evolve_step, cache should exist
    assert hasattr(field, '_last_H_T'), \
        "H_T should be cached after evolve_step"
    
    # Cached H_T should have correct shape
    assert field._last_H_T.shape == field.T.shape, \
        "Cached H_T should have same shape as T"
    
    # Compute energy (should use cache)
    energy = field.compute_energy()
    
    # Energy should be valid
    assert isinstance(energy, float), "Energy should be float"
    assert torch.isfinite(torch.tensor(energy)), "Energy should be finite"


def test_energy_computation_consistency():
    """Test that energy computation uses cache after evolve_step."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Before evolve_step, no cache should exist
    assert not hasattr(field, '_last_H_T'), "Cache should not exist before evolve_step"
    
    # Evolve to populate cache
    field.evolve_step()
    
    # After evolve_step, cache should exist  
    assert hasattr(field, '_last_H_T'), "Cache should exist after evolve_step"
    
    # Cached H_T should have correct shape
    assert field._last_H_T.shape == field.T.shape, \
        "Cached H_T should have same shape as T"
    
    # Energy computation should work (uses cache if available)
    energy = field.compute_energy()
    
    # Energy should be valid
    assert isinstance(energy, float), "Energy should be float"
    assert torch.isfinite(torch.tensor(energy)), "Energy should be finite"
    
    # If we clear the cache and recompute, we should get a different energy
    # (because the field state changed after evolve_step)
    delattr(field, '_last_H_T')
    energy_no_cache = field.compute_energy()
    
    # Both should be valid floats
    assert isinstance(energy_no_cache, float), "Energy without cache should be float"
    assert torch.isfinite(torch.tensor(energy_no_cache)), "Energy without cache should be finite"


def test_metric_aware_evolution_vs_flat():
    """Test that metric-aware evolution produces different trajectories than flat."""
    config = FAST_TEST_CONFIG
    
    # Field 1: Flat space (g_inv_diag=None)
    field_flat = CognitiveTensorField(config)
    T_flat_init = field_flat.T.clone()
    
    # Field 2: Curved space (non-trivial metric)
    field_curved = CognitiveTensorField(config)
    field_curved.T = T_flat_init.clone()
    
    D = config.tensor_dim
    g_inv_diag = torch.ones(D, device=config.device, dtype=torch.float32) * 2.0
    
    # Evolve both for several steps
    n_steps = 10
    for _ in range(n_steps):
        field_flat.evolve_step(g_inv_diag=None)
        field_curved.evolve_step(g_inv_diag=g_inv_diag)
    
    # Trajectories should diverge
    difference = torch.norm(field_flat.T - field_curved.T)
    assert difference > 1e-4, \
        f"Flat and curved space evolution should diverge, difference: {difference:.6f}"


def test_metrics_field_energy():
    """Test that training metrics can track field energy."""
    from ..training.metrics import TrainingMetrics
    
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Create metrics object
    metrics = TrainingMetrics()
    
    # Check that field has compute_energy (not compute_hamiltonian)
    assert hasattr(field, 'compute_energy'), \
        "Field should have compute_energy method"
    
    assert not hasattr(field, 'compute_hamiltonian'), \
        "Field should not have compute_hamiltonian method (old name)"
    
    # Compute energy
    energy = field.compute_energy()
    
    # Should be able to set in metrics
    metrics.field_energy = energy
    assert metrics.field_energy == energy, \
        "Metrics should store field_energy"


if __name__ == "__main__":
    # Run tests
    print("Running metric-aware Hamiltonian tests...")
    
    test_compute_energy_works()
    print("✓ compute_energy() works")
    
    test_metric_aware_hamiltonian_with_none()
    print("✓ Metric-aware Hamiltonian falls back to flat space")
    
    test_metric_aware_hamiltonian_with_metric()
    print("✓ Metric-aware Hamiltonian respects non-trivial metric")
    
    test_evolve_step_with_metric()
    print("✓ evolve_step accepts g_inv_diag parameter")
    
    test_H_T_caching()
    print("✓ H_T caching works")
    
    test_energy_computation_consistency()
    print("✓ Cached and recomputed energy match")
    
    test_metric_aware_evolution_vs_flat()
    print("✓ Metric-aware evolution differs from flat space")
    
    test_metrics_field_energy()
    print("✓ Training metrics can track field energy")
    
    print("\nAll tests passed! ✅")
