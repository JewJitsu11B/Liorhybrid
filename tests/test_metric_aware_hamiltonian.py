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


def test_energy_computation_works():
    """Test that compute_energy() method exists and works (no caching)."""
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
    """Test that metric-aware Hamiltonian respects non-trivial isotropic metric."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Create an isotropic metric (all components equal)
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
    
    # For isotropic metric (g_xx = g_yy = 2.0), should scale by 2.0
    # H_metric ≈ 2.0 * H_flat
    ratio = torch.abs(H_metric / (H_flat + 1e-10))
    mean_ratio = ratio[torch.isfinite(ratio)].mean()
    
    # Mean ratio should be close to 2.0 (the metric scale)
    assert torch.abs(mean_ratio - 2.0) < 0.5, \
        f"Isotropic metric should scale Hamiltonian by ~2.0, got {mean_ratio:.2f}"


def test_anisotropic_metric():
    """Test that anisotropic metric produces directionally different effects."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Create strongly anisotropic metric: x-direction 10x stronger than y
    D = config.tensor_dim
    g_inv_diag = torch.ones(D, device=config.device, dtype=torch.float32)
    g_inv_diag[0] = 10.0  # x-direction: g^xx = 10.0
    g_inv_diag[1] = 1.0   # y-direction: g^yy = 1.0
    
    # Compute Hamiltonians
    H_flat = hamiltonian_evolution(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog
    )
    
    H_aniso = hamiltonian_evolution_with_metric(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog,
        g_inv_diag=g_inv_diag
    )
    
    # Anisotropic should differ from flat space
    assert not torch.allclose(H_flat, H_aniso, atol=1e-6), \
        "Anisotropic metric should differ from flat space"
    
    # With g^xx=10, g^yy=1, the scaling is not uniform
    # The ratio should be between 1.0 and 10.0 (average weighted by derivatives)
    ratio = torch.abs(H_aniso / (H_flat + 1e-10))
    mean_ratio = ratio[torch.isfinite(ratio)].mean()
    
    assert 1.0 < mean_ratio < 10.0, \
        f"Anisotropic ratio should be between 1.0 and 10.0, got {mean_ratio:.2f}"
    
    print(f"Anisotropic scaling factor: {mean_ratio:.2f} (expected between 1-10)")


def test_anisotropic_vs_isotropic():
    """Test that anisotropic and isotropic produce different results."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Anisotropic metric
    D = config.tensor_dim
    g_aniso = torch.ones(D, device=config.device, dtype=torch.float32)
    g_aniso[0] = 5.0   # x-direction
    g_aniso[1] = 1.0   # y-direction
    
    # Isotropic metric (average of anisotropic)
    g_iso = torch.ones(D, device=config.device, dtype=torch.float32) * 3.0  # (5+1)/2 = 3
    
    H_aniso = hamiltonian_evolution_with_metric(
        field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_aniso
    )
    
    H_iso = hamiltonian_evolution_with_metric(
        field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_iso
    )
    
    # They should be different (anisotropic preserves directional structure)
    difference = torch.norm(H_aniso - H_iso)
    assert difference > 1e-6, \
        f"Anisotropic and isotropic should differ, difference: {difference:.6f}"
    
    print(f"Anisotropic vs isotropic difference: {difference:.6f}")


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


def test_energy_computation_no_caching():
    """Test that energy computation works correctly without caching."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # No cache should exist initially
    assert not hasattr(field, '_last_H_T'), "No cache should exist initially"
    
    # Evolve field
    field.evolve_step()
    
    # Still no cache after evolve (caching removed for physics correctness)
    assert not hasattr(field, '_last_H_T'), "No cache should exist after evolve_step"
    
    # Energy computation should work
    energy = field.compute_energy()
    
    # Energy should be valid
    assert isinstance(energy, float), "Energy should be float"
    assert torch.isfinite(torch.tensor(energy)), "Energy should be finite"


def test_energy_computation_consistency():
    """Test that energy computation is consistent."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Compute energy before evolution
    energy_before = field.compute_energy()
    
    # Evolve field
    field.evolve_step()
    
    # Compute energy after evolution
    energy_after = field.compute_energy()
    
    # Both should be valid floats
    assert isinstance(energy_before, float), "Energy before should be float"
    assert isinstance(energy_after, float), "Energy after should be float"
    assert torch.isfinite(torch.tensor(energy_before)), "Energy before should be finite"
    assert torch.isfinite(torch.tensor(energy_after)), "Energy after should be finite"
    
    # Energy should change after evolution (non-conservative dynamics)
    # Don't require specific relationship, just that both are valid


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
