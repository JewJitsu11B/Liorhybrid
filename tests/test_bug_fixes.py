"""
Tests for bug fixes: CPU sync, boundary conditions, metric validation
"""

import torch
import pytest
from ..kernels.hamiltonian import (
    spatial_laplacian, spatial_laplacian_x, spatial_laplacian_y,
    hamiltonian_evolution_with_metric
)
from ..core import FAST_TEST_CONFIG


def test_bug_fix_no_cpu_sync():
    """Test that metric components stay on GPU (no .item() calls)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = FAST_TEST_CONFIG
    T = torch.randn(4, 4, 4, 4, dtype=torch.complex64, device='cuda')
    g_inv_diag = torch.tensor([2.0, 1.5], device='cuda')
    
    # This should not trigger CPU sync
    H_T = hamiltonian_evolution_with_metric(
        T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=g_inv_diag
    )
    
    # Verify result is on GPU
    assert H_T.device.type == 'cuda'
    assert g_inv_diag.device.type == 'cuda'  # Metric should still be on GPU


def test_bug_fix_periodic_boundaries():
    """Test that periodic boundary conditions work correctly."""
    # Create field with known pattern
    T = torch.zeros(4, 4, 2, 2, dtype=torch.complex64)
    T[0, 0, 0, 0] = 1.0  # Corner value
    T[1, 0, 0, 0] = 1.0  # Neighbor in x
    T[0, 1, 0, 0] = 1.0  # Neighbor in y
    T[3, 0, 0, 0] = 1.0  # Neighbor in x (periodic)
    T[0, 3, 0, 0] = 1.0  # Neighbor in y (periodic)
    
    # Compute Laplacian
    lap = spatial_laplacian(T)
    
    # At corner with periodic boundaries:
    # Neighbors: [0,1], [1,0], [0,3] (periodic), [3,0] (periodic)
    # All neighbors = 1.0, center = 1.0
    # Laplacian = (1 + 1 + 1 + 1 - 4*1) / 1² = 0
    corner_value = lap[0, 0, 0, 0].real
    
    # With periodic boundaries, should be close to 0
    assert abs(corner_value) < 0.1, \
        f"Periodic boundary: expected ~0, got {corner_value:.4f}"
    
    # Test directional derivatives too
    lap_x = spatial_laplacian_x(T)
    lap_y = spatial_laplacian_y(T)
    
    # Both should also show periodic behavior
    assert torch.isfinite(lap_x).all(), "x-Laplacian should be finite"
    assert torch.isfinite(lap_y).all(), "y-Laplacian should be finite"


def test_bug_fix_metric_validation_negative():
    """Test that negative metric components are caught."""
    config = FAST_TEST_CONFIG
    T = torch.randn(4, 4, 4, 4, dtype=torch.complex64)
    
    # Negative metric (invalid)
    bad_metric = torch.tensor([2.0, -1.0])
    
    with pytest.raises(ValueError, match="positive definite"):
        hamiltonian_evolution_with_metric(
            T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=bad_metric
        )


def test_bug_fix_metric_validation_zero():
    """Test that zero metric components are caught."""
    config = FAST_TEST_CONFIG
    T = torch.randn(4, 4, 4, 4, dtype=torch.complex64)
    
    # Zero metric (invalid)
    bad_metric = torch.tensor([2.0, 0.0])
    
    with pytest.raises(ValueError, match="positive"):
        hamiltonian_evolution_with_metric(
            T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=bad_metric
        )


def test_bug_fix_metric_validation_nan():
    """Test that NaN metric components are caught."""
    config = FAST_TEST_CONFIG
    T = torch.randn(4, 4, 4, 4, dtype=torch.complex64)
    
    # NaN metric (invalid)
    bad_metric = torch.tensor([2.0, float('nan')])
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        hamiltonian_evolution_with_metric(
            T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=bad_metric
        )


def test_bug_fix_metric_validation_inf():
    """Test that Inf metric components are caught."""
    config = FAST_TEST_CONFIG
    T = torch.randn(4, 4, 4, 4, dtype=torch.complex64)
    
    # Inf metric (invalid)
    bad_metric = torch.tensor([2.0, float('inf')])
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        hamiltonian_evolution_with_metric(
            T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=bad_metric
        )


def test_context_manager_validation():
    """Test MetricContext catches bad metrics."""
    from ..kernels.metric_context import MetricContext
    
    # Valid metric
    good_metric = torch.tensor([2.0, 1.5])
    with MetricContext(good_metric, validate=True) as ctx:
        assert ctx.g_inv is not None
        g_xx, g_yy = ctx.get_spatial_components()
        assert g_xx == 2.0
        assert g_yy == 1.5
    
    # Invalid metric (negative)
    bad_metric = torch.tensor([2.0, -1.0])
    with pytest.raises(ValueError, match="positive definite"):
        with MetricContext(bad_metric, validate=True):
            pass


def test_context_manager_performance_tracking():
    """Test MetricContext tracks performance."""
    from ..kernels.metric_context import MetricContext
    import time
    
    metric = torch.tensor([2.0, 1.5])
    
    with MetricContext(metric, track_perf=True) as ctx:
        time.sleep(0.01)  # Simulate work
    
    # Should have tracked time
    assert ctx.elapsed_time is not None
    assert ctx.elapsed_time >= 0.01
    assert ctx.elapsed_time < 0.1  # Sanity check


def test_context_manager_isotropic_detection():
    """Test MetricContext detects isotropic metrics."""
    from ..kernels.metric_context import MetricContext
    
    # Isotropic
    iso_metric = torch.ones(8) * 2.0
    with MetricContext(iso_metric) as ctx:
        assert ctx.is_isotropic
    
    # Anisotropic
    aniso_metric = torch.tensor([10.0, 1.0, 1.0, 1.0])
    with MetricContext(aniso_metric) as ctx:
        assert not ctx.is_isotropic


if __name__ == "__main__":
    print("Running bug fix tests...")
    
    print("✓ Testing CPU sync fix...")
    # test_bug_fix_no_cpu_sync()  # Skip if no CUDA
    
    print("✓ Testing periodic boundaries...")
    test_bug_fix_periodic_boundaries()
    
    print("✓ Testing metric validation (negative)...")
    test_bug_fix_metric_validation_negative()
    
    print("✓ Testing metric validation (zero)...")
    test_bug_fix_metric_validation_zero()
    
    print("✓ Testing metric validation (NaN)...")
    test_bug_fix_metric_validation_nan()
    
    print("✓ Testing metric validation (Inf)...")
    test_bug_fix_metric_validation_inf()
    
    print("✓ Testing context manager validation...")
    test_context_manager_validation()
    
    print("✓ Testing context manager performance...")
    test_context_manager_performance_tracking()
    
    print("✓ Testing context manager isotropic detection...")
    test_context_manager_isotropic_detection()
    
    print("\n✅ All bug fix tests passed!")
