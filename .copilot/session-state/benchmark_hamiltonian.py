"""
Performance benchmark for hamiltonian.py
Tests the performance claims in the code review.
"""

import torch
import time
from kernels.hamiltonian import (
    spatial_laplacian,
    spatial_laplacian_x,
    spatial_laplacian_y,
    hamiltonian_evolution_with_metric
)
from core import CognitiveTensorField, FAST_TEST_CONFIG


def benchmark_cpu_sync():
    """Test impact of .item() CPU sync."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    g_inv_diag = torch.ones(config.tensor_dim, device=config.device) * 2.0
    
    # Warmup
    for _ in range(10):
        hamiltonian_evolution_with_metric(
            field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_inv_diag
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        hamiltonian_evolution_with_metric(
            field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_inv_diag
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Hamiltonian with metric (100 iters): {elapsed*1000:.2f}ms")
    print(f"Per iteration: {elapsed*10:.3f}ms")
    return elapsed


def benchmark_convolution_ops():
    """Benchmark convolution operations separately."""
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Warmup
    for _ in range(10):
        spatial_laplacian(field.T, dx=1.0)
    
    torch.cuda.synchronize()
    
    # Benchmark full Laplacian
    start = time.time()
    for _ in range(100):
        spatial_laplacian(field.T, dx=1.0)
    torch.cuda.synchronize()
    lap_time = time.time() - start
    
    # Benchmark x derivative
    start = time.time()
    for _ in range(100):
        spatial_laplacian_x(field.T, dx=1.0)
    torch.cuda.synchronize()
    lapx_time = time.time() - start
    
    # Benchmark y derivative
    start = time.time()
    for _ in range(100):
        spatial_laplacian_y(field.T, dy=1.0)
    torch.cuda.synchronize()
    lapy_time = time.time() - start
    
    print(f"\nConvolution ops (100 iters):")
    print(f"  spatial_laplacian:   {lap_time*1000:.2f}ms ({lap_time*10:.3f}ms/iter)")
    print(f"  spatial_laplacian_x: {lapx_time*1000:.2f}ms ({lapx_time*10:.3f}ms/iter)")
    print(f"  spatial_laplacian_y: {lapy_time*1000:.2f}ms ({lapy_time*10:.3f}ms/iter)")
    print(f"  Combined x+y:        {(lapx_time+lapy_time)*1000:.2f}ms")
    
    return lap_time, lapx_time, lapy_time


def test_boundary_conditions():
    """Test if periodic boundary conditions work correctly."""
    print("\n=== Boundary Condition Test ===")
    
    # Create simple field with edge values
    T = torch.zeros((28, 28, 16, 16), dtype=torch.complex64, device='cuda')
    
    # Set value at edge
    T[0, 14, 0, 0] = 1.0  # Left edge, middle
    T[-1, 14, 0, 0] = 1.0  # Right edge, middle (should couple with left in PBC)
    
    lap = spatial_laplacian(T, dx=1.0)
    
    # Check if edges couple (for periodic BC)
    left_lap = lap[0, 14, 0, 0]
    right_lap = lap[-1, 14, 0, 0]
    
    print(f"Laplacian at left edge:  {left_lap:.6f}")
    print(f"Laplacian at right edge: {right_lap:.6f}")
    
    if torch.abs(left_lap) > 1e-6 and torch.abs(right_lap) > 1e-6:
        print("✓ Edges couple (likely periodic)")
    else:
        print("✗ Edges don't couple (likely zero/reflection padding)")
        print("  WARNING: Boundary conditions may not be periodic as documented!")


def test_metric_edge_cases():
    """Test edge cases for metric values."""
    print("\n=== Metric Edge Case Tests ===")
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    
    # Test 1: Very small metric (near zero)
    g_small = torch.ones(config.tensor_dim, device=config.device) * 1e-8
    try:
        H = hamiltonian_evolution_with_metric(
            field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_small
        )
        finite = torch.all(torch.isfinite(H))
        print(f"Small metric (1e-8): {'✓ Finite' if finite else '✗ NaN/Inf detected'}")
    except Exception as e:
        print(f"Small metric (1e-8): ✗ Exception: {e}")
    
    # Test 2: Zero metric
    g_zero = torch.zeros(config.tensor_dim, device=config.device)
    try:
        H = hamiltonian_evolution_with_metric(
            field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_zero
        )
        finite = torch.all(torch.isfinite(H))
        print(f"Zero metric: {'⚠️  Accepts (should reject?)' if finite else '✗ NaN/Inf'}")
    except Exception as e:
        print(f"Zero metric: ✓ Rejected with: {type(e).__name__}")
    
    # Test 3: Negative metric
    g_neg = -torch.ones(config.tensor_dim, device=config.device)
    try:
        H = hamiltonian_evolution_with_metric(
            field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_neg
        )
        finite = torch.all(torch.isfinite(H))
        print(f"Negative metric: {'⚠️  Accepts (should reject!)' if finite else '✗ NaN/Inf'}")
    except Exception as e:
        print(f"Negative metric: ✓ Rejected with: {type(e).__name__}")
    
    # Test 4: Very large metric
    g_large = torch.ones(config.tensor_dim, device=config.device) * 1e8
    try:
        H = hamiltonian_evolution_with_metric(
            field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_large
        )
        finite = torch.all(torch.isfinite(H))
        print(f"Large metric (1e8): {'✓ Finite' if finite else '✗ NaN/Inf detected'}")
    except Exception as e:
        print(f"Large metric (1e8): ✗ Exception: {e}")


def test_shape_validation():
    """Test if shape validation exists."""
    print("\n=== Shape Validation Tests ===")
    
    # Test 1: 3D tensor
    T_3d = torch.randn(28, 28, 16, dtype=torch.complex64, device='cuda')
    try:
        lap = spatial_laplacian(T_3d, dx=1.0)
        print("3D tensor: ⚠️  Accepted (should reject!)")
    except (ValueError, RuntimeError) as e:
        print(f"3D tensor: ✓ Rejected")
    
    # Test 2: 5D tensor
    T_5d = torch.randn(4, 28, 28, 16, 16, dtype=torch.complex64, device='cuda')
    try:
        lap = spatial_laplacian(T_5d, dx=1.0)
        print("5D tensor: ⚠️  Accepted (should reject!)")
    except (ValueError, RuntimeError) as e:
        print(f"5D tensor: ✓ Rejected")
    
    # Test 3: Very small grid
    T_small = torch.randn(2, 2, 16, 16, dtype=torch.complex64, device='cuda')
    try:
        lap = spatial_laplacian(T_small, dx=1.0)
        print("2x2 grid: ⚠️  Accepted (may have artifacts with 3x3 stencil)")
    except (ValueError, RuntimeError) as e:
        print(f"2x2 grid: ✓ Rejected")


def memory_profiling():
    """Profile memory allocations."""
    print("\n=== Memory Profiling ===")
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)
    g_inv_diag = torch.ones(config.tensor_dim, device=config.device) * 2.0
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Single call
    start_mem = torch.cuda.memory_allocated() / 1e6
    H = hamiltonian_evolution_with_metric(
        field.T, config.hbar_cog, config.m_cog, g_inv_diag=g_inv_diag
    )
    end_mem = torch.cuda.memory_allocated() / 1e6
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    
    T_size = field.T.element_size() * field.T.nelement() / 1e6
    
    print(f"Input tensor size: {T_size:.2f} MB")
    print(f"Memory before:     {start_mem:.2f} MB")
    print(f"Memory after:      {end_mem:.2f} MB")
    print(f"Peak memory:       {peak_mem:.2f} MB")
    print(f"Peak overhead:     {(peak_mem - start_mem) / T_size:.2f}x input size")


if __name__ == "__main__":
    print("=" * 60)
    print("Hamiltonian Performance Benchmark")
    print("=" * 60)
    
    print("\n1. Overall Performance")
    total_time = benchmark_cpu_sync()
    
    print("\n2. Convolution Operation Breakdown")
    lap_time, lapx_time, lapy_time = benchmark_convolution_ops()
    
    print("\n3. Boundary Conditions")
    test_boundary_conditions()
    
    print("\n4. Edge Case Handling")
    test_metric_edge_cases()
    
    print("\n5. Shape Validation")
    test_shape_validation()
    
    print("\n6. Memory Usage")
    memory_profiling()
    
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Total time (100 iters): {total_time*1000:.2f}ms")
    print(f"Conv overhead:          {(lap_time+lapx_time+lapy_time)*1000:.2f}ms")
    print(f"Conv % of total:        {(lap_time+lapx_time+lapy_time)/total_time*100:.1f}%")
    print("=" * 60)
