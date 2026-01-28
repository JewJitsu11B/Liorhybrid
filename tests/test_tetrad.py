"""
Tests for Tetrad (Vielbein) implementation
"""

import torch
import pytest
from ..kernels.tetrad import (
    Tetrad,
    compute_metric_from_tetrad,
    anisotropic_laplacian_from_tetrad
)
from ..kernels.hamiltonian import spatial_laplacian_x, spatial_laplacian_y


def test_tetrad_from_isotropic_metric():
    """Test tetrad computation from isotropic metric."""
    tetrad = Tetrad(dim=2)
    
    # Isotropic metric: g^xx = g^yy = 4
    g_inv_diag = torch.tensor([4.0, 4.0])
    
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # For diagonal metric, tetrad should be diagonal
    assert torch.allclose(e, torch.diag(torch.diag(e)), atol=1e-6)
    
    # Diagonal elements should be √4 = 2
    assert torch.allclose(torch.diag(e), torch.tensor([2.0, 2.0]), atol=1e-5)


def test_tetrad_from_anisotropic_metric():
    """Test tetrad computation from anisotropic metric."""
    tetrad = Tetrad(dim=2)
    
    # Anisotropic metric: g^xx = 9, g^yy = 1
    g_inv_diag = torch.tensor([9.0, 1.0])
    
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Diagonal elements should be [√9, √1] = [3, 1]
    expected = torch.tensor([3.0, 1.0])
    assert torch.allclose(torch.diag(e), expected, atol=1e-5)


def test_tetrad_orthonormality():
    """Test that tetrad satisfies orthonormality condition."""
    tetrad = Tetrad(dim=2)
    
    g_inv_diag = torch.tensor([4.0, 1.0])
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Check e^a_μ e^μ_b = δ^a_b
    is_ortho, error = tetrad.verify_orthonormality(e)
    
    assert is_ortho, f"Tetrad not orthonormal, error: {error:.2e}"
    assert error < 1e-5


def test_metric_reconstruction_from_tetrad():
    """Test reconstructing metric from tetrad."""
    tetrad = Tetrad(dim=2)
    
    # Start with metric
    g_inv_diag = torch.tensor([9.0, 4.0])
    
    # Compute tetrad
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Reconstruct metric: g_μν = e^T @ e
    g = compute_metric_from_tetrad(e)
    
    # For diagonal case, should get back diagonal metric
    expected = torch.diag(g_inv_diag)
    assert torch.allclose(g, expected, atol=1e-5)


def test_tetrad_inverse():
    """Test inverse tetrad computation."""
    tetrad = Tetrad(dim=2)
    
    g_inv_diag = torch.tensor([4.0, 1.0])
    e = tetrad.compute_from_metric(g_inv_diag)
    
    e_inv = tetrad.compute_inverse(e)
    
    # e @ e_inv should be identity
    product = e @ e_inv
    identity = torch.eye(2)
    
    assert torch.allclose(product, identity, atol=1e-5)


def test_tetrad_with_clifford_connection():
    """Test tetrad contraction with Clifford gamma matrices."""
    tetrad = Tetrad(dim=2)
    
    # Simple metric
    g_inv_diag = torch.tensor([2.0, 1.0])
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Create simple gamma matrices (Pauli-like)
    gamma = torch.zeros(2, 2, 2)
    gamma[0] = torch.tensor([[0, 1], [1, 0]])  # σ_x
    gamma[1] = torch.tensor([[0, -1j], [1j, 0]])  # σ_y
    
    # Contract: Γ_μ = e^a_μ γ_a
    Gamma = tetrad.contract_with_clifford(e, gamma)
    
    # Should get 2 matrices of shape (2, 2)
    assert Gamma.shape == (2, 2, 2)
    
    # Γ_0 should be e[0,0] * γ_0 (since e is diagonal)
    expected_0 = e[0, 0] * gamma[0]
    assert torch.allclose(Gamma[0], expected_0, atol=1e-5)


def test_anisotropic_laplacian_via_tetrad():
    """Test computing anisotropic Laplacian using tetrad."""
    # Create test field
    T = torch.randn(4, 4, 2, 2, dtype=torch.complex64)
    
    # Create tetrad
    tetrad = Tetrad(dim=2)
    g_inv_diag = torch.tensor([4.0, 1.0])
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Compute via tetrad
    lap_tetrad = anisotropic_laplacian_from_tetrad(
        T, e, spatial_laplacian_x, spatial_laplacian_y
    )
    
    # Compute directly
    d2_dx2 = spatial_laplacian_x(T)
    d2_dy2 = spatial_laplacian_y(T)
    lap_direct = g_inv_diag[0] * d2_dx2 + g_inv_diag[1] * d2_dy2
    
    # Should match
    assert torch.allclose(lap_tetrad, lap_direct, atol=1e-5)


def test_tetrad_dimension_handling():
    """Test tetrad handles different dimensions correctly."""
    tetrad = Tetrad(dim=2)
    
    # Metric with more components than needed
    g_inv_diag = torch.tensor([4.0, 1.0, 2.0, 3.0])
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Should only use first 2 components
    assert e.shape == (2, 2)
    assert torch.allclose(torch.diag(e), torch.sqrt(g_inv_diag[:2]), atol=1e-5)
    
    # Metric with fewer components
    g_inv_diag_short = torch.tensor([9.0])
    e_short = tetrad.compute_from_metric(g_inv_diag_short)
    
    # Should pad with ones
    assert e_short.shape == (2, 2)


def test_tetrad_forward():
    """Test tetrad forward pass."""
    tetrad = Tetrad(dim=2)
    
    # Without metric - returns identity
    e = tetrad.forward()
    assert torch.allclose(e, torch.eye(2), atol=1e-5)
    
    # With metric
    g_inv_diag = torch.tensor([4.0, 1.0])
    e = tetrad.forward(g_inv_diag)
    assert torch.allclose(torch.diag(e), torch.sqrt(g_inv_diag), atol=1e-5)


def test_learnable_tetrad():
    """Test learnable tetrad mode."""
    tetrad = Tetrad(dim=2, learnable=True)
    
    # Should have learnable parameter
    assert hasattr(tetrad, 'e')
    assert isinstance(tetrad.e, torch.nn.Parameter)
    
    # Should start as identity
    assert torch.allclose(tetrad.e, torch.eye(2), atol=1e-5)


if __name__ == "__main__":
    print("Running tetrad tests...")
    
    print("✓ Testing isotropic metric...")
    test_tetrad_from_isotropic_metric()
    
    print("✓ Testing anisotropic metric...")
    test_tetrad_from_anisotropic_metric()
    
    print("✓ Testing orthonormality...")
    test_tetrad_orthonormality()
    
    print("✓ Testing metric reconstruction...")
    test_metric_reconstruction_from_tetrad()
    
    print("✓ Testing inverse tetrad...")
    test_tetrad_inverse()
    
    print("✓ Testing Clifford connection...")
    test_tetrad_with_clifford_connection()
    
    print("✓ Testing Laplacian via tetrad...")
    test_anisotropic_laplacian_via_tetrad()
    
    print("✓ Testing dimension handling...")
    test_tetrad_dimension_handling()
    
    print("✓ Testing forward pass...")
    test_tetrad_forward()
    
    print("✓ Testing learnable mode...")
    test_learnable_tetrad()
    
    print("\n✅ All tetrad tests passed!")
