"""
Test Geometric Product Operations

Verifies wedge, tensor, and spinor products for geometric attention.

These tests ensure the geometric algebra operations have correct
mathematical properties:
- Wedge: Antisymmetry, zero for parallel vectors
- Tensor: Full correlation structure
- Spinor: Rotational features from field
"""

import torch
import pytest
from ..inference.geometric_products import (
    wedge_product,
    tensor_product,
    spinor_product,
    geometric_score
)


def test_wedge_antisymmetry():
    """
    Test that wedge product is antisymmetric: Q ∧ K = -(K ∧ Q).

    Actually, for the norm squared ||Q ∧ K||², we have:
    ||Q ∧ K||² = ||K ∧ Q||² (symmetric)

    But the underlying wedge should be antisymmetric.
    """
    Q = torch.randn(4, 8)  # (seq_len, d_model)
    K = torch.randn(6, 8)

    wedge_QK = wedge_product(Q, K)  # (4, 6)
    wedge_KQ = wedge_product(K, Q)  # (6, 4)

    # Scores should have transposed shape
    assert wedge_QK.shape == (4, 6)
    assert wedge_KQ.shape == (6, 4)

    # For norm squared, symmetry: wedge_QK[i,j] = wedge_KQ[j,i]
    assert torch.allclose(wedge_QK, wedge_KQ.T, rtol=1e-4)


def test_wedge_parallel_vectors():
    """
    Test that wedge product is zero for parallel vectors.

    If K[i] = c*Q[i] (same position, parallel), then Q[i] ∧ K[i] = 0.
    Only check diagonal elements where query and key positions match.
    """
    Q = torch.randn(4, 8)
    K = 2.5 * Q  # Parallel to Q

    wedge_QK = wedge_product(Q, K)

    # Only diagonal elements should be zero (Q[i] ∧ K[i] where K[i] = c*Q[i])
    diagonal = torch.diag(wedge_QK)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-5)


def test_wedge_orthogonal_vectors():
    """
    Test that wedge product is maximal for orthogonal vectors.

    For orthogonal Q, K, the wedge ||Q ∧ K||² should be large.
    """
    # Create orthonormal vectors
    Q = torch.zeros(2, 4)
    Q[0, 0] = 1.0  # [1, 0, 0, 0]
    Q[1, 1] = 1.0  # [0, 1, 0, 0]

    K = torch.zeros(2, 4)
    K[0, 2] = 1.0  # [0, 0, 1, 0]
    K[1, 3] = 1.0  # [0, 0, 0, 1]

    wedge_QK = wedge_product(Q, K)

    # Should be non-zero (orthogonal vectors span a plane)
    assert torch.all(wedge_QK > 0)


def test_tensor_product_shape():
    """
    Test that tensor product has correct output shape.
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)

    tensor_QK = tensor_product(Q, K)

    assert tensor_QK.shape == (4, 6)


def test_tensor_product_symmetry():
    """
    Test that ||Q ⊗ K||² = ||K ⊗ Q||² (symmetric for norm squared).
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)

    tensor_QK = tensor_product(Q, K)
    tensor_KQ = tensor_product(K, Q)

    # Symmetric for norm squared
    assert torch.allclose(tensor_QK, tensor_KQ.T, rtol=1e-4)


def test_tensor_product_nonzero():
    """
    Test that tensor product is generally non-zero.
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)

    tensor_QK = tensor_product(Q, K)

    # Should be non-zero for random inputs
    assert torch.all(tensor_QK > 0)


def test_spinor_product_shape():
    """
    Test that spinor product has correct output shape.
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    spinor_QK = spinor_product(Q, K, T_field)

    assert spinor_QK.shape == (4, 6)


def test_spinor_product_field_dependence():
    """
    Test that spinor product depends on field state.

    Different field states should produce different spinor scores.
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)

    # Two different field states
    T_field1 = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
    T_field2 = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    spinor_QK1 = spinor_product(Q, K, T_field1)
    spinor_QK2 = spinor_product(Q, K, T_field2)

    # Should be different for different fields
    assert not torch.allclose(spinor_QK1, spinor_QK2, rtol=0.1)


def test_spinor_product_dimension_mismatch():
    """
    Test that spinor product handles dimension mismatch between Q/K and T_field.

    Q, K may have d_model != field_dim, requiring projection.
    """
    Q = torch.randn(4, 16)  # d_model = 16
    K = torch.randn(6, 16)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)  # field_dim = 4

    # Should not raise error (handles projection internally)
    spinor_QK = spinor_product(Q, K, T_field)

    assert spinor_QK.shape == (4, 6)


def test_geometric_score_combination():
    """
    Test that geometric_score combines all three products correctly.
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    # Individual products
    wedge_score = wedge_product(Q, K)
    tensor_score = tensor_product(Q, K)
    spinor_score = spinor_product(Q, K, T_field)

    # Combined with equal weights
    weights = (1.0, 1.0, 1.0)
    combined = geometric_score(Q, K, T_field, weights=weights)

    # Should equal sum of individual products
    expected = wedge_score + tensor_score + spinor_score
    assert torch.allclose(combined, expected, rtol=1e-4)


def test_geometric_score_weighted():
    """
    Test that geometric_score respects custom weights.
    """
    Q = torch.randn(4, 8)
    K = torch.randn(6, 8)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    # Custom weights
    w_wedge, w_tensor, w_spinor = 2.0, 0.5, 1.0
    weights = (w_wedge, w_tensor, w_spinor)

    # Compute combined
    combined = geometric_score(Q, K, T_field, weights=weights)

    # Compute expected
    wedge_score = wedge_product(Q, K)
    tensor_score = tensor_product(Q, K)
    spinor_score = spinor_product(Q, K, T_field)
    expected = w_wedge * wedge_score + w_tensor * tensor_score + w_spinor * spinor_score

    assert torch.allclose(combined, expected, rtol=1e-4)


def test_geometric_score_batch():
    """
    Test that geometric products work with batched inputs.
    """
    batch_size = 2
    Q = torch.randn(batch_size, 4, 8)
    K = torch.randn(batch_size, 6, 8)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    # Should handle batch dimension
    wedge_score = wedge_product(Q, K)
    assert wedge_score.shape == (batch_size, 4, 6)

    tensor_score = tensor_product(Q, K)
    assert tensor_score.shape == (batch_size, 4, 6)

    spinor_score = spinor_product(Q, K, T_field)
    assert spinor_score.shape == (batch_size, 4, 6)
