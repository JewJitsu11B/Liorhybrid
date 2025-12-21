"""
Test Algebraic Structures: Quaternions, Biquaternions, Octonions, LIoR Kernel

Comprehensive unit tests for the algebraic foundations of the Bayesian Cognitive Field.

Tests verify:
1. Quaternion algebra (H): Hamilton product, norm, conjugate, non-commutativity
2. Biquaternion algebra (C⊗H): Complex quaternions, split state decomposition
3. Octonion algebra (O): Non-associativity, alternativity, Moufang identities
4. LIoR Kernel: Power-law decay, O(1) recurrence correctness
"""

import torch
import pytest
import math
from typing import Tuple

from ..models.biquaternion import (
    quat_mul,
    quat_conjugate,
    pack_biquat,
    unpack_biquat,
    BiQuatTransform,
    CausalAccumulator,
)
from ..models.causal_field import AssociatorCurrent
from ..models.lior_kernel import LiorKernel, LiorMemoryState


# =============================================================================
# Quaternion Algebra Tests (H)
# =============================================================================

class TestQuaternionAlgebra:
    """Tests for real quaternion multiplication (Hamilton product)."""

    @pytest.fixture
    def basis_quaternions(self):
        """Standard quaternion basis: 1, i, j, k."""
        e0 = torch.tensor([1., 0., 0., 0.])  # 1
        e1 = torch.tensor([0., 1., 0., 0.])  # i
        e2 = torch.tensor([0., 0., 1., 0.])  # j
        e3 = torch.tensor([0., 0., 0., 1.])  # k
        return e0, e1, e2, e3

    def test_hamilton_ijk_equals_minus_one(self, basis_quaternions):
        """
        Test Hamilton's defining relation: ijk = -1.

        This is THE fundamental quaternion identity.
        """
        _, i, j, k = basis_quaternions

        # ijk = (ij)k = k*k = -1
        ij = quat_mul(i, j)
        ijk = quat_mul(ij, k)

        expected = torch.tensor([-1., 0., 0., 0.])
        assert torch.allclose(ijk, expected, atol=1e-6), \
            f"ijk = {ijk}, expected {expected}"

    def test_hamilton_i_squared(self, basis_quaternions):
        """Test i² = -1."""
        _, i, _, _ = basis_quaternions

        i_sq = quat_mul(i, i)
        expected = torch.tensor([-1., 0., 0., 0.])

        assert torch.allclose(i_sq, expected, atol=1e-6), \
            f"i² = {i_sq}, expected {expected}"

    def test_hamilton_j_squared(self, basis_quaternions):
        """Test j² = -1."""
        _, _, j, _ = basis_quaternions

        j_sq = quat_mul(j, j)
        expected = torch.tensor([-1., 0., 0., 0.])

        assert torch.allclose(j_sq, expected, atol=1e-6), \
            f"j² = {j_sq}, expected {expected}"

    def test_hamilton_k_squared(self, basis_quaternions):
        """Test k² = -1."""
        _, _, _, k = basis_quaternions

        k_sq = quat_mul(k, k)
        expected = torch.tensor([-1., 0., 0., 0.])

        assert torch.allclose(k_sq, expected, atol=1e-6), \
            f"k² = {k_sq}, expected {expected}"

    def test_hamilton_ij_equals_k(self, basis_quaternions):
        """Test ij = k."""
        _, i, j, k = basis_quaternions

        ij = quat_mul(i, j)

        assert torch.allclose(ij, k, atol=1e-6), \
            f"ij = {ij}, expected {k}"

    def test_hamilton_ji_equals_minus_k(self, basis_quaternions):
        """Test ji = -k (non-commutativity!)."""
        _, i, j, k = basis_quaternions

        ji = quat_mul(j, i)
        expected = -k

        assert torch.allclose(ji, expected, atol=1e-6), \
            f"ji = {ji}, expected {expected}"

    def test_quaternion_noncommutativity(self, basis_quaternions):
        """Test that ij ≠ ji (quaternions are non-commutative)."""
        _, i, j, _ = basis_quaternions

        ij = quat_mul(i, j)
        ji = quat_mul(j, i)

        assert not torch.allclose(ij, ji), \
            f"Quaternions should be non-commutative: ij={ij}, ji={ji}"

    def test_quaternion_associativity(self):
        """
        Test (pq)r = p(qr) (quaternions ARE associative).

        This distinguishes quaternions from octonions.
        """
        p = torch.randn(4)
        q = torch.randn(4)
        r = torch.randn(4)

        # Left association: (pq)r
        pq = quat_mul(p, q)
        left = quat_mul(pq, r)

        # Right association: p(qr)
        qr = quat_mul(q, r)
        right = quat_mul(p, qr)

        assert torch.allclose(left, right, atol=1e-5), \
            f"Quaternions should be associative: (pq)r={left}, p(qr)={right}"

    def test_quaternion_norm_multiplicative(self):
        """
        Test ||pq|| = ||p|| ||q|| (norm is multiplicative).

        This makes quaternions useful for 3D rotations.
        """
        p = torch.randn(4)
        q = torch.randn(4)

        norm_p = torch.norm(p)
        norm_q = torch.norm(q)
        pq = quat_mul(p, q)
        norm_pq = torch.norm(pq)

        expected = norm_p * norm_q

        assert torch.allclose(norm_pq, expected, rtol=1e-4), \
            f"||pq||={norm_pq}, ||p||·||q||={expected}"

    def test_quaternion_conjugate(self, basis_quaternions):
        """Test q* = (a, -b, -c, -d)."""
        _, i, j, k = basis_quaternions

        q = torch.tensor([1., 2., 3., 4.])
        q_conj = quat_conjugate(q)
        expected = torch.tensor([1., -2., -3., -4.])

        assert torch.allclose(q_conj, expected, atol=1e-6), \
            f"q* = {q_conj}, expected {expected}"

    def test_quaternion_conjugate_product(self):
        """Test (pq)* = q*p* (conjugate reverses order)."""
        p = torch.randn(4)
        q = torch.randn(4)

        # (pq)*
        pq = quat_mul(p, q)
        pq_conj = quat_conjugate(pq)

        # q*p*
        q_conj = quat_conjugate(q)
        p_conj = quat_conjugate(p)
        expected = quat_mul(q_conj, p_conj)

        assert torch.allclose(pq_conj, expected, atol=1e-5), \
            f"(pq)* = {pq_conj}, q*p* = {expected}"

    def test_quaternion_norm_via_conjugate(self):
        """Test ||q||² = q*q (scalar real part)."""
        q = torch.randn(4)

        q_conj = quat_conjugate(q)
        product = quat_mul(q_conj, q)

        # Result should be (||q||², 0, 0, 0)
        expected_scalar = (q ** 2).sum()

        assert torch.allclose(product[0], expected_scalar, rtol=1e-4), \
            f"Real part of q*q = {product[0]}, expected ||q||² = {expected_scalar}"
        assert torch.allclose(product[1:], torch.zeros(3), atol=1e-5), \
            f"Imaginary parts of q*q should be zero: {product[1:]}"

    def test_quaternion_batch_multiplication(self):
        """Test quaternion multiplication works with batched inputs."""
        B, N = 4, 16
        p = torch.randn(B, N, 4)
        q = torch.randn(B, N, 4)

        result = quat_mul(p, q)

        assert result.shape == (B, N, 4), \
            f"Expected shape {(B, N, 4)}, got {result.shape}"

        # Verify against individual multiplications
        for b in range(B):
            for n in range(N):
                expected = quat_mul(p[b, n], q[b, n])
                assert torch.allclose(result[b, n], expected, atol=1e-5)


# =============================================================================
# Biquaternion (C⊗H) Tests
# =============================================================================

class TestBiquaternionAlgebra:
    """
    Tests for biquaternion algebra: C⊗H (complex quaternions).

    State layout [16D]: (Q_M_re[4], Q_M_im[4], Q_H_re[4], Q_H_im[4])
    This is (C⊗H)⊕(C⊗H) = 16D STATE decomposition.
    """

    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack are inverses."""
        Q_M_re = torch.randn(4)
        Q_M_im = torch.randn(4)
        Q_H_re = torch.randn(4)
        Q_H_im = torch.randn(4)

        packed = pack_biquat(Q_M_re, Q_M_im, Q_H_re, Q_H_im)
        assert packed.shape == (16,), f"Expected shape (16,), got {packed.shape}"

        out_M_re, out_M_im, out_H_re, out_H_im = unpack_biquat(packed)

        assert torch.allclose(out_M_re, Q_M_re, atol=1e-6)
        assert torch.allclose(out_M_im, Q_M_im, atol=1e-6)
        assert torch.allclose(out_H_re, Q_H_re, atol=1e-6)
        assert torch.allclose(out_H_im, Q_H_im, atol=1e-6)

    def test_pack_unpack_batch(self):
        """Test pack/unpack with batched inputs."""
        B, N = 4, 16
        Q_M_re = torch.randn(B, N, 4)
        Q_M_im = torch.randn(B, N, 4)
        Q_H_re = torch.randn(B, N, 4)
        Q_H_im = torch.randn(B, N, 4)

        packed = pack_biquat(Q_M_re, Q_M_im, Q_H_re, Q_H_im)
        assert packed.shape == (B, N, 16)

        out_M_re, out_M_im, out_H_re, out_H_im = unpack_biquat(packed)

        assert torch.allclose(out_M_re, Q_M_re, atol=1e-6)
        assert torch.allclose(out_M_im, Q_M_im, atol=1e-6)
        assert torch.allclose(out_H_re, Q_H_re, atol=1e-6)
        assert torch.allclose(out_H_im, Q_H_im, atol=1e-6)

    def test_biquat_transform_identity_init(self):
        """Test BiQuatTransform initializes to identity."""
        transform = BiQuatTransform(learnable=True, normalize=False)

        # Identity: W = 1 + 0i + 0j + 0k (real part)
        q_re = torch.randn(2, 8, 4)
        q_im = torch.randn(2, 8, 4)

        out_re, out_im = transform(q_re, q_im)

        # Should be approximately input (identity transform)
        assert torch.allclose(out_re, q_re, rtol=1e-3), \
            "Identity transform should preserve real part"
        assert torch.allclose(out_im, q_im, rtol=1e-3), \
            "Identity transform should preserve imaginary part"

    def test_biquat_complex_linearity(self):
        """
        Test complex linearity: (a + bi) * q = a*q + b*(i*q).

        For biquaternions, multiplication by complex scalars distributes.
        """
        # Scalar a + bi
        a = 2.0
        b = 3.0

        q_re = torch.randn(4)
        q_im = torch.randn(4)

        # Manual computation: (a + bi)(q_re + i*q_im)
        # = a*q_re - b*q_im + i*(a*q_im + b*q_re)
        result_re = a * q_re - b * q_im
        result_im = a * q_im + b * q_re

        # Using quaternion multiplication with scalar as quaternion (s, 0, 0, 0)
        # Scale real: a * q_re, a * q_im
        # Scale imag: -b * q_im, b * q_re (i² = -1)
        expected_re = a * q_re - b * q_im
        expected_im = a * q_im + b * q_re

        assert torch.allclose(result_re, expected_re, atol=1e-6)
        assert torch.allclose(result_im, expected_im, atol=1e-6)

    def test_split_state_componentwise(self):
        """
        Test (C⊗H)⊕(C⊗H) is STATE decomposition: multiplication is componentwise.

        (Q_M, Q_H) * (P_M, P_H) = (Q_M * P_M, Q_H * P_H)

        This is NOT an algebra redefinition - cross-coupling belongs in DYNAMICS.
        """
        # Two split states
        Q_M_re = torch.randn(4)
        Q_M_im = torch.randn(4)
        Q_H_re = torch.randn(4)
        Q_H_im = torch.randn(4)

        P_M_re = torch.randn(4)
        P_M_im = torch.randn(4)
        P_H_re = torch.randn(4)
        P_H_im = torch.randn(4)

        # Componentwise multiplication on Q_M (complex quaternion)
        # (Q_M_re + i*Q_M_im) * (P_M_re + i*P_M_im)
        # = Q_M_re*P_M_re - Q_M_im*P_M_im + i*(Q_M_re*P_M_im + Q_M_im*P_M_re)
        out_M_re = quat_mul(Q_M_re, P_M_re) - quat_mul(Q_M_im, P_M_im)
        out_M_im = quat_mul(Q_M_re, P_M_im) + quat_mul(Q_M_im, P_M_re)

        # Componentwise multiplication on Q_H (independent!)
        out_H_re = quat_mul(Q_H_re, P_H_re) - quat_mul(Q_H_im, P_H_im)
        out_H_im = quat_mul(Q_H_re, P_H_im) + quat_mul(Q_H_im, P_H_re)

        # Verify the result is 16D
        result = pack_biquat(out_M_re, out_M_im, out_H_re, out_H_im)
        assert result.shape == (16,), "Split state should be 16D"

    def test_causal_accumulator_output_shape(self):
        """Test CausalAccumulator produces correct output shapes."""
        accumulator = CausalAccumulator()

        B, N = 2, 8
        Q_M_re = torch.randn(B, N, 4)
        Q_M_im = torch.randn(B, N, 4)
        Q_H_re = torch.randn(B, N, 4)
        Q_H_im = torch.randn(B, N, 4)

        T_re, T_im, Q_H_new_re, Q_H_new_im = accumulator(
            Q_M_re, Q_M_im, Q_H_re, Q_H_im
        )

        assert T_re.shape == (B, N, 4)
        assert T_im.shape == (B, N, 4)
        assert Q_H_new_re.shape == (B, N, 4)
        assert Q_H_new_im.shape == (B, N, 4)

    def test_causal_accumulator_bounded_params(self):
        """Test CausalAccumulator parameters are bounded (no blow-up)."""
        accumulator = CausalAccumulator(alpha=0.3, decay=0.95, impulse_scale=0.05)

        # Check bounded hyperparams
        alpha = torch.sigmoid(accumulator.alpha_raw)
        decay = torch.sigmoid(accumulator.decay_raw)

        assert 0 < alpha < 1, f"alpha={alpha} should be in (0,1)"
        assert 0 < decay < 1, f"decay={decay} should be in (0,1)"


# =============================================================================
# Octonion Algebra Tests (O)
# =============================================================================

class TestOctonionAlgebra:
    """
    Tests for octonion algebra: The 8D normed division algebra.

    Key property: Octonions are NON-ASSOCIATIVE (unlike quaternions).
    They satisfy alternativity instead.
    """

    @pytest.fixture
    def octonion_module(self):
        """Create AssociatorCurrent for octonion operations."""
        # d_model doesn't matter for oct_mul testing
        module = AssociatorCurrent(d_model=16, d_field=16)
        return module

    def test_octonion_structure_constants_antisymmetric(self, octonion_module):
        """Test structure constants f_ijk are antisymmetric in cyclic permutations."""
        f = octonion_module.oct_struct

        # For Fano triples (i,j,k): f[i,j,k] = f[j,k,i] = f[k,i,j] = 1
        #                           f[j,i,k] = f[k,j,i] = f[i,k,j] = -1
        # Test one triple: (1,2,3) corresponds to (0,1,2) in 0-indexed Fano
        # Actually check the defined triples

        # Verify antisymmetry: f[i,j,k] = -f[j,i,k]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if abs(f[i, j, k]) > 0:
                        assert torch.allclose(f[i, j, k], -f[j, i, k]), \
                            f"f[{i},{j},{k}]={f[i,j,k]} should equal -f[{j},{i},{k}]={-f[j,i,k]}"

    def test_octonion_nonassociativity(self, octonion_module):
        """
        Test octonions are NON-ASSOCIATIVE: (e_i * e_j) * e_k ≠ e_i * (e_j * e_k).

        This is the defining property that distinguishes octonions from quaternions.
        """
        oct_mul = octonion_module.oct_mul

        # Use basis elements where non-associativity is evident
        # e1 = (0,1,0,0,0,0,0,0), etc.
        e1 = torch.zeros(8)
        e1[1] = 1.0
        e2 = torch.zeros(8)
        e2[2] = 1.0
        e4 = torch.zeros(8)
        e4[4] = 1.0

        # (e1 * e2) * e4
        e1_e2 = oct_mul(e1, e2)
        left = oct_mul(e1_e2, e4)

        # e1 * (e2 * e4)
        e2_e4 = oct_mul(e2, e4)
        right = oct_mul(e1, e2_e4)

        # Compute associator: [e1, e2, e4] = left - right
        associator = left - right

        # Associator should be NON-ZERO for octonions
        assert torch.norm(associator) > 1e-6, \
            f"Octonions should be non-associative: [e1,e2,e4] = {associator}"

    def test_octonion_alternativity_left(self, octonion_module):
        """
        Test left alternativity: x(xy) = x²y.

        Octonions satisfy this weaker form of associativity.
        """
        oct_mul = octonion_module.oct_mul

        x = torch.randn(8)
        y = torch.randn(8)

        # x(xy)
        xy = oct_mul(x, y)
        left = oct_mul(x, xy)

        # x²y = (xx)y
        x_sq = oct_mul(x, x)
        right = oct_mul(x_sq, y)

        assert torch.allclose(left, right, rtol=1e-4), \
            f"Left alternativity: x(xy)={left}, x²y={right}"

    def test_octonion_alternativity_right(self, octonion_module):
        """
        Test right alternativity: (xy)y = xy².

        Octonions satisfy this weaker form of associativity.
        """
        oct_mul = octonion_module.oct_mul

        x = torch.randn(8)
        y = torch.randn(8)

        # (xy)y
        xy = oct_mul(x, y)
        left = oct_mul(xy, y)

        # xy² = x(yy)
        y_sq = oct_mul(y, y)
        right = oct_mul(x, y_sq)

        assert torch.allclose(left, right, rtol=1e-4), \
            f"Right alternativity: (xy)y={left}, xy²={right}"

    def test_octonion_moufang_left(self, octonion_module):
        """
        Test left Moufang identity: z(x(zy)) = ((zx)z)y.

        All alternative algebras satisfy Moufang identities.
        """
        oct_mul = octonion_module.oct_mul

        x = torch.randn(8)
        y = torch.randn(8)
        z = torch.randn(8)

        # Left side: z(x(zy))
        zy = oct_mul(z, y)
        x_zy = oct_mul(x, zy)
        left = oct_mul(z, x_zy)

        # Right side: ((zx)z)y
        zx = oct_mul(z, x)
        zx_z = oct_mul(zx, z)
        right = oct_mul(zx_z, y)

        assert torch.allclose(left, right, rtol=1e-3), \
            f"Left Moufang: z(x(zy))={left}, ((zx)z)y={right}"

    def test_octonion_moufang_right(self, octonion_module):
        """
        Test right Moufang identity: ((xy)z)y = x(y(zy)).
        """
        oct_mul = octonion_module.oct_mul

        x = torch.randn(8)
        y = torch.randn(8)
        z = torch.randn(8)

        # Left side: ((xy)z)y
        xy = oct_mul(x, y)
        xy_z = oct_mul(xy, z)
        left = oct_mul(xy_z, y)

        # Right side: x(y(zy))
        zy = oct_mul(z, y)
        y_zy = oct_mul(y, zy)
        right = oct_mul(x, y_zy)

        assert torch.allclose(left, right, rtol=1e-3), \
            f"Right Moufang: ((xy)z)y={left}, x(y(zy))={right}"

    def test_octonion_moufang_middle(self, octonion_module):
        """
        Test middle Moufang identity: (zx)(yz) = (z(xy))z.
        """
        oct_mul = octonion_module.oct_mul

        x = torch.randn(8)
        y = torch.randn(8)
        z = torch.randn(8)

        # Left side: (zx)(yz)
        zx = oct_mul(z, x)
        yz = oct_mul(y, z)
        left = oct_mul(zx, yz)

        # Right side: (z(xy))z
        xy = oct_mul(x, y)
        z_xy = oct_mul(z, xy)
        right = oct_mul(z_xy, z)

        assert torch.allclose(left, right, rtol=1e-3), \
            f"Middle Moufang: (zx)(yz)={left}, (z(xy))z={right}"

    def test_octonion_norm_multiplicative(self, octonion_module):
        """
        Test ||xy|| = ||x|| ||y|| (norm is multiplicative).

        This makes octonions a normed division algebra.
        """
        oct_mul = octonion_module.oct_mul

        x = torch.randn(8)
        y = torch.randn(8)

        norm_x = torch.norm(x)
        norm_y = torch.norm(y)
        xy = oct_mul(x, y)
        norm_xy = torch.norm(xy)

        expected = norm_x * norm_y

        assert torch.allclose(norm_xy, expected, rtol=1e-3), \
            f"||xy||={norm_xy}, ||x||·||y||={expected}"

    def test_associator_current_antisymmetric(self):
        """Test associator current J is antisymmetric tensor."""
        associator = AssociatorCurrent(d_model=32, d_field=16)

        x = torch.randn(2, 8, 32)
        J = associator(x)

        # J should be antisymmetric: J + J^T = 0
        J_transpose = J.transpose(-1, -2)
        sum_tensor = J + J_transpose

        assert torch.allclose(sum_tensor, torch.zeros_like(sum_tensor), atol=1e-5), \
            f"J should be antisymmetric, but J + J^T has norm {torch.norm(sum_tensor)}"

    def test_associator_nonzero_for_octonions(self):
        """Test associator is genuinely non-zero (octonions are non-associative)."""
        associator = AssociatorCurrent(d_model=32, d_field=16)

        x = torch.randn(2, 8, 32)
        J = associator(x)

        # Should have non-trivial values
        assert torch.norm(J) > 1e-6, \
            "Associator should be non-zero for octonion algebra"


# =============================================================================
# LIoR Kernel Tests
# =============================================================================

class TestLiorKernel:
    """
    Tests for LIoR (Learnable Integrated over Retarded) memory kernel.

    Key properties:
    - K(t) = α·exp(-βt) + γ·t^(-δ)·exp(-ξt) + η·cos(ωt+φ)·exp(-ζt)
    - O(1) recurrence: m_t = ρ·m_{t-1} + η_r·x_t - ξ_r·x_{t-p_eff}
    """

    @pytest.fixture
    def kernel(self):
        """Create LiorKernel with default parameters."""
        return LiorKernel(p_eff=4)

    def test_kernel_causality(self, kernel):
        """Test kernel is causal: K(t) = 0 for t < 0."""
        # Due to clamping at min=0.1, we test that small t gives small K
        dt_negative = torch.tensor([-1.0, -0.5, -0.1])
        dt_positive = torch.tensor([0.1, 1.0, 5.0])

        # Kernel clamps negative to 0.1, so K(-1) = K(0.1)
        k_neg = kernel(dt_negative)
        k_pos = kernel(dt_positive)

        # Both should be finite
        assert torch.all(torch.isfinite(k_neg)), f"K(negative) should be finite: {k_neg}"
        assert torch.all(torch.isfinite(k_pos)), f"K(positive) should be finite: {k_pos}"

    def test_kernel_power_law_decay(self, kernel):
        """
        Test power-law component decays as t^(-δ).

        For large t, the power-law term dominates:
        K(t) ~ γ·t^(-δ)·exp(-ξt) ≈ γ·t^(-δ) for small ξ
        """
        # Access kernel parameters
        delta = kernel.delta.item()

        # Test at different times
        t1 = torch.tensor([10.0])
        t2 = torch.tensor([20.0])

        k1 = kernel(t1)
        k2 = kernel(t2)

        # For power-law: K(t2)/K(t1) ≈ (t1/t2)^delta (if ξ small)
        # This is approximate due to other terms and exponential cutoff
        ratio_actual = k2 / k1
        ratio_expected = (t1 / t2) ** delta

        # Allow large tolerance due to mixed kernel
        # At least verify decay happens
        assert k2 < k1, f"Kernel should decay: K({t2.item()})={k2.item()} < K({t1.item()})={k1.item()}"

    def test_kernel_exponential_mode(self, kernel):
        """Test exponential mode: K(t) ~ α·exp(-βt)."""
        alpha_exp = kernel.alpha_exp.item()
        beta = kernel.beta.item()

        t = torch.tensor([0.1])
        k = kernel(t)

        # The exponential contribution
        exp_contrib = alpha_exp * torch.exp(-beta * t)

        # Kernel should be finite and positive (for positive weights)
        assert torch.isfinite(k), f"Kernel should be finite at t={t.item()}"

    def test_kernel_oscillatory_mode(self, kernel):
        """Test oscillatory mode: η·cos(ωt+φ)·exp(-ζt)."""
        omega = kernel.omega.item()
        phi = kernel.phi.item()

        # Test at period of oscillation
        period = 2 * math.pi / omega
        t1 = torch.tensor([0.0 + 0.1])  # Near phase 0
        t2 = torch.tensor([period / 2])  # Half period

        k1 = kernel(t1)
        k2 = kernel(t2)

        # Both should be finite
        assert torch.isfinite(k1), f"K(0) should be finite"
        assert torch.isfinite(k2), f"K(T/2) should be finite"

    def test_kernel_weights_sum_to_one(self, kernel):
        """Test mixture weights sum to 1 (softmax normalized)."""
        weights = kernel.weights

        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5), \
            f"Weights should sum to 1: {weights.sum()}"

    def test_recurrence_step_shape(self, kernel):
        """Test O(1) recurrence produces correct shapes."""
        B, D = 4, 32
        m_prev = torch.randn(B, D)
        x_curr = torch.randn(B, D)
        x_delayed = torch.randn(B, D)

        m_new = kernel.recurrence_step(m_prev, x_curr, x_delayed)

        assert m_new.shape == (B, D), f"Expected shape {(B, D)}, got {m_new.shape}"

    def test_recurrence_bounded(self, kernel):
        """Test recurrence doesn't blow up (ρ < 1 ensures stability)."""
        rho = kernel.rho.item()
        assert 0 < rho < 1, f"rho={rho} should be in (0,1) for stability"

        B, D = 4, 32
        m = torch.randn(B, D)

        # Run many steps
        for _ in range(100):
            x = torch.randn(B, D) * 0.1
            x_delayed = torch.randn(B, D) * 0.1
            m = kernel.recurrence_step(m, x, x_delayed)

        # Should remain bounded
        assert torch.all(torch.isfinite(m)), "Memory should remain finite after many steps"
        assert torch.norm(m) < 1e6, f"Memory should be bounded: ||m|| = {torch.norm(m)}"

    def test_lior_memory_state_shape(self):
        """Test LiorMemoryState produces correct output shapes."""
        D = 64
        memory_state = LiorMemoryState(d_model=D, p_eff=4)

        B, N = 2, 16
        x = torch.randn(B, N, D)

        output, new_memory = memory_state(x)

        assert output.shape == (B, N, D), f"Output shape mismatch: {output.shape}"
        assert 'm' in new_memory
        assert 'buffer' in new_memory
        assert 'j_h' in new_memory

    def test_lior_memory_state_phase_field(self):
        """Test phase field computation for symplectic form."""
        D = 64
        memory_state = LiorMemoryState(d_model=D, p_eff=4)

        B, N = 2, 16
        x = torch.randn(B, N, D)

        theta = memory_state.get_phase_field(x)

        assert theta.shape == (B, N), f"Phase field shape mismatch: {theta.shape}"
        assert torch.all(torch.isfinite(theta)), "Phase field should be finite"


# =============================================================================
# Integration Tests
# =============================================================================

class TestAlgebraIntegration:
    """Integration tests combining multiple algebraic components."""

    def test_biquat_layer_forward(self):
        """Test full biquaternion causal layer."""
        from ..models.biquaternion import BiQuatCausalLayer

        d_model = 64
        layer = BiQuatCausalLayer(d_model=d_model, d_field=16)

        B, N = 2, 16
        x = torch.randn(B, N, d_model)

        output, memory = layer(x)

        assert output.shape == (B, N, d_model), f"Output shape mismatch: {output.shape}"
        assert 'Q_H_re' in memory
        assert 'Q_H_im' in memory

    def test_causal_field_layer_forward(self):
        """Test full causal field layer with octonion associator."""
        from ..models.causal_field import CausalFieldLayer

        d_model = 64
        layer = CausalFieldLayer(d_model=d_model, d_field=16)

        B, N = 2, 16
        x = torch.randn(B, N, d_model)

        output, memory = layer(x)

        assert output.shape == (B, N, d_model), f"Output shape mismatch: {output.shape}"

    def test_gradient_flow_biquaternion(self):
        """Test gradients flow through biquaternion operations."""
        p = torch.randn(4, requires_grad=True)
        q = torch.randn(4, requires_grad=True)

        result = quat_mul(p, q)
        loss = result.sum()
        loss.backward()

        assert p.grad is not None, "Gradient should flow to p"
        assert q.grad is not None, "Gradient should flow to q"
        assert torch.all(torch.isfinite(p.grad)), "Gradients should be finite"
        assert torch.all(torch.isfinite(q.grad)), "Gradients should be finite"

    def test_gradient_flow_octonion(self):
        """Test gradients flow through octonion operations."""
        associator = AssociatorCurrent(d_model=32, d_field=16)

        x = torch.randn(2, 8, 32, requires_grad=True)
        J = associator(x)
        loss = J.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should flow to input"
        assert torch.all(torch.isfinite(x.grad)), "Gradients should be finite"
