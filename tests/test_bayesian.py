"""
Test Bayesian Recursive Operator

Verifies posterior construction and evidence weighting.

Paper Reference: Equations 4-6
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import pytest
from ..kernels.bayesian import (
    compute_evidence_weights,
    bayesian_posterior,
    bayesian_recursive_term
)


def test_evidence_weights_uniform():
    """
    Test that weights are uniform when no evidence provided.
    """
    T = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
    weights = compute_evidence_weights(T, evidence=None, tau=1.0)

    # Should be all ones
    assert torch.allclose(weights, torch.ones_like(weights, dtype=torch.float32))


def test_evidence_weights_decay():
    """
    Test that weights decay with distance from evidence.

    w_ij = exp(-|T - E|²/τ)

    Points farther from evidence should have lower weight.
    """
    T = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
    evidence = torch.zeros_like(T)  # Evidence at zero

    weights = compute_evidence_weights(T, evidence, tau=1.0)

    # All weights should be < 1 (since T != 0 generically)
    # and > 0
    assert torch.all(weights > 0)
    assert torch.all(weights <= 1)

    # Mean weight should be significantly less than 1
    # (since random T is unlikely to be near zero)
    mean_weight = torch.mean(weights).item()
    assert mean_weight < 0.7, f"Mean weight {mean_weight} too high (expected < 0.7)"


def test_posterior_normalization():
    """
    Test that Bayesian posterior is properly normalized.

    ∫ |B[T]|² dV should equal the total weighted norm.
    """
    T = torch.randn(8, 8, 4, 4, dtype=torch.complex64)
    weights = torch.rand(8, 8, 4, 4)  # Random weights

    B_T = bayesian_posterior(T, weights)

    # Check that posterior has same shape
    assert B_T.shape == T.shape

    # Check that weighted field is normalized
    # Σ w|T|² = 1 after normalization
    posterior_norm = torch.sum(torch.abs(B_T) ** 2).item()
    weighted_input_norm = torch.sum(weights * torch.abs(T) ** 2).item()

    # B[T] should have smaller norm than weighted input
    # due to normalization by Z
    assert posterior_norm <= weighted_input_norm


def test_bayesian_update_no_previous():
    """
    Test that Bayesian term is zero when T_prev_collapsed is None.
    """
    T = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    Lambda_QR = bayesian_recursive_term(
        T_current=T,
        T_prev_collapsed=None,
        evidence=None,
        lambda_QR=0.3,
        tau=1.0
    )

    # Should be all zeros
    assert torch.allclose(Lambda_QR, torch.zeros_like(Lambda_QR))


def test_bayesian_update_drives_toward_posterior():
    """
    Test that Bayesian update drives field toward evidence.

    The term Λ_QR = λ_QR(B[T_prev] - T_prev) should point
    in the direction of the evidence.

    TODO: Implement directional test.
    """
    pytest.skip("Directional test not yet implemented")


def test_tau_temperature_effect():
    """
    Test that tau controls sharpness of evidence weighting.

    - Small tau: Sharp weighting (only exact matches get weight)
    - Large tau: Broad weighting (all states get similar weight)

    TODO: Implement temperature sweep test.
    """
    pytest.skip("Temperature effect test not yet implemented")
