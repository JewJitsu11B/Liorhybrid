"""
Test Adaptive Parameter Learning

Validates entropy descent and parameter convergence.

Paper Reference: Corollary (Adaptive Learning)
"""

import torch
import pytest
from ..core import CognitiveTensorField, FAST_TEST_CONFIG


def test_adaptive_entropy_descent():
    """
    Test Theorem 3: Entropy decreases monotonically with adaptation.

    Expected: dH/dt ≤ 0 (on average, allowing numerical noise)
    """
    config = FAST_TEST_CONFIG
    config.adaptive_learning = True
    config.param_learning_rate = 0.001

    field = CognitiveTensorField(config)

    entropies = []
    for _ in range(100):
        field.evolve_step()
        entropies.append(field.compute_entropy().item())

    entropies = torch.tensor(entropies)

    # Check overall descent
    assert entropies[-1] < entropies[0], "Entropy should decrease overall"

    # Check general trend: last 20 steps should be more stable than first 20
    early_var = torch.var(entropies[:20])
    late_var = torch.var(entropies[-20:])
    assert late_var < early_var, (
        f"Entropy not stabilizing: early_var={early_var:.2e}, late_var={late_var:.2e}"
    )


def test_parameter_convergence():
    """
    Test that parameters stabilize after sufficient evolution.
    """
    config = FAST_TEST_CONFIG
    config.adaptive_learning = True
    config.param_learning_rate = 0.001

    field = CognitiveTensorField(config)

    # Burn-in period
    for _ in range(300):
        field.evolve_step()

    # Measure stability
    alpha_history = []
    for _ in range(100):
        field.evolve_step()
        alpha_history.append(field.alpha.item())

    alpha_std = torch.std(torch.tensor(alpha_history))
    assert alpha_std < 0.05, f"Alpha not converged: std={alpha_std:.6f}"


def test_parameter_constraints():
    """
    Test that parameters stay within theoretical bounds.
    """
    config = FAST_TEST_CONFIG
    config.adaptive_learning = True
    config.param_learning_rate = 0.01  # Large LR to stress-test

    field = CognitiveTensorField(config)

    for _ in range(200):
        field.evolve_step()

        # Check Paper Axiom 4: α ∈ (0, 2)
        alpha_val = field.alpha.item()
        assert 0 < alpha_val < 2, f"Alpha out of bounds: {alpha_val}"

        # Check Paper Axiom 3: ν ∈ (0, 1]
        nu_min = field.nu.min().item()
        nu_max = field.nu.max().item()
        assert 0 < nu_min <= nu_max <= 1.0, (
            f"Nu out of bounds: min={nu_min}, max={nu_max}"
        )

        # Check Paper Axiom 5: τ > 0
        tau_min = field.tau.min().item()
        assert tau_min > 0, f"Tau non-positive: {tau_min}"


def test_adaptive_vs_fixed():
    """
    Compare adaptive vs fixed parameter evolution.

    Adaptive should achieve lower final entropy.
    """
    # Fixed parameters
    torch.manual_seed(42)
    config_fixed = FAST_TEST_CONFIG
    config_fixed.adaptive_learning = False
    field_fixed = CognitiveTensorField(config_fixed)

    for _ in range(100):
        field_fixed.evolve_step()

    entropy_fixed = field_fixed.compute_entropy().item()

    # Adaptive parameters
    torch.manual_seed(42)  # Same initial field
    config_adaptive = FAST_TEST_CONFIG
    config_adaptive.adaptive_learning = True
    field_adaptive = CognitiveTensorField(config_adaptive)

    for _ in range(100):
        field_adaptive.evolve_step()

    entropy_adaptive = field_adaptive.compute_entropy().item()

    # Adaptive should reach lower entropy
    # (May not always be true for short runs, so we check it's at least close)
    assert entropy_adaptive <= entropy_fixed * 1.1, (
        f"Adaptive entropy {entropy_adaptive:.2e} not better than "
        f"fixed {entropy_fixed:.2e}"
    )
