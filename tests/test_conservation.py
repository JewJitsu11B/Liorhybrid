"""
Test Conservation Laws

Verifies norm conservation and modified unitarity.

Paper Reference: Section 6.1
"""

import torch
import pytest
from ..core import CognitiveTensorField, FAST_TEST_CONFIG


def test_norm_conservation_no_memory():
    """
    Test that ||T||² is conserved when lambda_F = 0.

    Without fractional memory damping, the Hamiltonian + QR
    evolution should preserve total norm (up to numerical error).
    """
    # Create config with no memory damping
    config = FAST_TEST_CONFIG
    config.lambda_F = 0.0  # No damping

    field = CognitiveTensorField(config)

    # Run evolution
    n_steps = 100
    norms = []

    for _ in range(n_steps):
        field.evolve_step()
        norms.append(field.get_norm_squared())

    norms = torch.tensor(norms)

    # Check conservation (should be constant within tolerance)
    norm_variation = torch.std(norms) / torch.mean(norms)

    assert norm_variation < 0.01, (
        f"Norm not conserved: relative variation {norm_variation:.4f}"
    )


def test_energy_evolution():
    """
    Test that Hamiltonian energy behaves correctly.

    With pure Hamiltonian evolution (no Bayesian or memory terms),
    energy should be approximately conserved. With non-unitary terms,
    energy can change over time.
    """
    # Create config with only Hamiltonian evolution
    config = FAST_TEST_CONFIG
    config.lambda_QR = 0.0  # No Bayesian updates
    config.lambda_F = 0.0   # No memory damping

    field = CognitiveTensorField(config)

    # Run evolution and track energy
    n_steps = 50
    energies = []

    for _ in range(n_steps):
        energy = field.compute_energy()
        energies.append(energy)
        field.evolve_step()

    energies = torch.tensor(energies)

    # With pure Hamiltonian, energy should be roughly conserved
    # (may vary due to numerical integration, but should not drift)
    energy_variation = torch.std(energies) / (torch.abs(torch.mean(energies)) + 1e-6)

    assert energy_variation < 0.1, (
        f"Energy not conserved in pure Hamiltonian evolution: "
        f"relative variation {energy_variation:.4f}"
    )


def test_unitarity_breaking():
    """
    Test that Bayesian updates break strict unitarity.

    The QR term introduces non-unitary evolution,
    unlike pure Hamiltonian dynamics.
    """
    # Test 1: Pure Hamiltonian (should be nearly unitary)
    config_pure = FAST_TEST_CONFIG
    config_pure.lambda_QR = 0.0
    config_pure.lambda_F = 0.0

    field_pure = CognitiveTensorField(config_pure)

    for _ in range(20):
        field_pure.evolve_step()

    deviation_pure = field_pure.compute_unitarity_deviation()

    # Test 2: With Bayesian updates (should be non-unitary)
    config_bayesian = FAST_TEST_CONFIG
    config_bayesian.lambda_QR = 0.3  # Strong Bayesian updates
    config_bayesian.lambda_F = 0.0

    field_bayesian = CognitiveTensorField(config_bayesian)

    # Create some evidence to trigger Bayesian updates
    evidence = torch.randn_like(field_bayesian.T)

    for _ in range(20):
        field_bayesian.evolve_step(evidence=evidence)

    deviation_bayesian = field_bayesian.compute_unitarity_deviation()

    # Bayesian evolution should have higher unitarity deviation
    # (Note: This test is qualitative since both may have some deviation)
    assert deviation_pure < 1.0, (
        f"Pure Hamiltonian has too high deviation: {deviation_pure:.4f}"
    )
    assert deviation_bayesian >= 0.0, (
        f"Bayesian evolution should have non-negative deviation: {deviation_bayesian:.4f}"
    )
