"""
Test Potential Landscapes for Hamiltonian Operator

Verifies that non-trivial potentials work correctly.
"""

import torch
import pytest
from ..kernels.hamiltonian import create_potential, hamiltonian_evolution
from ..core import CognitiveTensorField, FAST_TEST_CONFIG


def test_create_harmonic_potential():
    """Test creation of harmonic oscillator potential."""
    spatial_size = (8, 8)
    tensor_dim = 4

    V = create_potential(
        spatial_size=spatial_size,
        tensor_dim=tensor_dim,
        potential_type="harmonic",
        strength=1.0
    )

    # Check shape
    assert V.shape == (8, 8, 4, 4)

    # Check that potential is diagonal (V = v(x,y) * I)
    for x in range(8):
        for y in range(8):
            # Off-diagonal elements should be zero
            V_xy = V[x, y, :, :]
            off_diag = V_xy - torch.diag(torch.diag(V_xy))
            assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)

    # Check that potential increases with distance from center
    center = (4.0, 4.0)
    V_center = V[4, 4, 0, 0].real
    V_corner = V[0, 0, 0, 0].real
    assert V_corner > V_center, "Harmonic potential should increase away from center"


def test_create_gaussian_well():
    """Test creation of Gaussian well potential."""
    V = create_potential(
        spatial_size=(8, 8),
        tensor_dim=4,
        potential_type="gaussian_well",
        strength=1.0
    )

    assert V.shape == (8, 8, 4, 4)

    # Gaussian well should be negative
    V_center = V[4, 4, 0, 0].real
    assert V_center < 0, "Gaussian well should have negative potential"


def test_create_gaussian_barrier():
    """Test creation of Gaussian barrier potential."""
    V = create_potential(
        spatial_size=(8, 8),
        tensor_dim=4,
        potential_type="gaussian_barrier",
        strength=1.0
    )

    assert V.shape == (8, 8, 4, 4)

    # Gaussian barrier should be positive
    V_center = V[4, 4, 0, 0].real
    assert V_center > 0, "Gaussian barrier should have positive potential"


def test_hamiltonian_with_potential():
    """Test that Hamiltonian evolution works with non-zero potential."""
    config = FAST_TEST_CONFIG

    # Create field
    field = CognitiveTensorField(config)

    # Create harmonic potential
    V = create_potential(
        spatial_size=config.spatial_size,
        tensor_dim=config.tensor_dim,
        potential_type="harmonic",
        strength=0.1,
        device=config.device,
        dtype=config.dtype
    )

    # Compute Hamiltonian with potential
    H_T = hamiltonian_evolution(
        field.T,
        hbar_cog=config.hbar_cog,
        m_cog=config.m_cog,
        V=V
    )

    # Check that result has correct shape
    assert H_T.shape == field.T.shape

    # Check that Hamiltonian is not all zeros (has both kinetic and potential)
    assert torch.abs(H_T).sum() > 0


def test_evolution_with_harmonic_potential():
    """Test field evolution in harmonic potential."""
    config = FAST_TEST_CONFIG
    config.lambda_QR = 0.0  # Pure Hamiltonian
    config.lambda_F = 0.0

    field = CognitiveTensorField(config)

    # Create harmonic potential
    V = create_potential(
        spatial_size=config.spatial_size,
        tensor_dim=config.tensor_dim,
        potential_type="harmonic",
        strength=0.1,
        device=config.device,
        dtype=config.dtype
    )

    # Evolve field with potential
    # Note: Currently, the field.evolve_step doesn't directly accept V
    # This test verifies that the Hamiltonian operator accepts V

    # Compute Hamiltonian evolution directly
    for _ in range(10):
        H_T = hamiltonian_evolution(
            field.T,
            hbar_cog=config.hbar_cog,
            m_cog=config.m_cog,
            V=V
        )

        # Manual time step (simplified)
        dT = (config.dt / (1j * config.hbar_cog)) * H_T
        field.T = field.T + dT

    # Check that field evolved
    assert field.T is not None
    assert not torch.isnan(field.T).any()


def test_potential_types():
    """Test that all potential types can be created."""
    potential_types = ["harmonic", "gaussian_well", "gaussian_barrier", "constant", "zero"]

    for pot_type in potential_types:
        V = create_potential(
            spatial_size=(8, 8),
            tensor_dim=4,
            potential_type=pot_type,
            strength=1.0
        )

        assert V.shape == (8, 8, 4, 4)
        assert not torch.isnan(V).any()


def test_invalid_potential_type():
    """Test that invalid potential type raises error."""
    with pytest.raises(ValueError, match="Unknown potential_type"):
        create_potential(
            spatial_size=(8, 8),
            tensor_dim=4,
            potential_type="invalid_type"
        )
