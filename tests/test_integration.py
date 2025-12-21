"""
Integration Tests for Full Field Evolution

Tests the complete evolution loop with all operators.

Paper Reference: Algorithm 1
"""

import torch
import pytest
from ..core import CognitiveTensorField, FAST_TEST_CONFIG, MNIST_CONFIG


def test_basic_evolution():
    """
    Test that field evolves without errors for multiple steps.
    """
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)

    # Run 50 steps
    n_steps = 50
    for _ in range(n_steps):
        T = field.evolve_step()

        # Check output shape
        assert T.shape == (
            config.spatial_size[0],
            config.spatial_size[1],
            config.tensor_dim,
            config.tensor_dim
        )

        # Check for NaNs
        assert not torch.isnan(T).any(), "NaN detected in field"

        # Check for Infs
        assert not torch.isinf(T).any(), "Inf detected in field"


def test_evolution_with_evidence():
    """
    Test evolution with external evidence provided.
    """
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)

    # Create random evidence
    evidence = torch.randn(
        *config.spatial_size,
        config.tensor_dim,
        config.tensor_dim,
        dtype=torch.complex64,
        device=field.device
    )

    # Run evolution with evidence
    n_steps = 20
    for _ in range(n_steps):
        T = field.evolve_step(evidence=evidence)

        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()


def test_evolution_with_external_input():
    """
    Test evolution with external input J(t).
    """
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)

    # Create external input (forcing term)
    J = torch.randn(
        *config.spatial_size,
        config.tensor_dim,
        config.tensor_dim,
        dtype=torch.complex64,
        device=field.device
    ) * 0.01  # Small forcing

    # Run evolution
    n_steps = 20
    for _ in range(n_steps):
        T = field.evolve_step(external_input=J)

        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()


def test_history_buffer():
    """
    Test that history buffer maintains correct length.
    """
    config = FAST_TEST_CONFIG
    config.memory_window = 10

    field = CognitiveTensorField(config)

    # Run more steps than memory window
    n_steps = 25
    for _ in range(n_steps):
        field.evolve_step()

    # History should be capped at memory_window
    assert len(field.history) == config.memory_window


def test_time_tracking():
    """
    Test that time and step count are correctly tracked.
    """
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)

    n_steps = 50
    for i in range(n_steps):
        field.evolve_step()

        # Check step count
        assert field.step_count == i + 1

        # Check time (should be dt * steps)
        expected_time = config.dt * (i + 1)
        assert abs(field.t - expected_time) < 1e-10


def test_mnist_config():
    """
    Test that MNIST config initializes correctly.

    This is a larger field, so just test initialization
    and a few steps.
    """
    config = MNIST_CONFIG
    field = CognitiveTensorField(config)

    # Check field shape
    assert field.T.shape == (28, 28, 16, 16)

    # Run a few steps
    for _ in range(5):
        T = field.evolve_step()
        assert not torch.isnan(T).any()


def test_device_placement():
    """
    Test that field is placed on correct device.
    """
    config = FAST_TEST_CONFIG

    # Test CPU
    config.device = 'cpu'
    field_cpu = CognitiveTensorField(config)
    assert field_cpu.T.device.type == 'cpu'

    # Test CUDA (if available)
    if torch.cuda.is_available():
        config.device = 'cuda'
        field_cuda = CognitiveTensorField(config)
        assert field_cuda.T.device.type == 'cuda'


def test_reproducibility():
    """
    Test that evolution is reproducible with same random seed.
    """
    torch.manual_seed(42)
    config = FAST_TEST_CONFIG
    field1 = CognitiveTensorField(config)

    # Run evolution
    n_steps = 20
    for _ in range(n_steps):
        field1.evolve_step()

    final_state_1 = field1.T.clone()

    # Reset and run again with same seed
    torch.manual_seed(42)
    field2 = CognitiveTensorField(config)

    for _ in range(n_steps):
        field2.evolve_step()

    final_state_2 = field2.T.clone()

    # Should be identical
    assert torch.allclose(final_state_1, final_state_2, atol=1e-6)
