"""
Bayesian Cognitive Field

A PyTorch implementation of rank-2 tensor field evolution under
Bayesian recursive dynamics with fractional memory.

Paper Reference: "Bayesian Recursive Operator for Cognitive Field Dynamics"

Main components:
- CognitiveTensorField: Main evolution class
- FieldConfig: Configuration dataclass
- Kernel operators: Hamiltonian, Bayesian update, fractional memory
- Utilities: Metrics, visualization, diagnostics

Quick Start:
    >>> from Liorhybrid import CognitiveTensorField, FAST_TEST_CONFIG
    >>> field = CognitiveTensorField(FAST_TEST_CONFIG)
    >>> for _ in range(100):
    ...     field.evolve_step()
    >>> print(f"Final norm: {field.get_norm_squared():.6f}")
"""

from .core import (
    CognitiveTensorField,
    FieldConfig,
    get_default_config,
    MNIST_CONFIG,
    FAST_TEST_CONFIG
)

from .kernels import (
    hamiltonian_evolution,
    bayesian_recursive_term,
    fractional_memory_term,
    spatial_laplacian
)

from .utils import (
    compute_norm_conservation,
    compute_local_correlation,
    compute_effective_dimension
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'CognitiveTensorField',
    'FieldConfig',
    'get_default_config',
    'MNIST_CONFIG',
    'FAST_TEST_CONFIG',

    # Kernels
    'hamiltonian_evolution',
    'bayesian_recursive_term',
    'fractional_memory_term',
    'spatial_laplacian',

    # Utils
    'compute_norm_conservation',
    'compute_local_correlation',
    'compute_effective_dimension',
]
