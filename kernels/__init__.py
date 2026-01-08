"""
Kernels for Bayesian Cognitive Field Evolution

This module contains the three main operator kernels from the paper:
- hamiltonian: H[T] (kinetic + potential)
- bayesian: Λ_QR[T] (recursive Bayesian update)
- fractional_memory: Λ_F[T] (power-law memory damping)

Paper Reference: Equation (1) - Master equation
"""

from .hamiltonian import spatial_laplacian, hamiltonian_evolution, create_potential
from .bayesian import (
    compute_evidence_weights,
    bayesian_posterior,
    bayesian_recursive_term
)
from .fractional_memory import (
    fractional_kernel_weights,
    fractional_memory_term,
    fractional_memory_weight,
    update_history_buffer
)

__all__ = [
    # Hamiltonian
    'spatial_laplacian',
    'hamiltonian_evolution',
    'create_potential',

    # Bayesian
    'compute_evidence_weights',
    'bayesian_posterior',
    'bayesian_recursive_term',

    # Fractional Memory
    'fractional_kernel_weights',
    'fractional_memory_term',
    'fractional_memory_weight',
    'update_history_buffer',
]
