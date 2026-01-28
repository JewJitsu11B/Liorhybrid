"""
Kernels for Bayesian Cognitive Field Evolution

This module contains the main operator kernels from the paper:
- hamiltonian: H[T] (kinetic + potential) with anisotropic metric
- bayesian: Λ_QR[T] (recursive Bayesian update)
- fractional_memory: Λ_F[T] (power-law memory damping)
- metric_context: Validation and tracking for metric operations
- tetrad: Vielbein field connecting metric to Clifford algebra

The tetrad e^a_μ provides the crucial link:
    g_μν = η_ab e^a_μ e^b_ν    (metric from tetrad)
    Γ_μ = e^a_μ γ_a             (Clifford connection)

Paper Reference: Equation (1) - Master equation
"""

from .hamiltonian import (
    spatial_laplacian,
    spatial_laplacian_x,
    spatial_laplacian_y,
    hamiltonian_evolution,
    hamiltonian_evolution_with_metric,
    create_potential
)
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
from .metric_context import MetricContext, metric_context
from .tetrad import (
    Tetrad,
    compute_metric_from_tetrad,
    anisotropic_laplacian_from_tetrad
)

__all__ = [
    # Hamiltonian
    'spatial_laplacian',
    'spatial_laplacian_x',
    'spatial_laplacian_y',
    'hamiltonian_evolution',
    'hamiltonian_evolution_with_metric',
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
    
    # Metric Context Manager
    'MetricContext',
    'metric_context',
    
    # Tetrad (Vielbein)
    'Tetrad',
    'compute_metric_from_tetrad',
    'anisotropic_laplacian_from_tetrad',
]
