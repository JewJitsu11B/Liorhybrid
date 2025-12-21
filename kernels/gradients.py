"""
Hamiltonian Gradient Computation

Computes ∇H[T] = δH/δT (functional derivative) for gradient-based field updates.

In Bayesian formulation:
    dψ/dt = -γ∇H[ψ] + noise

where ∇H is the gradient of the Hamiltonian energy functional.

This replaces additive field updates with multiplicative probability updates,
ensuring energy conservation and preventing field explosion.
"""

import torch


def compute_hamiltonian_gradient(
    T: torch.Tensor,
    H_T: torch.Tensor,
    hbar_cog: float = 0.1
) -> torch.Tensor:
    """
    Compute gradient of Hamiltonian functional ∇H[T].

    The Hamiltonian H[T] = -(ℏ²/2m)∇²T + V·T defines an energy functional.
    Its gradient drives the Bayesian field update.

    For the cognitive field, we use the Hamiltonian evolution term H_T
    as the effective force that would drive changes. The gradient is:

        ∇H = H_T / ℏ  (in natural units)

    This represents the "force" direction in the field's configuration space.

    Args:
        T: Current field state (N_x, N_y, D, D) complex
        H_T: Hamiltonian evolution term = -(ℏ²/2m)∇²T + V·T
        hbar_cog: Cognitive Planck constant

    Returns:
        grad_H: Gradient ∇H[T] (same shape as T)

    Note:
        In the Schrödinger/Liouville formulation, the evolution is:
            ∂_t T = (1/iℏ)[H, T]

        In the Langevin/Fokker-Planck formulation, it's:
            ∂_t T = -∇F[T] + noise

        We use H_T as the effective gradient since it represents the
        direction of energy increase in the field.
    """
    # The Hamiltonian term H_T is already the "force" from the energy landscape
    # Divide by ℏ to get the gradient in natural units
    # (The 1/iℏ factor in Schrödinger becomes 1/ℏ in real Langevin dynamics)
    grad_H = H_T / hbar_cog

    return grad_H


def compute_free_energy_gradient(
    T: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    """
    Compute gradient of free energy functional F[T].

    Alternative gradient computation based on free energy:
        F[T] = ⟨H[T]⟩ - (1/β)S[T]

    where S[T] is the von Neumann entropy.

    The gradient is:
        ∇F = ∇⟨H⟩ - (1/β)∇S

    For now, this is a placeholder. The main gradient computation
    uses the Hamiltonian term directly.

    Args:
        T: Current field state (N_x, N_y, D, D)
        beta: Inverse temperature (controls exploration vs exploitation)

    Returns:
        grad_F: Free energy gradient (same shape as T)
    """
    # Placeholder: compute gradient via autograd when free energy is defined
    # For now, use simple energy gradient

    # Energy gradient (L2 regularization term)
    # Encourages smaller field magnitudes
    if T.is_complex():
        grad_E = 2.0 * T.conj()  # ∂|T|²/∂T* = T
    else:
        grad_E = 2.0 * T  # ∂|T|²/∂T = 2T

    return grad_E


def compute_gradient_norm(grad: torch.Tensor) -> float:
    """
    Compute norm of gradient for monitoring.

    Args:
        grad: Gradient tensor (any shape)

    Returns:
        Gradient norm (scalar)
    """
    if grad.is_complex():
        norm = torch.sqrt(torch.sum(torch.abs(grad) ** 2))
    else:
        norm = torch.norm(grad)

    return norm.item()


def clip_gradient(
    grad: torch.Tensor,
    max_norm: float = 1.0
) -> torch.Tensor:
    """
    Clip gradient to maximum norm for stability.

    Prevents gradient explosion while preserving direction.

    Args:
        grad: Gradient tensor (any shape)
        max_norm: Maximum allowed gradient norm

    Returns:
        Clipped gradient (same shape as input)
    """
    current_norm = compute_gradient_norm(grad)

    if current_norm > max_norm:
        # Scale down to max_norm
        scale_factor = max_norm / current_norm
        grad = grad * scale_factor

    return grad
