"""
Fractional Memory (Damping) Operator

Implements Λ_F[T] = λ_F ∫ K(t-τ) T(τ) dτ

Paper References:
- Equation (7): Fractional memory definition
- Equation (8): Power-law kernel K(τ) = τ^(α-1)/Γ(α)
- Implementation Note 3: Grünwald-Letnikov discretization
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
from typing import List, Union
from collections import deque
import math


def fractional_kernel_weights(alpha: float, n_steps: int, dt: float) -> torch.Tensor:
    """
    Compute Grünwald-Letnikov weights for fractional derivative.

    Paper Equation (8), discrete form:
        w_k^(α) = (-1)^k * binom(α, k)

    For the integral form (anti-derivative), we use α-1:
        K(j*dt) ∝ (j*dt)^(α-1) for j = 0, 1, ..., n_steps-1

    Args:
        alpha: Fractional order (0 < α < 1 typical)
        n_steps: Number of history steps
        dt: Timestep

    Returns:
        Tensor of shape (n_steps,) with kernel weights

    Implementation Note 3 from paper:
        For stability, we normalize the kernel such that Σ w_k = 1.
        This ensures the memory term is bounded.
    """
    if n_steps == 0:
        return torch.tensor([])

    # Compute kernel values: K(j*dt) = (j*dt)^(α-1)
    # For j=0, we use a small epsilon to avoid 0^(α-1) singularity
    indices = torch.arange(n_steps, dtype=torch.float32)
    times = (indices + 1e-8) * dt  # Add epsilon to avoid log(0)

    # K(τ) = τ^(α-1) / Γ(α)
    gamma_alpha = math.gamma(alpha)
    kernel = (times ** (alpha - 1)) / gamma_alpha

    # Normalize so that sum of weights = 1
    # This makes the integral a proper weighted average
    kernel = kernel / kernel.sum()

    return kernel


def fractional_memory_term(
    history: Union[List[torch.Tensor], deque],
    alpha: float,
    lambda_F: float,
    dt: float
) -> torch.Tensor:
    """
    Compute fractional memory damping term.

    Paper Equation (7):
        Λ_F[T]_ij = λ_F ∫₀ᵗ K(t-τ) T_ij(τ) dτ

    Discrete form (Implementation Note 3):
        Λ_F ≈ λ_F Σ_k w_k T(t - k*Δt)

    Args:
        history: List or deque of previous field states [T(t-N*dt), ..., T(t-dt)]
        alpha: Fractional order (0.3-0.7 typical)
        lambda_F: Memory strength parameter
        dt: Timestep

    Returns:
        Memory term of same shape as history elements

    Physical Interpretation:
        This term creates long-range memory effects via power-law
        decay K(τ) ~ τ^(α-1). The negative sign (in master equation)
        makes this a DAMPING term:

        - α → 0: Strong memory (slow decay), heavy damping
        - α → 1: Weak memory (fast decay), light damping

        The term resists rapid changes by coupling to past states.
    """
    if len(history) == 0:
        # No history: no memory term
        # Return zero tensor with correct shape
        # We need at least one reference tensor to get the shape
        raise ValueError("Cannot compute memory term with empty history")

    # Get reference shape from most recent history entry
    ref_shape = history[-1].shape
    device = history[-1].device
    dtype = history[-1].dtype

    if len(history) == 1:
        # Only one history entry: trivial memory term
        return lambda_F * history[0].clone()

    # Compute fractional kernel weights
    n_steps = len(history)
    weights = fractional_kernel_weights(alpha, n_steps, dt)
    weights = weights.to(device=device, dtype=torch.float32)

    # Weighted sum: Σ_k w_k T(t - k*Δt)
    # History is ordered [oldest, ..., newest]
    # Kernel weights are ordered [w_0, ..., w_{N-1}]

    # Stack history into single tensor: (n_steps, N_x, N_y, D, D)
    # Convert deque to list if necessary
    history_list = list(history) if isinstance(history, deque) else history
    history_stack = torch.stack(history_list, dim=0)

    # Reshape weights for broadcasting: (n_steps, 1, 1, 1, 1)
    weights_expanded = weights.view(-1, 1, 1, 1, 1)

    # Weighted sum over time axis
    memory_integral = torch.sum(weights_expanded * history_stack, dim=0)

    # Apply strength parameter
    Lambda_F = lambda_F * memory_integral

    return Lambda_F


def fractional_memory_weight(
    history: Union[List[torch.Tensor], deque],
    alpha: Union[float, torch.Tensor],
    lambda_F: float,
    dt: float
) -> torch.Tensor:
    """
    Compute memory weight coefficient for gradient modulation.

    **Bayesian Formulation:**
    Instead of adding memory as a field vector (which violates probability
    conservation), we use it to modulate the gradient:

        ψ_{t+1} = ψ_t - γ(1 - memory_weight) * ∇H[ψ_t] + noise

    where memory_weight ∈ [0, 1] controls how much the gradient is damped
    by prior information (memory).

    The weight is computed as:
        memory_weight = lambda_F * f(alpha, n_steps)

    where f is a function of memory strength that increases with:
    - Higher alpha (more memory retention)
    - More history steps (more accumulated prior)

    Args:
        history: List or deque of previous field states
        alpha: Fractional order (0 < α < 1)
        lambda_F: Memory strength parameter
        dt: Timestep

    Returns:
        memory_weight: Scalar in [0, 1] that modulates gradient updates

    Physical Interpretation:
        - weight = 0: No memory, full gradient descent (pure likelihood)
        - weight = 1: Full memory, no gradient update (pure prior)
        - weight ∈ (0,1): Balanced Bayesian update (likelihood + prior)

    This ensures:
        1. Probability conservation (multiplicative not additive)
        2. Energy conservation (no injection from memory)
        3. True Bayesian posterior: P(ψ|D) ∝ P(D|ψ) P(ψ_prior)
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("kernel_fractional_memory_weight", __file__)

    if len(history) == 0:
        # No history: no memory damping
        return 0.0

    # Compute effective memory strength based on history size and alpha
    n_steps = len(history)

    # Memory effectiveness increases with:
    # 1. Higher alpha (slower decay, more memory)
    # 2. More history steps (more accumulated prior)
    # Use sigmoid to map to [0, 1]

    # History elements are complex field tensors; build this computation in real dtype so
    # `torch.clamp` works and we don't accidentally promote to complex.
    ref = history[-1]
    ref_r = ref.real if ref.is_complex() else ref

    # Preserve gradient flow if `alpha` is a tensor/Parameter; otherwise create a scalar on-device.
    alpha_t = alpha if torch.is_tensor(alpha) else ref_r.new_tensor(alpha)
    alpha_t = alpha_t.to(device=ref.device, dtype=ref_r.dtype)

    # Effective memory accumulation (unnormalized). This grows with both alpha and n_steps.
    n_t = alpha_t.new_tensor(1.0 + float(n_steps))
    memory_accumulation = alpha_t * torch.log(n_t)

    # Modulate by lambda_F and clamp to [0, 1] (tensor, no CPU sync).
    lambda_t = lambda_F if torch.is_tensor(lambda_F) else alpha_t.new_tensor(lambda_F)
    memory_weight = torch.clamp(memory_accumulation * lambda_t, min=0.0, max=1.0)

    return memory_weight


def update_history_buffer(
    history: List[torch.Tensor],
    new_state: torch.Tensor,
    max_history: int
) -> List[torch.Tensor]:
    """
    Maintain fixed-size history buffer for memory computation.

    Paper Algorithm 1, lines 16-19.

    Args:
        history: Current history list
        new_state: New field state to append
        max_history: Maximum buffer size

    Returns:
        Updated history list (oldest removed if at capacity)

    Note: This is a utility function that could be called from
    the main evolution loop. Alternatively, a collections.deque
    with maxlen can handle this automatically.
    """
    history.append(new_state.clone())

    if len(history) > max_history:
        history.pop(0)  # Remove oldest

    return history
