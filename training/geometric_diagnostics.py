"""
Geometric Diagnostics for LIoR Training

Three internal consistency measures (no external supervision required):

1. E_geo  - Geodesic consistency residual
           |γ̈ + Γ(γ̇, γ̇)| → 0 if metric explains motion

2. E_var  - LIoR path optimality gap
           E[max(0, -ΔLIoR)] → 0 if path is locally optimal

3. E_struct - Curvature-velocity coupling
              |corr(R, |γ̇|) + 1| → 0 if geometry acts as resistance

These are the correct convergence signals in the LIoR framework.
"""

import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GeometricDiagnostics:
    """Container for all three geometric diagnostics."""
    E_geo: float      # Geodesic residual (should decrease)
    E_var: float      # LIoR optimality gap (should decrease)
    E_struct: float   # Curvature-velocity coupling (should approach 0)


def compute_christoffel_diagonal(
    metric_diag: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Christoffel symbols for diagonal metric.

    For diagonal metric g_ii, the only non-zero Christoffel symbols are:
        Γ^i_ii = (1/2g_ii) ∂_i g_ii
        Γ^i_jj = -(1/2g_ii) ∂_j g_jj  (i ≠ j)
        Γ^i_ij = (1/2g_ii) ∂_j g_ii   (i ≠ j)

    For constant diagonal metric (our case), all Γ = 0.
    For learned metric, we approximate derivatives via finite differences.

    Args:
        metric_diag: Diagonal of metric tensor (batch, dim)

    Returns:
        christoffel: Approximated Christoffel contribution (batch, dim)
    """
    # For constant diagonal metric, Christoffel = 0
    # This is the first-order approximation
    # Future: compute actual derivatives if metric varies spatially
    return torch.zeros_like(metric_diag)


def geodesic_residual(
    velocity: torch.Tensor,          # γ̇ at time t (batch, dim)
    velocity_prev: torch.Tensor,     # γ̇ at time t-1 (batch, dim)
    metric_diag: torch.Tensor,       # g_ii diagonal (batch, dim) or (dim,)
    dt: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute geodesic consistency residual E_geo.

    E_geo = |γ̈ + Γ^λ_μν γ̇^μ γ̇^ν|

    If the metric correctly describes the dynamics, trajectories should
    satisfy the geodesic equation, making E_geo ≈ 0.

    Args:
        velocity: Current velocity (batch, dim)
        velocity_prev: Previous velocity (batch, dim)
        metric_diag: Metric diagonal (batch, dim) or (dim,)
        dt: Time step

    Returns:
        E_geo: Geodesic residual per batch element (batch,)
    """
    # Compute acceleration: γ̈ ≈ (γ̇_t - γ̇_{t-1}) / dt
    acceleration = (velocity - velocity_prev) / dt

    # For diagonal metric, Christoffel contribution
    # Γ^λ_μν γ̇^μ γ̇^ν simplifies significantly
    # For constant diagonal: Γ = 0, so geodesic eq is just γ̈ = 0
    christoffel_term = compute_christoffel_diagonal(metric_diag, eps)

    # Geodesic equation: γ̈^λ + Γ^λ_μν γ̇^μ γ̇^ν = 0
    # Residual = |LHS|
    residual = acceleration + christoffel_term * velocity.pow(2)

    # Norm of residual vector
    E_geo = torch.norm(residual, dim=-1)

    return E_geo


def lior_optimality_gap(
    path_lior: torch.Tensor,         # LIoR of current path (batch,)
    perturbed_liors: torch.Tensor,   # LIoR of perturbed paths (batch, n_perturbations)
) -> torch.Tensor:
    """
    Compute LIoR path optimality gap E_var.

    E_var = E[max(0, -ΔLIoR)] where ΔLIoR = LIoR[γ + δγ] - LIoR[γ]

    If the path is optimal, perturbations should increase LIoR (or stay same).
    Large E_var means the path could be improved → metric is misaligned.

    Args:
        path_lior: LIoR of the actual path (batch,)
        perturbed_liors: LIoR of perturbed paths (batch, n_perturbations)

    Returns:
        E_var: Optimality gap per batch element (batch,)
    """
    # ΔLIoR = perturbed - original
    delta_lior = perturbed_liors - path_lior.unsqueeze(-1)

    # max(0, -ΔLIoR) = amount by which perturbation improved (lowered) LIoR
    # This should be 0 if path is optimal
    improvement = torch.clamp(-delta_lior, min=0)

    # Average over perturbations
    E_var = improvement.mean(dim=-1)

    return E_var


def curvature_velocity_coupling(
    curvature: torch.Tensor,    # R(γ(t)) - scalar curvature along path (batch, time)
    velocity_mag: torch.Tensor, # |γ̇(t)| - velocity magnitude (batch, time)
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute curvature-velocity coupling E_struct.

    E_struct = |corr(R, |γ̇|) + 1|

    In optimal flow through a resistance geometry:
    - High curvature (high cost) → low velocity (careful traversal)
    - Low curvature (low cost) → high velocity (fast traversal)

    So correlation should be ≈ -1, making E_struct ≈ 0.

    Args:
        curvature: Scalar curvature along trajectory (batch, time)
        velocity_mag: Velocity magnitudes along trajectory (batch, time)

    Returns:
        E_struct: Coupling measure per batch element (batch,)
    """
    # Center the variables
    R_centered = curvature - curvature.mean(dim=-1, keepdim=True)
    v_centered = velocity_mag - velocity_mag.mean(dim=-1, keepdim=True)

    # Compute correlation
    numerator = (R_centered * v_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (R_centered.pow(2).sum(dim=-1) + eps) *
        (v_centered.pow(2).sum(dim=-1) + eps)
    )

    correlation = numerator / denominator

    # E_struct = |corr + 1| (should be near 0 for perfect anti-correlation)
    E_struct = torch.abs(correlation + 1.0)

    return E_struct


class PathBuffer:
    """
    Circular buffer for tracking path history (velocities, curvatures).

    Used to compute time-series diagnostics without storing full trajectories.
    """

    def __init__(self, buffer_size: int = 64, dim: int = 16, device: str = "cuda"):
        self.buffer_size = buffer_size
        self.dim = dim
        self.device = device

        # Buffers for last N steps
        self.velocities = torch.zeros(buffer_size, dim, device=device)
        self.curvatures = torch.zeros(buffer_size, device=device)
        self.velocity_mags = torch.zeros(buffer_size, device=device)
        self.liors = torch.zeros(buffer_size, device=device)

        self.ptr = 0
        self.full = False

    def push(
        self,
        velocity: torch.Tensor,     # (dim,) or (batch, dim) → take mean
        curvature: torch.Tensor,    # scalar or (batch,) → take mean
        lior: torch.Tensor,         # scalar or (batch,) → take mean
    ):
        """Add a step to the buffer."""
        # Handle batched input
        if velocity.dim() > 1:
            velocity = velocity.mean(dim=0)
        if curvature.dim() > 0:
            curvature = curvature.mean()
        if lior.dim() > 0:
            lior = lior.mean()

        self.velocities[self.ptr] = velocity.detach()
        self.curvatures[self.ptr] = curvature.detach()
        self.velocity_mags[self.ptr] = velocity.norm().detach()
        self.liors[self.ptr] = lior.detach()

        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def compute_diagnostics(
        self,
        metric_diag: torch.Tensor,
        n_perturbations: int = 8,
        perturbation_scale: float = 0.1,
    ) -> Optional[GeometricDiagnostics]:
        """
        Compute all three diagnostics from buffer history.

        Returns None if buffer not full enough.
        """
        if not self.full and self.ptr < 10:
            return None  # Need some history

        # Get valid range
        n = self.buffer_size if self.full else self.ptr

        # === E_geo: Geodesic residual ===
        # Use last two velocities
        v_curr = self.velocities[(self.ptr - 1) % self.buffer_size]
        v_prev = self.velocities[(self.ptr - 2) % self.buffer_size]
        E_geo = geodesic_residual(
            v_curr.unsqueeze(0),
            v_prev.unsqueeze(0),
            metric_diag.unsqueeze(0) if metric_diag.dim() == 1 else metric_diag
        ).item()

        # === E_var: LIoR optimality gap ===
        # Perturb current velocity, estimate perturbed LIoR
        current_lior = self.liors[(self.ptr - 1) % self.buffer_size]

        # Simple perturbation: add noise to velocity, estimate LIoR change
        # This is an approximation - full computation would re-run dynamics
        perturbations = torch.randn(n_perturbations, self.dim, device=self.device)
        perturbations = perturbations * perturbation_scale

        # Approximate perturbed LIoR via linear response
        # ΔLIoR ≈ ∂LIoR/∂v · δv ≈ R · |v + δv|² - R · |v|²
        R_curr = self.curvatures[(self.ptr - 1) % self.buffer_size]
        v_curr_expanded = v_curr.unsqueeze(0).expand(n_perturbations, -1)
        perturbed_v = v_curr_expanded + perturbations

        perturbed_liors = R_curr * perturbed_v.pow(2).sum(dim=-1)
        original_lior_approx = R_curr * v_curr.pow(2).sum()

        E_var = lior_optimality_gap(
            original_lior_approx.unsqueeze(0),
            perturbed_liors.unsqueeze(0)
        ).item()

        # === E_struct: Curvature-velocity coupling ===
        # Use full buffer history
        if self.full:
            curvatures = self.curvatures
            vel_mags = self.velocity_mags
        else:
            curvatures = self.curvatures[:self.ptr]
            vel_mags = self.velocity_mags[:self.ptr]

        E_struct = curvature_velocity_coupling(
            curvatures.unsqueeze(0),
            vel_mags.unsqueeze(0)
        ).item()

        return GeometricDiagnostics(
            E_geo=E_geo,
            E_var=E_var,
            E_struct=E_struct
        )


def format_diagnostics(diag: GeometricDiagnostics) -> str:
    """Format diagnostics for logging."""
    return (
        f"E_geo={diag.E_geo:.4f} "
        f"E_var={diag.E_var:.4f} "
        f"E_struct={diag.E_struct:.4f}"
    )
