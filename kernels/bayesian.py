"""
Bayesian Recursive Update Operator

Implements Λ_QR[T] = λ_QR(B[T(t-1)] - T(t-1))

Paper References:
- Equation (4): Bayesian update definition
- Equation (5): Posterior construction B[T]
- Equation (6): Evidence weighting w_ij
"""

import torch
from typing import Optional


def compute_evidence_weights(
    T: torch.Tensor,
    evidence: Optional[torch.Tensor],
    tau: float
) -> torch.Tensor:
    """
    Compute evidence weights w_ij(x) for Bayesian update.

    Paper Equation (6):
        w_ij(x) = exp(-|T_ij - E_ij|²/τ)

    Args:
        T: Current tensor field (N_x, N_y, D, D)
        evidence: Evidence tensor E_ij (same shape as T), or None
        tau: Temperature parameter (controls sharpness)

    Returns:
        Weight tensor of same shape as T

    Note: If evidence is None, uniform weights are returned.
    """
    if evidence is None:
        # No evidence: uniform weights
        return torch.ones_like(T, dtype=torch.float32)

    # Compute |T - E|²
    diff_sq = torch.abs(T - evidence) ** 2

    # w_ij = exp(-diff²/τ)
    weights = torch.exp(-diff_sq / tau)

    return weights


def bayesian_posterior(
    T: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Construct Bayesian posterior B[T] via weighted normalization.

    Paper Equation (5):
        B[T]_ij(x) = (w_ij(x) * T_ij(x)) / Z
        where Z = Σ_kl ∫ w_kl(x')|T_kl(x')|² dV

    Args:
        T: Tensor field (N_x, N_y, D, D)
        weights: Evidence weights (same shape)

    Returns:
        Posterior B[T] of same shape

    Implementation Note:
        The integral ∫ dV becomes a sum over spatial grid.
        The normalization Z ensures ∫ |B[T]|² dV = 1.
    """
    # Weighted field: w * T
    weighted_T = weights * T

    # Normalization: Z = Σ_ijxy w_ij(x,y)|T_ij(x,y)|²
    # Sum over all indices (spatial and tensor). Keep tensor control flow (no Python if).
    Z = torch.sum(weights * torch.abs(T) ** 2)

    eps = Z.new_tensor(1e-12)
    B_T = weighted_T / (Z + eps)
    # If Z is effectively zero, fall back to unchanged field (broadcasted scalar condition).
    B_T = torch.where(Z > eps, B_T, T)

    return B_T


def bayesian_recursive_term(
    T_current: torch.Tensor,
    T_prev_collapsed: Optional[torch.Tensor],
    evidence: Optional[torch.Tensor],
    lambda_QR: float,
    tau: float
) -> torch.Tensor:
    """
    Compute Bayesian recursive update term.

    Paper Equation (4):
        Λ_QR[T]_ij = λ_QR * (B[T(t-Δt)]_ij - T_ij(t-Δt))

    Args:
        T_current: Current field T(t) for shape reference
        T_prev_collapsed: Previous collapsed state T(t-Δt)
        evidence: Evidence tensor E_ij (optional)
        lambda_QR: Update strength parameter
        tau: Bayesian temperature

    Returns:
        Update term Λ_QR of same shape as T_current

    Physical Interpretation:
        This term drives the field toward the Bayesian posterior
        based on evidence. It represents "belief revision" from
        the previous collapsed state toward the new evidence.

        - If T_prev_collapsed is None (first step), no update occurs
        - lambda_QR controls update rate (0 = no learning, 1 = full jump)
        - tau controls sharpness of evidence weighting
    """
    if T_prev_collapsed is None:
        # No previous state: no Bayesian update
        return torch.zeros_like(T_current)

    # 1. Compute evidence weights (Paper Eq 6)
    weights = compute_evidence_weights(T_prev_collapsed, evidence, tau)

    # 2. Construct Bayesian posterior (Paper Eq 5)
    B_T_prev = bayesian_posterior(T_prev_collapsed, weights)

    # 3. Compute update term: λ_QR * (B[T_prev] - T_prev)
    Lambda_QR = lambda_QR * (B_T_prev - T_prev_collapsed)

    return Lambda_QR
