"""
Metrics and Diagnostics for Bayesian Cognitive Field

Conservation laws, entropy, correlation measures.

Paper References:
- Section 6.1: Conservation laws
- Section 6.2: Emergent clustering metrics
"""

import torch
import numpy as np
from typing import List


def compute_norm_conservation(history: List[torch.Tensor]) -> torch.Tensor:
    """
    Check norm conservation over evolution history.

    Paper Section 6.1: Modified unitarity with dissipation.

    Args:
        history: List of field states T(t)

    Returns:
        Tensor of ||T(t)||² values over time

    Expected behavior:
        - Without fractional memory: ||T||² constant
        - With fractional memory: ||T||² slowly decays (damping)
    """
    norms = torch.tensor([
        torch.sum(torch.abs(T)**2).item()
        for T in history
    ])
    return norms


def compute_entropy(T: torch.Tensor, epsilon: float = 1e-12) -> float:
    """
    Compute von Neumann-like entropy of the field.

    Formula:
        S = -Tr(ρ log ρ)
        where ρ = T†T / Tr(T†T) (density matrix analog)

    Args:
        T: Tensor field (N_x, N_y, D, D)
        epsilon: Small constant to avoid log(0)

    Returns:
        Entropy value S

    Physical interpretation:
        Measures uncertainty/spread in the field configuration.
        Lower entropy = more localized/decided state.
        Higher entropy = more distributed/uncertain state.

    Implementation: Vectorized for efficiency - computes all spatial
    points in parallel using batch eigenvalue decomposition.
    """
    N_x, N_y, D, _ = T.shape

    # Reshape to batch all spatial points: (N_x*N_y, D, D)
    T_batch = T.reshape(N_x * N_y, D, D)

    # Construct density matrices: ρ = T†T for all points
    T_dag = T_batch.conj()  # (N_x*N_y, D, D)
    rho_batch = torch.bmm(T_dag.transpose(-2, -1), T_batch)  # (N_x*N_y, D, D)

    # Compute traces for normalization
    traces = torch.diagonal(rho_batch, dim1=-2, dim2=-1).sum(dim=-1)  # (N_x*N_y,)
    traces_real = traces.real

    # Normalize: ρ = ρ / Tr(ρ), only for non-zero traces
    valid_mask = traces_real > epsilon
    rho_normalized = rho_batch.clone()
    rho_normalized[valid_mask] = rho_batch[valid_mask] / traces_real[valid_mask].unsqueeze(-1).unsqueeze(-1)

    # Compute eigenvalues in batch (Hermitian eigenvalue solver)
    # Only process valid (non-zero trace) density matrices
    if valid_mask.sum() == 0:
        return 0.0

    eigenvalues = torch.linalg.eigvalsh(rho_normalized[valid_mask])  # (n_valid, D)
    eigenvalues_real = eigenvalues.real.clamp(min=epsilon)  # Ensure positive

    # von Neumann entropy: S = -Σ λ log λ for each point
    entropies = -torch.sum(eigenvalues_real * torch.log(eigenvalues_real), dim=-1)  # (n_valid,)

    # Average entropy per spatial point
    total_entropy = entropies.sum().item()
    avg_entropy = total_entropy / (N_x * N_y)

    return avg_entropy


def compute_local_correlation(
    T: torch.Tensor,
    point1: tuple,
    point2: tuple
) -> complex:
    """
    Compute correlation between two spatial points.

    Formula:
        C(x,y) = Tr(T†(x) T(y)) / sqrt(Tr(T†(x)T(x)) Tr(T†(y)T(y)))

    Args:
        T: Tensor field (N_x, N_y, D, D)
        point1: (x1, y1) spatial coordinates
        point2: (x2, y2) spatial coordinates

    Returns:
        Correlation coefficient [-1, 1] (complex)

    Note: This computes LOCAL correlation without instantiating
    the full N²xN² matrix.
    """
    x1, y1 = point1
    x2, y2 = point2

    # Extract tensors at the two points
    T1 = T[x1, y1, :, :]  # (D, D)
    T2 = T[x2, y2, :, :]  # (D, D)

    # Compute correlation
    numerator = torch.trace(T1.conj().T @ T2)
    denom1 = torch.trace(T1.conj().T @ T1)
    denom2 = torch.trace(T2.conj().T @ T2)

    correlation = numerator / torch.sqrt(denom1 * denom2)

    return correlation.item()


def compute_correlation_length(T: torch.Tensor, max_distance: int = None) -> float:
    """
    Estimate correlation length of the field.

    Measures characteristic distance over which correlations decay.

    Args:
        T: Tensor field (N_x, N_y, D, D)
        max_distance: Maximum distance to sample (default: min(N_x, N_y) / 4)

    Returns:
        Correlation length ξ in grid units

    Method:
        Sample correlations C(r) at various distances r from center point,
        fit to exponential decay C(r) ~ exp(-r/ξ), extract ξ.

    Physical Interpretation:
        - Large ξ: Long-range correlations (field is spatially coherent)
        - Small ξ: Short-range correlations (field is localized)
        - ξ ≈ 1: Nearest-neighbor correlations only
    """
    N_x, N_y, D, _ = T.shape

    # Use center point as reference
    x0, y0 = N_x // 2, N_y // 2

    if max_distance is None:
        max_distance = min(N_x, N_y) // 4

    # Sample correlations at various distances
    distances = []
    correlations = []

    for r in range(1, max_distance + 1):
        # Sample points at distance r from center (in 4 cardinal directions)
        test_points = [
            (x0 + r, y0) if x0 + r < N_x else None,
            (x0 - r, y0) if x0 - r >= 0 else None,
            (x0, y0 + r) if y0 + r < N_y else None,
            (x0, y0 - r) if y0 - r >= 0 else None,
        ]

        # Compute average correlation at this distance
        correlations_at_r = []
        for point in test_points:
            if point is not None:
                corr = compute_local_correlation(T, (x0, y0), point)
                correlations_at_r.append(abs(corr))  # Use magnitude

        if correlations_at_r:
            avg_corr = sum(correlations_at_r) / len(correlations_at_r)
            distances.append(r)
            correlations.append(avg_corr)

    if len(distances) < 3:
        # Not enough points to fit
        return 1.0  # Default to nearest-neighbor

    # Fit exponential decay: C(r) = C0 * exp(-r/ξ)
    # Taking log: log C(r) = log C0 - r/ξ
    # Linear fit: log C ~ -r/ξ

    distances_np = np.array(distances, dtype=np.float32)
    correlations_np = np.array(correlations, dtype=np.float32)

    # Filter out near-zero correlations
    valid = correlations_np > 1e-6
    if valid.sum() < 2:
        return 1.0

    distances_np = distances_np[valid]
    correlations_np = correlations_np[valid]

    log_corr = np.log(correlations_np + 1e-10)

    # Linear regression: log_corr = a + b * distances
    # where b = -1/ξ
    coeffs = np.polyfit(distances_np, log_corr, deg=1)
    slope = coeffs[0]  # This is -1/ξ

    if slope < -1e-6:  # Negative slope (decay)
        xi = -1.0 / slope
    else:
        # No decay or positive slope (unusual), return max distance
        xi = float(max_distance)

    return xi


def compute_effective_dimension(eigenvalues: torch.Tensor) -> float:
    """
    Compute effective dimensionality from eigenspectrum.

    Formula:
        D_eff = (Σ λ_i)² / (Σ λ_i²)

    This is the participation ratio, measuring how many
    eigenmodes are significantly populated.

    Args:
        eigenvalues: Eigenvalue spectrum (sorted descending)

    Returns:
        Effective dimension D_eff

    Interpretation:
        - D_eff ≈ 1: Field dominated by single mode
        - D_eff ≈ N: Field spread over many modes uniformly
    """
    sum_eig = torch.sum(eigenvalues)
    sum_eig_sq = torch.sum(eigenvalues ** 2)

    D_eff = (sum_eig ** 2) / sum_eig_sq

    return D_eff.item()
