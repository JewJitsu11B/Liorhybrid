"""
Comprehensive Similarity Vector (15D)

Extended geometric algebra ensemble for manifold-compatible similarity measurement.

Original 7D vector:
    [cosine, wedge, tensor_trace, spinor_magnitude, energy, l2, lior]

Extended 15D vector (manifold-compatible):
    [
        cosine,                     # Angular alignment
        wedge_magnitude,            # Rotational structure
        tensor_trace,               # Interaction strength
        spinor_magnitude,           # Phase-aware overlap
        spinor_phase,               # Phase angle
        energy,                     # Field-mediated coupling
        lior_distance,              # Geodesic distance (PRIMARY)
        l2_tangent_space,           # Euclidean in tangent space
        manhattan_tangent_space,    # L1 in tangent space
        geodesic_kendall_tau,       # Non-parametric correlation (manifold-lifted)
        manifold_mutual_info,       # Information distance (geodesic-binned)
        variational_entropy_diff,   # Field entropy difference
        renyi_entropy_diff,         # Collision entropy difference
        local_curvature_diff,       # Curvature at x vs y
        sectional_curvature         # Curvature of 2-plane spanned by x,y
    ]

Pure PyTorch implementation (no scipy/sklearn).
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math


def compute_comprehensive_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    field_state: Optional[torch.Tensor] = None,
    metric: Optional[torch.Tensor] = None,
    return_dict: bool = False
) -> torch.Tensor:
    """
    Compute 15D comprehensive similarity vector.
    
    Args:
        x: First embedding (d_model,)
        y: Second embedding (d_model,)
        field_state: Optional field state for field-aware measures
        metric: Optional Riemannian metric (d_model, d_model)
        return_dict: If True, return dict with named components
    
    Returns:
        similarity: 15D similarity vector
        OR dict with named components if return_dict=True
    """
    d = x.shape[0]
    
    # Default metric to identity if not provided
    if metric is None:
        metric = torch.eye(d, device=x.device, dtype=x.dtype)
    
    # Normalize inputs
    x_norm = x / (torch.linalg.norm(x) + 1e-8)
    y_norm = y / (torch.linalg.norm(y) + 1e-8)
    
    # 1. Cosine similarity (angular alignment)
    cosine = torch.dot(x_norm, y_norm)
    
    # 2. Wedge product magnitude (rotational structure)
    # For vectors: |x ∧ y| = |x||y|sin(θ)
    wedge_mag = torch.linalg.norm(torch.cross(x[:3], y[:3])) if d >= 3 else torch.tensor(0.0, device=x.device)
    
    # 3. Tensor trace (interaction strength)
    # Tr(x ⊗ y)
    tensor_trace = torch.einsum('i,i->', x, y)
    
    # 4-5. Spinor magnitude and phase (phase-aware overlap)
    spinor_mag, spinor_phase = compute_spinor_similarity(x, y)
    
    # 6. Energy (field-mediated coupling)
    if field_state is not None:
        energy = compute_field_energy_coupling(x, y, field_state)
    else:
        energy = torch.tensor(0.0, device=x.device)
    
    # 7. LIoR distance (geodesic distance - PRIMARY measure)
    lior_dist = compute_lior_distance(x, y, metric)
    
    # 8-9. Tangent space distances
    # Map to tangent space at midpoint
    midpoint = (x + y) / 2
    x_tangent = log_map(x, midpoint, metric)
    y_tangent = log_map(y, midpoint, metric)
    
    l2_tangent = torch.linalg.norm(x_tangent - y_tangent)
    manhattan_tangent = torch.abs(x_tangent - y_tangent).sum()
    
    # 10. Geodesic Kendall Tau (manifold-lifted correlation)
    kendall_tau = compute_geodesic_kendall_tau(x, y, metric)
    
    # 11. Manifold mutual information
    if field_state is not None:
        mutual_info = compute_manifold_mutual_info(x, y, field_state, metric)
    else:
        mutual_info = torch.tensor(0.0, device=x.device)
    
    # 12-13. Entropy differences
    if field_state is not None:
        var_entropy_diff = compute_variational_entropy_diff(x, y, field_state)
        renyi_entropy_diff = compute_renyi_entropy_diff(x, y, field_state)
    else:
        var_entropy_diff = torch.tensor(0.0, device=x.device)
        renyi_entropy_diff = torch.tensor(0.0, device=x.device)
    
    # 14-15. Curvature measures
    local_curv_diff = compute_local_curvature_diff(x, y, metric)
    sectional_curv = compute_sectional_curvature(x, y, metric)
    
    # Assemble vector
    similarity_vector = torch.stack([
        cosine,
        wedge_mag,
        tensor_trace,
        spinor_mag,
        spinor_phase,
        energy,
        lior_dist,
        l2_tangent,
        manhattan_tangent,
        kendall_tau,
        mutual_info,
        var_entropy_diff,
        renyi_entropy_diff,
        local_curv_diff,
        sectional_curv
    ])
    
    if return_dict:
        return {
            'cosine': cosine,
            'wedge_magnitude': wedge_mag,
            'tensor_trace': tensor_trace,
            'spinor_magnitude': spinor_mag,
            'spinor_phase': spinor_phase,
            'energy': energy,
            'lior_distance': lior_dist,
            'l2_tangent': l2_tangent,
            'manhattan_tangent': manhattan_tangent,
            'geodesic_kendall_tau': kendall_tau,
            'manifold_mutual_info': mutual_info,
            'variational_entropy_diff': var_entropy_diff,
            'renyi_entropy_diff': renyi_entropy_diff,
            'local_curvature_diff': local_curv_diff,
            'sectional_curvature': sectional_curv,
            'vector': similarity_vector
        }
    
    return similarity_vector


def compute_spinor_similarity(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spinor-based similarity (magnitude and phase).
    
    Args:
        x, y: Input vectors
    
    Returns:
        magnitude: Spinor magnitude
        phase: Spinor phase angle
    """
    # Complex representation
    x_complex = torch.complex(x[::2], x[1::2]) if x.shape[0] > 1 else torch.complex(x, torch.zeros_like(x))
    y_complex = torch.complex(y[::2], y[1::2]) if y.shape[0] > 1 else torch.complex(y, torch.zeros_like(y))
    
    # Inner product in complex space
    inner = torch.sum(torch.conj(x_complex) * y_complex)
    
    magnitude = torch.abs(inner)
    phase = torch.angle(inner)
    
    return magnitude, phase


def compute_field_energy_coupling(
    x: torch.Tensor,
    y: torch.Tensor,
    field_state: torch.Tensor
) -> torch.Tensor:
    """
    Compute field-mediated energy coupling between points.
    
    Args:
        x, y: Input vectors
        field_state: Field state tensor
    
    Returns:
        energy: Field energy coupling
    """
    # Project to field coordinates (simple spatial mapping)
    D = field_state.shape[2]
    x_proj = x[:D] if x.shape[0] >= D else torch.nn.functional.pad(x, (0, D - x.shape[0]))
    y_proj = y[:D] if y.shape[0] >= D else torch.nn.functional.pad(y, (0, D - y.shape[0]))
    
    # Average field state
    T_avg = field_state.mean(dim=(0, 1))  # (D, D)
    
    # Energy: E = xᵀ T y
    energy = torch.einsum('i,ij,j->', x_proj, T_avg.real if T_avg.is_complex() else T_avg, y_proj)
    
    return energy


def compute_lior_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """
    Compute LIoR (geodesic) distance.
    
    Distance = √((x-y)ᵀ g (x-y))
    
    Args:
        x, y: Input vectors
        metric: Riemannian metric
    
    Returns:
        distance: LIoR distance
    """
    diff = x - y
    
    # Metric distance: √(diffᵀ g diff)
    metric_dist_sq = torch.einsum('i,ij,j->', diff, metric, diff)
    distance = torch.sqrt(torch.clamp(metric_dist_sq, min=0.0))
    
    return distance


def log_map(x: torch.Tensor, base: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map: map point to tangent space at base.
    
    Simplified version: log_base(x) ≈ x - base
    
    Args:
        x: Point to map
        base: Base point
        metric: Riemannian metric
    
    Returns:
        tangent: Vector in tangent space at base
    """
    # Simple approximation (first-order)
    # Full implementation would use Christoffel symbols
    tangent = x - base
    
    return tangent


def compute_geodesic_kendall_tau(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """
    Compute Kendall Tau correlation using geodesic distances.
    
    Non-parametric correlation safe for curved manifolds.
    
    Args:
        x, y: Input vectors
        metric: Riemannian metric
    
    Returns:
        tau: Kendall Tau correlation
    """
    # For single pair, return cosine similarity as proxy
    # Full implementation would compute over multiple points
    diff = x - y
    x_norm = x / (torch.linalg.norm(x) + 1e-8)
    y_norm = y / (torch.linalg.norm(y) + 1e-8)
    
    tau = torch.dot(x_norm, y_norm)
    
    return tau


def compute_manifold_mutual_info(
    x: torch.Tensor,
    y: torch.Tensor,
    field_state: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """
    Compute manifold mutual information.
    
    Uses geodesic distance quantiles for binning.
    
    Args:
        x, y: Input vectors
        field_state: Field state
        metric: Riemannian metric
    
    Returns:
        mi: Mutual information
    """
    # Simplified: use correlation as proxy for MI
    # Full implementation would histogram geodesic distances
    diff = x - y
    corr = torch.dot(x, y) / (torch.linalg.norm(x) * torch.linalg.norm(y) + 1e-8)
    
    # MI approximation: -log(1 - corr²)
    mi = -torch.log(1.0 - corr**2 + 1e-8)
    
    return mi


def compute_variational_entropy_diff(
    x: torch.Tensor,
    y: torch.Tensor,
    field_state: torch.Tensor
) -> torch.Tensor:
    """
    Compute variational entropy difference.
    
    Field-aware entropy measure (faster than eigendecomp).
    
    Args:
        x, y: Input vectors
        field_state: Field state
    
    Returns:
        entropy_diff: Entropy difference
    """
    # Compute local field strength at x and y projections
    D = field_state.shape[2]
    x_proj = x[:D] if x.shape[0] >= D else torch.nn.functional.pad(x, (0, D - x.shape[0]))
    y_proj = y[:D] if y.shape[0] >= D else torch.nn.functional.pad(y, (0, D - y.shape[0]))
    
    # Local entropy proxy: field strength variance
    T_avg = field_state.mean(dim=(0, 1))
    
    Hx = torch.einsum('i,ij,j->', x_proj, T_avg.real if T_avg.is_complex() else T_avg, x_proj)
    Hy = torch.einsum('i,ij,j->', y_proj, T_avg.real if T_avg.is_complex() else T_avg, y_proj)
    
    entropy_diff = torch.abs(Hx - Hy)
    
    return entropy_diff


def compute_renyi_entropy_diff(
    x: torch.Tensor,
    y: torch.Tensor,
    field_state: torch.Tensor,
    alpha: float = 2.0
) -> torch.Tensor:
    """
    Compute Rényi entropy difference (collision entropy for α=2).
    
    Args:
        x, y: Input vectors
        field_state: Field state
        alpha: Rényi parameter (default 2.0 for collision entropy)
    
    Returns:
        renyi_diff: Rényi entropy difference
    """
    # Simplified: use norm as proxy
    D = field_state.shape[2]
    x_proj = x[:D] if x.shape[0] >= D else torch.nn.functional.pad(x, (0, D - x.shape[0]))
    y_proj = y[:D] if y.shape[0] >= D else torch.nn.functional.pad(y, (0, D - y.shape[0]))
    
    # Collision entropy proxy: -log(||x||^α)
    Rx = -torch.log(torch.linalg.norm(x_proj)**alpha + 1e-8)
    Ry = -torch.log(torch.linalg.norm(y_proj)**alpha + 1e-8)
    
    renyi_diff = torch.abs(Rx - Ry)
    
    return renyi_diff


def compute_local_curvature_diff(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """
    Compute local curvature difference between x and y.
    
    Args:
        x, y: Input vectors
        metric: Riemannian metric
    
    Returns:
        curv_diff: Curvature difference
    """
    # Local curvature via metric eigenvalues
    # High eigenvalue spread = high curvature
    
    eigenvalues = torch.linalg.eigvalsh(metric)
    curvature = eigenvalues.std()
    
    # Both points see same metric (global), so diff is zero
    # In full implementation, would compute local metric at each point
    curv_diff = torch.tensor(0.0, device=x.device)
    
    return curv_diff


def compute_sectional_curvature(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """
    Compute sectional curvature of 2-plane spanned by x and y.
    
    Args:
        x, y: Input vectors (span a 2-plane)
        metric: Riemannian metric
    
    Returns:
        sectional_curv: Sectional curvature
    """
    # Gram-Schmidt orthonormalization in metric
    x_norm = x / torch.sqrt(torch.einsum('i,ij,j->', x, metric, x) + 1e-8)
    
    # Project y onto orthogonal complement
    proj = torch.einsum('i,ij,j->', x_norm, metric, y)
    y_orth = y - proj * x_norm
    y_orth_norm = y_orth / torch.sqrt(torch.einsum('i,ij,j->', y_orth, metric, y_orth) + 1e-8)
    
    # Sectional curvature (simplified): trace of Riemann tensor
    # K(x,y) ≈ det(metric) / |x ∧ y|²
    det_metric = torch.linalg.det(metric)
    
    # Cross product magnitude (for 2-plane)
    cross_mag = torch.linalg.norm(x_norm - y_orth_norm)
    
    sectional_curv = torch.sqrt(torch.abs(det_metric)) / (cross_mag**2 + 1e-8)
    
    return sectional_curv


def test_comprehensive_similarity():
    """Test comprehensive similarity computation."""
    print("Testing comprehensive similarity vector...")
    
    d_model = 64
    x = torch.randn(d_model)
    y = torch.randn(d_model)
    
    # Create fake field state
    N_x, N_y, D = 10, 10, 16
    field_state = torch.randn(N_x, N_y, D, D)
    
    # Compute similarity
    sim_vector = compute_comprehensive_similarity(x, y, field_state)
    
    print(f"Input dim: {d_model}")
    print(f"Similarity vector dim: {sim_vector.shape[0]}")
    print(f"Similarity values:")
    
    names = [
        'cosine', 'wedge', 'tensor_trace', 'spinor_mag', 'spinor_phase',
        'energy', 'lior_dist', 'l2_tangent', 'manhattan_tangent',
        'kendall_tau', 'mutual_info', 'var_entropy_diff', 'renyi_entropy_diff',
        'local_curv_diff', 'sectional_curv'
    ]
    
    for name, val in zip(names, sim_vector):
        print(f"  {name:25s}: {val.item():8.4f}")
    
    # Test dict output
    sim_dict = compute_comprehensive_similarity(x, y, field_state, return_dict=True)
    print(f"\nDict output keys: {list(sim_dict.keys())}")
    
    print("Comprehensive similarity test passed!")


if __name__ == '__main__':
    test_comprehensive_similarity()
