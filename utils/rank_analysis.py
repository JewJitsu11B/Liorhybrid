"""
Field Rank Analysis Utilities

Tools for analyzing the effective rank of cognitive tensor fields
and determining optimal compression strategies.

These utilities help answer:
1. What is the effective rank of the field during training?
2. How much can we compress without losing physics accuracy?
3. When is low-rank approximation safe vs risky?
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_field_spectrum(
    T: torch.Tensor,
    n_samples: int = 100
) -> Dict[str, torch.Tensor]:
    """
    Analyze singular value spectrum of cognitive field.

    Args:
        T: Tensor field (N_x, N_y, D, D) complex
        n_samples: Number of spatial points to sample

    Returns:
        Dictionary with spectrum statistics:
        - 'singular_values_mean': Mean singular values across samples
        - 'singular_values_std': Std of singular values
        - 'effective_rank': Estimated effective rank
        - 'compression_ratio': Potential compression at 99% accuracy
    """
    N_x, N_y, D, _ = T.shape
    
    # Sample random points
    singular_values_list = []
    
    for _ in range(n_samples):
        x = torch.randint(0, N_x, (1,)).item()
        y = torch.randint(0, N_y, (1,)).item()
        T_xy = T[x, y, :, :]
        
        # Compute SVD
        _, S, _ = torch.linalg.svd(T_xy)
        singular_values_list.append(S.cpu())
    
    # Stack and analyze
    S_stack = torch.stack(singular_values_list)  # (n_samples, D)
    S_mean = S_stack.mean(dim=0)
    S_std = S_stack.std(dim=0)
    
    # Effective rank (number of singular values above threshold)
    threshold_99 = 0.01 * S_mean[0]  # 1% of largest
    threshold_999 = 0.001 * S_mean[0]  # 0.1% of largest
    
    rank_99 = (S_mean > threshold_99).sum().item()
    rank_999 = (S_mean > threshold_999).sum().item()
    
    # Compression ratio
    compression_ratio = D / rank_99 if rank_99 > 0 else 1.0
    
    return {
        'singular_values_mean': S_mean,
        'singular_values_std': S_std,
        'effective_rank_99': rank_99,
        'effective_rank_999': rank_999,
        'compression_ratio': compression_ratio,
        'full_rank': D
    }


def validate_lowrank_physics(
    field,
    ranks: List[int] = [4, 8, 16, 32],
    metrics: List[str] = ['energy', 'unitarity', 'entropy']
) -> Dict[str, Dict]:
    """
    Validate accuracy of low-rank approximations for physics computations.

    Args:
        field: CognitiveTensorField instance
        ranks: List of ranks to test
        metrics: Which physics metrics to validate

    Returns:
        Dictionary mapping metric -> rank -> {'value', 'error'}
    """
    from core.tensor_field import CognitiveTensorField
    
    results = {}
    
    # Compute full-rank reference values
    reference = {}
    if 'energy' in metrics:
        reference['energy'] = field.compute_energy()
    if 'unitarity' in metrics:
        reference['unitarity'] = field.compute_unitarity_deviation()
    if 'entropy' in metrics:
        from utils.metrics import compute_entropy
        reference['entropy'] = compute_entropy(field.T)
    
    # Test each rank
    for rank in ranks:
        if rank >= field.T.shape[2]:
            continue  # Skip if rank >= D
        
        # Compress field
        U, S, V = compress_field_lowrank(field.T, rank)
        T_approx = reconstruct_from_lowrank(U, S, V)
        
        # Create temporary field with approximation
        field_approx = CognitiveTensorField(field.config)
        field_approx.T = T_approx
        
        # Compute metrics
        rank_results = {}
        for metric in metrics:
            if metric == 'energy':
                value = field_approx.compute_energy()
                ref = reference['energy']
            elif metric == 'unitarity':
                value = field_approx.compute_unitarity_deviation()
                ref = reference['unitarity']
            elif metric == 'entropy':
                from utils.metrics import compute_entropy
                value = compute_entropy(field_approx.T)
                ref = reference['entropy']
            else:
                continue
            
            error = abs(value - ref) / (abs(ref) + 1e-8)
            rank_results[metric] = {
                'value': value,
                'reference': ref,
                'absolute_error': abs(value - ref),
                'relative_error': error
            }
        
        results[f'rank_{rank}'] = rank_results
    
    return results


def compress_field_lowrank(
    T: torch.Tensor,
    target_rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compress field to low-rank representation via SVD.

    Args:
        T: Tensor field (N_x, N_y, D, D) complex
        target_rank: Target rank r

    Returns:
        U: (N_x*N_y, D, r) left singular vectors
        S: (N_x*N_y, r) singular values
        Vh: (N_x*N_y, r, D) right singular vectors (conjugate transposed)
    """
    N_x, N_y, D, _ = T.shape
    
    # Reshape to batch
    T_batch = T.reshape(N_x * N_y, D, D)
    
    # Batch SVD
    U, S, Vh = torch.linalg.svd(T_batch, full_matrices=False)
    
    # Truncate to target rank
    U_r = U[:, :, :target_rank]
    S_r = S[:, :target_rank]
    Vh_r = Vh[:, :target_rank, :]
    
    return U_r, S_r, Vh_r


def reconstruct_from_lowrank(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct field from low-rank factors.

    Args:
        U: (N, D, r) left singular vectors
        S: (N, r) singular values
        Vh: (N, r, D) right singular vectors

    Returns:
        T_approx: (N, D, D) reconstructed tensors
    """
    # T ≈ U @ diag(S) @ Vh
    # For batch: use einsum or bmm
    
    # Method 1: einsum
    # T_approx = torch.einsum('ndr,nr,nrd->ndd', U, S, Vh)
    
    # Method 2: bmm (more efficient)
    US = U * S.unsqueeze(1)  # (N, D, r) * (N, 1, r) = (N, D, r)
    T_approx = torch.bmm(US, Vh)  # (N, D, r) @ (N, r, D) = (N, D, D)
    
    return T_approx


def plot_spectrum_decay(
    spectrum_stats: Dict[str, torch.Tensor],
    save_path: Optional[str] = None
):
    """
    Plot singular value decay to visualize field rank.

    Args:
        spectrum_stats: Output from analyze_field_spectrum()
        save_path: Optional path to save figure
    """
    S_mean = spectrum_stats['singular_values_mean'].cpu().numpy()
    S_std = spectrum_stats['singular_values_std'].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Linear scale
    ax = axes[0]
    x = np.arange(len(S_mean))
    ax.plot(x, S_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(x, S_mean - S_std, S_mean + S_std, alpha=0.3, label='±1 std')
    ax.axhline(y=0.01 * S_mean[0], color='r', linestyle='--', label='1% threshold')
    ax.axhline(y=0.001 * S_mean[0], color='orange', linestyle='--', label='0.1% threshold')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value Magnitude')
    ax.set_title('Singular Value Spectrum (Linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax = axes[1]
    ax.semilogy(x, S_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(x, S_mean - S_std, S_mean + S_std, alpha=0.3, label='±1 std')
    ax.axhline(y=0.01 * S_mean[0], color='r', linestyle='--', label='1% threshold')
    ax.axhline(y=0.001 * S_mean[0], color='orange', linestyle='--', label='0.1% threshold')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value Magnitude (log scale)')
    ax.set_title('Singular Value Spectrum (Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectrum plot saved to {save_path}")
    
    plt.close()


def recommend_rank(
    spectrum_stats: Dict[str, torch.Tensor],
    accuracy_target: float = 0.99
) -> Dict[str, int]:
    """
    Recommend rank based on accuracy target.

    Args:
        spectrum_stats: Output from analyze_field_spectrum()
        accuracy_target: Target accuracy (e.g., 0.99 for 99%)

    Returns:
        Dictionary with recommendations for different use cases
    """
    S_mean = spectrum_stats['singular_values_mean']
    S_cumsum = torch.cumsum(S_mean ** 2, dim=0)
    S_total = S_cumsum[-1]
    
    # Find rank that captures target energy
    energy_ratios = S_cumsum / S_total
    
    rank_99 = (energy_ratios < 0.99).sum().item() + 1
    rank_999 = (energy_ratios < 0.999).sum().item() + 1
    rank_9999 = (energy_ratios < 0.9999).sum().item() + 1
    
    return {
        'monitoring': min(8, rank_99),  # Conservative for dashboards
        'diagnostics': rank_99,          # Good balance
        'training': rank_999,             # High accuracy
        'critical': rank_9999,            # Near-exact
        'full': spectrum_stats['full_rank']
    }


def benchmark_rank_performance(
    field,
    ranks: List[int] = [4, 8, 16, 32],
    n_iterations: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark computational time for different ranks.

    Args:
        field: CognitiveTensorField instance
        ranks: List of ranks to test
        n_iterations: Number of iterations for timing

    Returns:
        Dictionary with timing results
    """
    import time
    
    results = {}
    
    # Benchmark full rank
    times = []
    for _ in range(n_iterations):
        start = time.time()
        _ = field.compute_energy()
        times.append(time.time() - start)
    
    results['full'] = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'speedup': 1.0
    }
    
    baseline_time = results['full']['mean_time']
    
    # Benchmark each rank
    # Note: Would need to implement lowrank compute_energy first
    # This is a placeholder for the API
    
    return results


# Example usage
if __name__ == "__main__":
    # This would be used in practice:
    """
    from core import CognitiveTensorField, FAST_TEST_CONFIG
    
    field = CognitiveTensorField(FAST_TEST_CONFIG)
    field.evolve_step()
    
    # Analyze spectrum
    stats = analyze_field_spectrum(field.T)
    print(f"Effective rank (99% accuracy): {stats['effective_rank_99']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Get recommendations
    recs = recommend_rank(stats)
    print(f"Recommended ranks: {recs}")
    
    # Validate accuracy
    validation = validate_lowrank_physics(field, ranks=[4, 8, 16])
    print(f"Validation results: {validation}")
    
    # Plot spectrum
    plot_spectrum_decay(stats, save_path='field_spectrum.png')
    """
    pass
