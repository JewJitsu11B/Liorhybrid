"""
Spectral Analysis of Prime Numbers on Neto Morphic Fields

This module implements spectral methods for analyzing prime number distribution
using the LIoRHybrid physics framework. It connects number theory (Riemann zeta
function, prime counting) with spectral decomposition, eigenvalue analysis, and
field dynamics.

Key concepts:
- Encode arithmetic functions (von Mangoldt, Möbius) into tensor field
- Perform spectral decomposition (SVD, eigenvalues)
- Extract oscillatory modes analogous to Riemann zeta zeros
- Visualize prime patterns through field dynamics

References:
- SPECTRAL_PRIME_ANALYSIS.md: Full theoretical background
- Montgomery-Odlyzko: Zeta zeros and random matrix theory
- Hilbert-Pólya: Zeta zeros as eigenvalues of self-adjoint operator
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def sieve_of_eratosthenes(n: int) -> np.ndarray:
    """
    Generate all primes up to n using Sieve of Eratosthenes.
    
    Args:
        n: Upper limit (exclusive)
        
    Returns:
        Array of prime numbers
    """
    if n < 2:
        return np.array([], dtype=np.int64)
    
    # Create boolean array
    is_prime = np.ones(n, dtype=bool)
    is_prime[:2] = False
    
    # Sieve
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i:n:i] = False
            
    return np.where(is_prime)[0]


def prime_factors(n: int) -> List[int]:
    """
    Return list of prime factors of n.
    
    Args:
        n: Integer to factor
        
    Returns:
        List of prime factors (with multiplicity)
    """
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def von_mangoldt(n: int, primes: Optional[np.ndarray] = None) -> float:
    """
    Von Mangoldt function Λ(n).
    
    Λ(n) = log(p) if n = p^k for some prime p and k ≥ 1
    Λ(n) = 0 otherwise
    
    Args:
        n: Input integer
        primes: Optional precomputed primes array
        
    Returns:
        Λ(n)
    """
    if n < 2:
        return 0.0
        
    factors = prime_factors(n)
    if len(factors) == 0:
        return 0.0
        
    # Check if all factors are the same (i.e., n = p^k)
    if len(set(factors)) == 1:
        return np.log(factors[0])
    else:
        return 0.0


def mobius(n: int) -> int:
    """
    Möbius function μ(n).
    
    μ(n) = 1 if n is square-free with even number of prime factors
    μ(n) = -1 if n is square-free with odd number of prime factors
    μ(n) = 0 if n has a squared prime factor
    
    Args:
        n: Input integer
        
    Returns:
        μ(n) ∈ {-1, 0, 1}
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
        
    factors = prime_factors(n)
    
    # Check for squared prime factor
    if len(factors) != len(set(factors)):
        return 0
        
    # Square-free: return (-1)^k where k = number of prime factors
    return (-1) ** len(factors)


def encode_primes_to_field(
    n_max: int,
    d_field: int = 16,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Encode prime number structure into cognitive tensor field.
    
    Each integer n ∈ [1, n_max) is mapped to a d_field × d_field complex matrix:
    - T[n, 0, 0] = Λ(n) + i·μ(n)  (von Mangoldt + Möbius)
    - T[n, k, k] = log(p_k) for kth prime factor (diagonal encoding)
    
    Args:
        n_max: Maximum number to consider
        d_field: Field dimension (tensor DOF)
        device: Torch device ('cpu' or 'cuda')
        
    Returns:
        T: Field tensor [n_max, d_field, d_field] (complex)
    """
    T = torch.zeros(n_max, d_field, d_field, dtype=torch.complex64, device=device)
    
    # Precompute primes
    primes = sieve_of_eratosthenes(n_max)
    primes_set = set(primes)
    
    for n in range(1, n_max):
        # Von Mangoldt function
        lambda_n = von_mangoldt(n)
        
        # Möbius function
        mu_n = mobius(n)
        
        # Encode into field tensor
        # Real part: von Mangoldt (prime content)
        # Imaginary part: Möbius (multiplicative parity)
        T[n, 0, 0] = complex(lambda_n, mu_n)
        
        # Off-diagonal: divisor structure
        factors = prime_factors(n)
        unique_factors = list(dict.fromkeys(factors))  # Remove duplicates, preserve order
        
        for i, p in enumerate(unique_factors):
            if i + 1 < d_field:
                # Encode prime p at position (i+1, i+1)
                T[n, i+1, i+1] = np.log(p)
                
        # Add small random perturbation for full-rank (optional)
        # T[n] += 0.01 * torch.randn(d_field, d_field, dtype=torch.complex64, device=device)
                
    return T


def spectral_decomposition(
    T_field: torch.Tensor,
    return_full: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Perform spectral decomposition of prime-encoded field.
    
    Computes eigenvalues, SVD, effective rank, and entropy for each n.
    
    Args:
        T_field: Field tensor [n_max, d_field, d_field]
        return_full: If True, return full eigenvectors/singular vectors
        
    Returns:
        Dictionary with:
        - eigenvalues: [n_max, d_field]
        - singular_values: [n_max, d_field]
        - effective_rank: [n_max]
        - entropy: [n_max]
        - spectral_density: [n_max, d_field]
    """
    n_max, d_field, _ = T_field.shape
    device = T_field.device
    
    eigenvalues_list = []
    singular_values_list = []
    
    for n in range(n_max):
        T_n = T_field[n]
        
        # Make Hermitian for eigenvalue decomposition
        T_herm = (T_n + T_n.conj().T) / 2
        
        # Compute eigenvalues (sorted descending by absolute value)
        try:
            eigs = torch.linalg.eigvalsh(T_herm)
            eigs = torch.sort(torch.abs(eigs), descending=True)[0]
        except:
            eigs = torch.zeros(d_field, device=device)
            
        eigenvalues_list.append(eigs)
        
        # Compute singular values
        try:
            U, S, Vh = torch.linalg.svd(T_n, full_matrices=False)
            singular_values_list.append(S)
        except:
            singular_values_list.append(torch.zeros(d_field, device=device))
        
    eigenvalues = torch.stack(eigenvalues_list)  # [n_max, d_field]
    singular_values = torch.stack(singular_values_list)  # [n_max, d_field]
    
    # Spectral density (use singular values as they're always non-negative)
    spectral_density = singular_values
    
    # Effective rank (participation ratio)
    epsilon = 1e-10
    sum_sv = spectral_density.sum(dim=1) + epsilon
    sum_sv_sq = (spectral_density ** 2).sum(dim=1) + epsilon
    effective_rank = sum_sv**2 / sum_sv_sq
    
    # Von Neumann entropy
    sv_normalized = spectral_density / sum_sv.unsqueeze(1)
    sv_normalized = sv_normalized.clamp(min=epsilon)
    entropy = -torch.sum(sv_normalized * torch.log(sv_normalized), dim=1)
    
    result = {
        'eigenvalues': eigenvalues,
        'singular_values': singular_values,
        'effective_rank': effective_rank,
        'entropy': entropy,
        'spectral_density': spectral_density,
    }
    
    return result


def extract_oscillatory_modes(
    spectral_data: Dict[str, torch.Tensor],
    n_modes: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract dominant oscillatory modes from spectral data.
    
    These modes are analogous to the imaginary parts of Riemann zeta zeros.
    
    Args:
        spectral_data: Output from spectral_decomposition()
        n_modes: Number of dominant modes to extract
        
    Returns:
        frequencies: Dominant frequencies [n_modes]
        amplitudes: Corresponding amplitudes [n_modes]
    """
    eigenvalues = spectral_data['eigenvalues']  # [n_max, d_field]
    
    # Sum over field dimensions to get total spectral weight per n
    spectral_weight = eigenvalues.sum(dim=1)  # [n_max]
    
    # Fourier transform to extract frequencies
    eig_fourier = torch.fft.fft(spectral_weight)
    frequencies = torch.fft.fftfreq(spectral_weight.shape[0], d=1.0)
    
    # Power spectrum
    power = torch.abs(eig_fourier) ** 2
    
    # Find dominant modes
    top_modes_idx = torch.argsort(power, descending=True)[:n_modes]
    
    # Extract frequencies and amplitudes of dominant modes
    dominant_frequencies = frequencies[top_modes_idx]
    dominant_amplitudes = torch.abs(eig_fourier[top_modes_idx])
    
    # Sort by frequency (for easier interpretation)
    sort_idx = torch.argsort(torch.abs(dominant_frequencies))
    dominant_frequencies = dominant_frequencies[sort_idx]
    dominant_amplitudes = dominant_amplitudes[sort_idx]
    
    return dominant_frequencies, dominant_amplitudes


def compute_level_spacing_statistics(
    eigenvalues: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute level spacing statistics for comparison with random matrix theory.
    
    Args:
        eigenvalues: Eigenvalue tensor [n_max, d_field] or [d_field]
        normalize: If True, normalize spacings by mean spacing
        
    Returns:
        spacings: Normalized spacings
    """
    if eigenvalues.dim() == 2:
        # Flatten all eigenvalues
        eigs = eigenvalues.flatten()
    else:
        eigs = eigenvalues
        
    # Sort eigenvalues
    eigs_sorted = torch.sort(eigs)[0]
    
    # Compute spacings
    spacings = torch.diff(eigs_sorted)
    
    # Remove zero or negative spacings
    spacings = spacings[spacings > 1e-10]
    
    if normalize:
        mean_spacing = spacings.mean()
        spacings = spacings / (mean_spacing + 1e-10)
        
    return spacings


def gue_distribution(s: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Unitary Ensemble (GUE) level spacing distribution.
    
    P_GUE(s) = (32/π²) s² exp(-4s²/π)
    
    Args:
        s: Normalized spacing values
        
    Returns:
        P(s): Probability density
    """
    return (32 / np.pi**2) * s**2 * torch.exp(-4 * s**2 / np.pi)


def visualize_prime_field(
    T_field: torch.Tensor,
    n_max: int = 1000,
    save_path: str = 'prime_field_visualization.png'
):
    """
    Visualize the prime-encoded field structure.
    
    Args:
        T_field: Field tensor [n_max_total, d_field, d_field]
        n_max: Number of points to plot
        save_path: Path to save figure
    """
    n_max = min(n_max, T_field.shape[0])
    
    # Extract diagonal (main prime encoding)
    T_diag = T_field[:n_max, 0, 0].cpu()
    
    # Real and imaginary parts
    real_part = T_diag.real.numpy()  # von Mangoldt
    imag_part = T_diag.imag.numpy()  # Möbius
    
    # Magnitude and phase
    magnitude = torch.abs(T_diag).numpy()
    phase = torch.angle(T_diag).numpy()
    
    # Prime counting function (cumulative)
    cumulative = np.cumsum(real_part)
    
    # Theoretical π(n) ~ n / log(n) (Prime Number Theorem)
    n_range = np.arange(2, n_max + 1)
    theoretical = n_range / np.log(n_range)
    theoretical_cumsum = np.cumsum(np.concatenate([[0], theoretical[:-1]]))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: von Mangoldt function (real part)
    ax1 = axes[0, 0]
    ax1.stem(range(n_max), real_part, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax1.set_ylabel('Λ(n) [von Mangoldt]', fontsize=10)
    ax1.set_title('Prime Content (von Mangoldt Function)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_max)
    
    # Plot 2: Möbius function (imaginary part)
    ax2 = axes[0, 1]
    ax2.stem(range(n_max), imag_part, linefmt='r-', markerfmt='ro', basefmt=' ')
    ax2.set_ylabel('μ(n) [Möbius]', fontsize=10)
    ax2.set_title('Multiplicative Structure (Möbius Function)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_max)
    ax2.set_ylim(-1.5, 1.5)
    
    # Plot 3: Magnitude (combined)
    ax3 = axes[1, 0]
    ax3.plot(magnitude, 'g-', alpha=0.7, linewidth=1)
    ax3.set_ylabel('|T(n)|', fontsize=10)
    ax3.set_xlabel('n', fontsize=10)
    ax3.set_title('Field Magnitude', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, n_max)
    
    # Plot 4: Cumulative (prime counting)
    ax4 = axes[1, 1]
    ax4.plot(cumulative, 'b-', label='Cumulative Λ(n)', alpha=0.8, linewidth=1.5)
    ax4.plot(theoretical_cumsum, 'r--', label='π(n) ~ n/log(n)', alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('n', fontsize=10)
    ax4.set_ylabel('Cumulative Sum', fontsize=10)
    ax4.set_title('Prime Counting Function Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, n_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_spectral_analysis(
    spectral_data: Dict[str, torch.Tensor],
    save_path: str = 'spectral_prime_analysis.png'
):
    """
    Visualize spectral analysis results.
    
    Args:
        spectral_data: Output from spectral_decomposition()
        save_path: Path to save figure
    """
    effective_rank = spectral_data['effective_rank'].cpu().numpy()
    entropy = spectral_data['entropy'].cpu().numpy()
    spectral_density = spectral_data['spectral_density'].cpu().numpy()
    
    n_max = len(effective_rank)
    n_plot = min(1000, n_max)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Effective rank
    ax1 = axes[0, 0]
    ax1.plot(effective_rank[:n_plot], 'b-', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Effective Rank', fontsize=10)
    ax1.set_title('Effective Dimension (Participation Ratio)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy
    ax2 = axes[0, 1]
    ax2.plot(entropy[:n_plot], 'r-', alpha=0.7, linewidth=1)
    ax2.set_ylabel('Entropy (nats)', fontsize=10)
    ax2.set_title('Von Neumann Entropy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spectral density heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(spectral_density[:n_plot, :].T, aspect='auto', cmap='viridis',
                    interpolation='nearest', origin='lower')
    ax3.set_xlabel('n', fontsize=10)
    ax3.set_ylabel('Mode index', fontsize=10)
    ax3.set_title('Spectral Density (Singular Values)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Magnitude')
    
    # Plot 4: Singular value decay
    ax4 = axes[1, 1]
    # Average singular value spectrum
    avg_spectrum = spectral_density.mean(axis=0)
    ax4.semilogy(avg_spectrum, 'go-', markersize=4, linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('Mode index', fontsize=10)
    ax4.set_ylabel('Average Singular Value (log scale)', fontsize=10)
    ax4.set_title('Spectral Decay', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_level_spacings(
    spacings: torch.Tensor,
    save_path: str = 'level_spacing_statistics.png'
):
    """
    Visualize level spacing distribution and compare with GUE.
    
    Args:
        spacings: Normalized eigenvalue spacings
        save_path: Path to save figure
    """
    spacings_np = spacings.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of spacings
    counts, bins, _ = ax.hist(spacings_np, bins=50, density=True, alpha=0.6,
                               color='blue', edgecolor='black', label='Observed')
    
    # GUE prediction
    s_theory = torch.linspace(0, bins[-1], 200)
    p_gue = gue_distribution(s_theory)
    ax.plot(s_theory.numpy(), p_gue.numpy(), 'r-', linewidth=2, label='GUE')
    
    # Poisson (for comparison)
    p_poisson = torch.exp(-s_theory)
    ax.plot(s_theory.numpy(), p_poisson.numpy(), 'g--', linewidth=2, label='Poisson')
    
    ax.set_xlabel('Normalized Spacing s', fontsize=12)
    ax.set_ylabel('Probability Density P(s)', fontsize=12)
    ax.set_title('Level Spacing Statistics vs Random Matrix Theory', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def run_full_analysis(
    n_max: int = 10000,
    d_field: int = 16,
    output_dir: str = '.',
    device: str = 'cpu'
) -> Dict:
    """
    Run complete spectral prime analysis pipeline.
    
    Args:
        n_max: Maximum number to analyze
        d_field: Field tensor dimension
        output_dir: Directory to save visualizations
        device: Torch device
        
    Returns:
        Dictionary with all analysis results
    """
    print(f"Starting spectral prime analysis for n ∈ [1, {n_max})")
    print(f"Field dimension: {d_field}")
    print(f"Device: {device}")
    
    # Step 1: Encode primes
    print("\n[1/5] Encoding primes to field...")
    T_field = encode_primes_to_field(n_max, d_field, device)
    print(f"  Field shape: {T_field.shape}")
    
    # Step 2: Spectral decomposition
    print("\n[2/5] Computing spectral decomposition...")
    spectral_data = spectral_decomposition(T_field)
    print(f"  Mean effective rank: {spectral_data['effective_rank'].mean().item():.2f}")
    print(f"  Mean entropy: {spectral_data['entropy'].mean().item():.2f}")
    
    # Step 3: Extract modes
    print("\n[3/5] Extracting oscillatory modes...")
    frequencies, amplitudes = extract_oscillatory_modes(spectral_data, n_modes=100)
    print(f"  Top 5 frequencies: {frequencies[:5].cpu().numpy()}")
    
    # Step 4: Level spacing statistics
    print("\n[4/5] Computing level spacing statistics...")
    spacings = compute_level_spacing_statistics(spectral_data['eigenvalues'])
    print(f"  Mean spacing: {spacings.mean().item():.4f}")
    print(f"  Std spacing: {spacings.std().item():.4f}")
    
    # Step 5: Visualizations
    print("\n[5/5] Generating visualizations...")
    import os
    
    visualize_prime_field(
        T_field, 
        n_max=min(1000, n_max),
        save_path=os.path.join(output_dir, 'prime_field_visualization.png')
    )
    
    visualize_spectral_analysis(
        spectral_data,
        save_path=os.path.join(output_dir, 'spectral_prime_analysis.png')
    )
    
    visualize_level_spacings(
        spacings,
        save_path=os.path.join(output_dir, 'level_spacing_statistics.png')
    )
    
    print("\n✓ Analysis complete!")
    
    return {
        'field': T_field,
        'spectral_data': spectral_data,
        'frequencies': frequencies,
        'amplitudes': amplitudes,
        'spacings': spacings,
    }


if __name__ == '__main__':
    # Run analysis on first 10,000 integers
    results = run_full_analysis(n_max=10000, d_field=16, output_dir='.')
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Analyzed n ∈ [1, 10000)")
    print(f"Mean effective rank: {results['spectral_data']['effective_rank'].mean():.2f}")
    print(f"Mean entropy: {results['spectral_data']['entropy'].mean():.2f}")
    print(f"Number of dominant modes: {len(results['frequencies'])}")
    print("\nVisualizations saved:")
    print("  - prime_field_visualization.png")
    print("  - spectral_prime_analysis.png")
    print("  - level_spacing_statistics.png")
