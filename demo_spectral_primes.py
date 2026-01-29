#!/usr/bin/env python3
"""
Demo: Spectral Analysis of Prime Numbers on Meromorphic Fields

This script demonstrates how the LIoRHybrid spectral framework can be used
to analyze prime number distribution through the lens of quantum field theory
and spectral decomposition.

Usage:
    python demo_spectral_primes.py [--n-max N] [--d-field D] [--device DEVICE]
    
Examples:
    python demo_spectral_primes.py --n-max 5000 --d-field 16
    python demo_spectral_primes.py --n-max 20000 --d-field 32 --device cuda
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import argparse
import sys
import os
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.spectral_primes import (
    encode_primes_to_field,
    spectral_decomposition,
    extract_oscillatory_modes,
    compute_level_spacing_statistics,
    visualize_prime_field,
    visualize_spectral_analysis,
    visualize_level_spacings,
    run_full_analysis,
    sieve_of_eratosthenes,
)


def demo_basic_encoding(n_max=1000, d_field=16):
    """
    Demonstrate basic prime encoding into tensor field.
    """
    print("\n" + "="*70)
    print("DEMO 1: Prime Number Encoding")
    print("="*70)
    
    print(f"\nEncoding primes up to {n_max} into {d_field}×{d_field} tensor field...")
    
    T_field = encode_primes_to_field(n_max, d_field)
    
    print(f"✓ Field tensor shape: {T_field.shape}")
    print(f"✓ Data type: {T_field.dtype}")
    
    # Show some examples
    print("\nExample encodings:")
    for n in [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10]:
        T_n = T_field[n, 0, 0]
        real_part = T_n.real.item()
        imag_part = T_n.imag.item()
        print(f"  n={n:3d}: T(n) = {real_part:6.3f} + {imag_part:+6.3f}i  "
              f"(|T|={abs(T_n.item()):.3f})")
    
    # Statistics
    T_diag = T_field[:, 0, 0]
    nonzero = torch.sum(torch.abs(T_diag) > 0.01).item()
    print(f"\nStatistics:")
    print(f"  Non-zero entries: {nonzero} / {n_max} ({100*nonzero/n_max:.1f}%)")
    print(f"  Mean |T(n)|: {torch.abs(T_diag).mean().item():.4f}")
    print(f"  Max |T(n)|: {torch.abs(T_diag).max().item():.4f}")


def demo_spectral_decomposition(n_max=1000, d_field=16):
    """
    Demonstrate spectral decomposition of prime field.
    """
    print("\n" + "="*70)
    print("DEMO 2: Spectral Decomposition")
    print("="*70)
    
    print(f"\nPerforming spectral analysis...")
    
    T_field = encode_primes_to_field(n_max, d_field)
    spectral_data = spectral_decomposition(T_field)
    
    print("✓ Spectral decomposition complete")
    
    # Statistics
    eff_rank = spectral_data['effective_rank']
    entropy = spectral_data['entropy']
    
    print(f"\nSpectral Statistics:")
    print(f"  Effective rank - Mean: {eff_rank.mean().item():.2f}, "
          f"Std: {eff_rank.std().item():.2f}")
    print(f"  Entropy - Mean: {entropy.mean().item():.2f}, "
          f"Std: {entropy.std().item():.2f}")
    
    # Show top singular values for a few numbers
    print(f"\nTop 5 singular values for selected n:")
    for n in [2, 10, 100, 500]:
        if n < n_max:
            sv = spectral_data['singular_values'][n, :5]
            print(f"  n={n:4d}: {sv.cpu().numpy()}")


def demo_zeta_modes(n_max=5000, d_field=16):
    """
    Demonstrate extraction of zeta-like oscillatory modes.
    """
    print("\n" + "="*70)
    print("DEMO 3: Zeta-Like Oscillatory Modes")
    print("="*70)
    
    print(f"\nExtracting oscillatory modes (analogous to Riemann zeta zeros)...")
    
    T_field = encode_primes_to_field(n_max, d_field)
    spectral_data = spectral_decomposition(T_field)
    frequencies, amplitudes = extract_oscillatory_modes(spectral_data, n_modes=50)
    
    print("✓ Mode extraction complete")
    
    # Show dominant modes
    print(f"\nTop 10 dominant modes:")
    print(f"{'Rank':<6} {'Frequency':<15} {'Amplitude':<15} {'Period':<15}")
    print("-" * 60)
    for i in range(min(10, len(frequencies))):
        freq = frequencies[i].item()
        amp = amplitudes[i].item()
        period = 1 / abs(freq) if abs(freq) > 1e-10 else float('inf')
        print(f"{i+1:<6} {freq:<15.6f} {amp:<15.2f} {period:<15.2f}")
    
    # Compare to known zeta zeros (first 10 imaginary parts, truncated list)
    # Full list available at: https://www.lmfdb.org/zeros/zeta/
    # Note: This is only a small sample for demonstration purposes
    known_zeta_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                        37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
    
    print(f"\nKnown Riemann zeta zeros (first 10 imaginary parts):")
    for i, z in enumerate(known_zeta_zeros):
        print(f"  ρ_{i+1}: Im(ρ) = {z:.6f}")
    
    print(f"\nNote: Direct comparison requires n_max ≫ 10^4 for convergence.")


def demo_gue_statistics(n_max=2000, d_field=16):
    """
    Demonstrate GUE statistics comparison (Montgomery-Odlyzko conjecture).
    """
    print("\n" + "="*70)
    print("DEMO 4: Random Matrix Statistics (GUE)")
    print("="*70)
    
    print(f"\nTesting Montgomery-Odlyzko conjecture (GUE statistics)...")
    
    T_field = encode_primes_to_field(n_max, d_field)
    spectral_data = spectral_decomposition(T_field)
    spacings = compute_level_spacing_statistics(spectral_data['eigenvalues'])
    
    print("✓ Level spacing statistics computed")
    
    # Statistics
    mean_s = spacings.mean().item()
    std_s = spacings.std().item()
    
    print(f"\nSpacing Statistics:")
    print(f"  Number of spacings: {len(spacings)}")
    print(f"  Mean spacing: {mean_s:.4f} (should be ~1 if normalized)")
    print(f"  Std spacing: {std_s:.4f}")
    
    # GUE prediction for mean and std
    # For GUE: <s> = 1, σ(s) ≈ 0.52
    print(f"\nGUE Prediction:")
    print(f"  Expected mean: 1.000")
    print(f"  Expected std: ~0.52")
    print(f"  Ratio (observed/expected): {std_s/0.52:.2f}")
    
    # Check level repulsion (should have P(0) ≈ 0 for GUE, not Poisson)
    small_spacings = torch.sum(spacings < 0.1).item()
    print(f"\nLevel Repulsion Test:")
    print(f"  Spacings < 0.1: {small_spacings} ({100*small_spacings/len(spacings):.2f}%)")
    print(f"  (GUE predicts low values near 0, Poisson predicts many)")


def demo_prime_number_theorem(n_max=10000):
    """
    Demonstrate Prime Number Theorem using field encoding.
    """
    print("\n" + "="*70)
    print("DEMO 5: Prime Number Theorem Verification")
    print("="*70)
    
    print(f"\nTesting Prime Number Theorem: π(n) ~ n / ln(n)...")
    
    # Get actual primes
    primes = sieve_of_eratosthenes(n_max)
    pi_n = len(primes)
    
    # Theoretical prediction
    pnt_prediction = n_max / np.log(n_max)
    
    # Relative error
    error = abs(pi_n - pnt_prediction) / pi_n * 100
    
    print(f"\nResults:")
    print(f"  n_max: {n_max}")
    print(f"  π(n) [actual]: {pi_n}")
    print(f"  n/ln(n) [PNT]: {pnt_prediction:.1f}")
    print(f"  Relative error: {error:.2f}%")
    
    # Also test at several checkpoints
    print(f"\nCheckpoints:")
    print(f"{'n':<10} {'π(n)':<10} {'n/ln(n)':<12} {'Error %':<10}")
    print("-" * 50)
    for n in [100, 500, 1000, 2000, 5000, n_max]:
        primes_n = sieve_of_eratosthenes(n + 1)
        pi_actual = len(primes_n)
        pi_theory = n / np.log(n)
        err = abs(pi_actual - pi_theory) / pi_actual * 100
        print(f"{n:<10} {pi_actual:<10} {pi_theory:<12.1f} {err:<10.2f}")


def main():
    """
    Main demonstration script.
    """
    parser = argparse.ArgumentParser(
        description='Demonstrate spectral analysis of prime numbers on meromorphic fields'
    )
    parser.add_argument('--n-max', type=int, default=5000,
                        help='Maximum number to analyze (default: 5000)')
    parser.add_argument('--d-field', type=int, default=16,
                        help='Field tensor dimension (default: 16)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Computation device (default: cpu)')
    parser.add_argument('--demo', type=str, default='all',
                        choices=['all', 'encoding', 'spectral', 'zeta', 'gue', 'pnt', 'full'],
                        help='Which demo to run (default: all)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*70)
    print("SPECTRAL ANALYSIS OF PRIME NUMBERS")
    print("LIoRHybrid Physics Framework")
    print("="*70)
    print(f"\nParameters:")
    print(f"  n_max: {args.n_max}")
    print(f"  d_field: {args.d_field}")
    print(f"  device: {args.device}")
    print(f"  demo: {args.demo}")
    
    # Run selected demos
    if args.demo in ['all', 'encoding']:
        demo_basic_encoding(min(1000, args.n_max), args.d_field)
    
    if args.demo in ['all', 'spectral']:
        demo_spectral_decomposition(min(1000, args.n_max), args.d_field)
    
    if args.demo in ['all', 'zeta']:
        demo_zeta_modes(args.n_max, args.d_field)
    
    if args.demo in ['all', 'gue']:
        demo_gue_statistics(min(2000, args.n_max), args.d_field)
    
    if args.demo in ['all', 'pnt']:
        demo_prime_number_theorem(args.n_max)
    
    if args.demo == 'full':
        print("\n" + "="*70)
        print("FULL ANALYSIS PIPELINE")
        print("="*70)
        results = run_full_analysis(
            n_max=args.n_max,
            d_field=args.d_field,
            output_dir='.',
            device=args.device
        )
        print("\nAll visualizations generated!")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nFor full analysis with visualizations, run:")
    print(f"  python demo_spectral_primes.py --demo full --n-max {args.n_max}")
    print("\nFor theoretical background, see:")
    print("  SPECTRAL_PRIME_ANALYSIS.md")


if __name__ == '__main__':
    main()
