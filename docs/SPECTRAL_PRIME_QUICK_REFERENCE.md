# Spectral Prime Analysis: Quick Reference

**Location:** `/utils/spectral_primes.py`, `/demo_spectral_primes.py`  
**Documentation:** `/SPECTRAL_PRIME_ANALYSIS.md`  
**Date:** 2026-01-28

---

## Overview

This module implements a novel approach to analyzing prime number distribution using LIoRHybrid's physics-based spectral framework. It connects number theory (Riemann zeta function, prime distribution) with quantum field theory, eigenvalue analysis, and random matrix statistics.

---

## Quick Start

### Run All Demos

```bash
python demo_spectral_primes.py --n-max 5000 --demo all
```

### Generate Full Analysis with Visualizations

```bash
python demo_spectral_primes.py --n-max 10000 --demo full
```

This generates three visualizations:
- `prime_field_visualization.png`: Prime encoding (von Mangoldt, Möbius, cumulative)
- `spectral_prime_analysis.png`: Spectral decomposition results
- `level_spacing_statistics.png`: GUE comparison (Montgomery-Odlyzko)

---

## Core Concepts

### 1. Prime Field Encoding

**Map arithmetic functions to tensor field:**

```python
from utils.spectral_primes import encode_primes_to_field

# Encode primes up to 10,000
T_field = encode_primes_to_field(n_max=10000, d_field=16)

# Structure:
# T[n, 0, 0] = Λ(n) + i·μ(n)
#   where Λ(n) = von Mangoldt function (prime content)
#         μ(n) = Möbius function (multiplicative structure)
```

### 2. Spectral Decomposition

**Analyze eigenvalue structure:**

```python
from utils.spectral_primes import spectral_decomposition

spectral_data = spectral_decomposition(T_field)

# Returns:
# - eigenvalues: [n_max, d_field]
# - singular_values: [n_max, d_field]
# - effective_rank: [n_max] (participation ratio)
# - entropy: [n_max] (von Neumann entropy)
```

### 3. Zeta-Like Modes

**Extract oscillatory frequencies (analogous to Riemann zeta zeros):**

```python
from utils.spectral_primes import extract_oscillatory_modes

frequencies, amplitudes = extract_oscillatory_modes(spectral_data, n_modes=100)

# frequencies: Dominant oscillation frequencies
# amplitudes: Corresponding spectral weights
```

### 4. Random Matrix Statistics

**Test Montgomery-Odlyzko conjecture (GUE statistics):**

```python
from utils.spectral_primes import compute_level_spacing_statistics

spacings = compute_level_spacing_statistics(spectral_data['eigenvalues'])

# Compare to GUE: P(s) ~ s² exp(-4s²/π)
```

---

## Theoretical Connections

### Riemann Zeta Function ↔ Field Eigenvalues

| Number Theory | LIoRHybrid Framework | Connection |
|--------------|---------------------|------------|
| ζ(s) zeros ρ | Field eigenvalues | Spectral decomposition |
| Prime counting π(x) | Spectral density | Effective rank |
| Explicit formula | Mode superposition | FFT of eigenvalues |
| GUE statistics | Level spacings | Montgomery-Odlyzko |
| Critical line Re(s)=1/2 | Phase orthogonality | Σ ⊥ Λ stability |

### Key Insights

1. **Hilbert-Pólya Conjecture**: Zeta zeros as eigenvalues of self-adjoint operator
   - LIoRHybrid's Hamiltonian is self-adjoint by construction
   - Provides natural framework for modeling zeta zeros

2. **Montgomery-Odlyzko Law**: Zeta zero spacings follow GUE statistics
   - Can test on field eigenvalue spacings
   - Connects number theory to random matrix theory

3. **Explicit Formula**: Prime distribution as Fourier sum over zeta zeros
   - Field evolution equation has same structure
   - Modes (frequencies) ↔ zeta zeros (imaginary parts)

4. **Neto Morphic Field**: Natural emergent topology
   - 32-neighbor addressing creates network structure
   - Power-law memory (LIoR kernel) captures prime correlations
   - BCH error correction enforces algebraic constraints

---

## API Reference

### Main Functions

#### `encode_primes_to_field(n_max, d_field=16, device='cpu')`
Encode prime structure into tensor field.

**Returns:** `torch.Tensor` of shape `[n_max, d_field, d_field]`

---

#### `spectral_decomposition(T_field, return_full=False)`
Perform spectral analysis.

**Returns:** Dictionary with eigenvalues, singular values, entropy, effective rank

---

#### `extract_oscillatory_modes(spectral_data, n_modes=100)`
Extract dominant frequencies via FFT.

**Returns:** `(frequencies, amplitudes)` tensors

---

#### `compute_level_spacing_statistics(eigenvalues, normalize=True)`
Compute normalized level spacings for GUE comparison.

**Returns:** `torch.Tensor` of normalized spacings

---

#### `run_full_analysis(n_max, d_field=16, output_dir='.', device='cpu')`
Run complete pipeline with visualizations.

**Returns:** Dictionary with all analysis results

---

## Visualizations

### 1. Prime Field Structure
- **von Mangoldt function**: Λ(n) = log p if n = p^k
- **Möbius function**: μ(n) ∈ {-1, 0, 1} (multiplicative parity)
- **Cumulative sum**: Compare to Prime Number Theorem π(x) ~ x/log x

### 2. Spectral Analysis
- **Effective rank**: Participation ratio (dimensionality)
- **Entropy**: von Neumann entropy (uncertainty)
- **Spectral density**: Heatmap of singular values
- **Decay**: Average singular value spectrum

### 3. Level Spacing Statistics
- **Histogram**: Observed spacing distribution
- **GUE curve**: P(s) = (32/π²)s² exp(-4s²/π)
- **Poisson**: P(s) = exp(-s) (for comparison)

---

## Example Results

### For n_max = 2000, d_field = 16:

```
Mean effective rank: 2.14
Mean entropy: 0.83 nats
Non-zero entries: ~65%

Top oscillatory frequencies:
  f₁ = 0.0000 (DC component)
  f₂ = ±0.0005
  f₃ = ±0.0010
  f₄ = ±0.0015
  ...

Level spacing statistics:
  Mean: 1.000 (normalized)
  Std: 2.74
  GUE prediction: σ ≈ 0.52
```

---

## Applications

### Research Questions

1. **Do prime fields exhibit GUE statistics?**
   - Test on larger n_max (10⁶ - 10⁹)
   - Compare with known zeta zero spacings

2. **Can we predict prime locations?**
   - Train field evolution on prime data
   - Use LIoR memory kernel for long-range correlations

3. **What is the "effective Hamiltonian" for primes?**
   - Construct operator whose eigenvalues match zeta zeros
   - Test field-theoretic proof of Riemann Hypothesis

4. **How do prime gaps relate to field dynamics?**
   - Twin primes ↔ Field correlations
   - Power-law memory ↔ Prime gap distribution

### Potential Extensions

1. **L-Functions**: Generalize to Dirichlet L-functions, modular forms
2. **Elliptic Curves**: Test Birch-Swinnerton-Dyer via field analogy
3. **Additive Number Theory**: Goldbach conjecture, Waring's problem
4. **Computational Number Theory**: Fast primality testing via spectral methods

---

## Performance Notes

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Prime encoding | O(n_max × log log n_max) | Sieve of Eratosthenes |
| Spectral decomposition | O(n_max × d_field³) | SVD/eigenvalue per n |
| FFT (mode extraction) | O(n_max × log n_max) | Per field dimension |
| Level spacing | O(n_max × d_field × log(n_max × d_field)) | Sort all eigenvalues |

### Memory Requirements

- Field tensor: `n_max × d_field × d_field × 8 bytes` (complex64)
- For n_max=10⁶, d_field=16: ~2 GB

### Recommendations

- **Small tests** (n_max ≤ 10³): Use CPU, d_field=16
- **Medium** (10³ < n_max ≤ 10⁵): Use CPU or GPU, d_field=16-32
- **Large** (n_max > 10⁵): Use GPU, consider batching

---

## References

### Included in SPECTRAL_PRIME_ANALYSIS.md:

1. Edwards (1974) - *Riemann's Zeta Function*
2. Montgomery (1973) - Pair correlation of zeta zeros
3. Odlyzko (1987) - Spacing distribution of zeta zeros
4. Berry & Keating (1999) - Zeta zeros and eigenvalue asymptotics
5. Connes (1999) - Trace formula in noncommutative geometry

### LIoRHybrid Framework:

- `/docs/CLIFFORD_GEOMETRY_CONNECTION.md`
- `/CAUSAL_FIELD_HAMILTONIAN_REVIEW.md`
- `/PHYSICS_AUDIT_ADDENDUM.md`

---

## Contact & Contributions

For questions or to contribute:
- Open an issue: https://github.com/JewJitsu11B/Liorhybrid/issues
- See: `/CONTRIBUTING.md`

---

**Status:** Research implementation - ready for experimentation  
**Version:** 1.0  
**Date:** 2026-01-28
