# Rank Reduction for Physics Computations

**Date:** 2026-01-08  
**Context:** O(D³) complexity in vectorized physics computations  
**Goal:** Reduce computational cost while preserving physics accuracy

## The D³ Problem

### Current Complexity

**Physics computations with O(D³) scaling:**

1. **Energy computation:**
   ```python
   # T_dag_H = torch.einsum('xyij,xyjk->xyik', T_dag, H_T)  # O(N_x·N_y·D³)
   # traces = torch.einsum('xyii->xy', T_dag_H)
   ```

2. **Unitarity deviation:**
   ```python
   # T_dag_T = torch.einsum('xyij,xyjk->xyik', T_dag, self.T)  # O(N_x·N_y·D³)
   # traces = torch.einsum('xyii->xy', T_dag_T)
   ```

3. **Entropy (eigendecomposition):**
   ```python
   # eigenvalues = torch.linalg.eigvalsh(rho_batch[valid_mask])  # O(n_valid·D³)
   ```

**Where:**
- D = tensor_dim (typically 16-32)
- N_x, N_y = spatial grid (typically 8x8 to 28x28)
- D³ dominates for large D

### Why This Matters

**Scaling Analysis:**
- D=16: D³ = 4,096 ops per spatial point
- D=32: D³ = 32,768 ops per spatial point (8x worse)
- D=64: D³ = 262,144 ops per spatial point (64x worse)

**For typical field (28x28 grid, D=32):**
- Energy: 28×28×32³ = 25.6M ops
- Can dominate training time at large scales

## Rank Reduction Strategies

### 1. Low-Rank Field Approximation

**Observation:** Cognitive field T_ij may have low effective rank

**Strategy:** Compress T using SVD/randomized projection
```python
T_full ∈ ℂ^(N_x × N_y × D × D)  # Full field
T_lr = U @ Σ @ V†  where Σ has rank r << D  # Low-rank approximation
```

**Benefits:**
- Matrix products: O(D³) → O(D·r²) if r << D
- Energy computation: O(N·D³) → O(N·D·r²)
- Physics approximately preserved if singular values decay

**Implementation:**
```python
def compress_field_lowrank(T, target_rank):
    """
    Compress field to low-rank representation.
    
    T: (N_x, N_y, D, D) complex
    Returns: U, S, V such that T ≈ U @ diag(S) @ V†
    """
    N_x, N_y, D, _ = T.shape
    
    # Reshape to (N_x*N_y, D, D)
    T_batch = T.reshape(-1, D, D)
    
    # Batch SVD
    U, S, Vh = torch.linalg.svd(T_batch, full_matrices=False)
    
    # Truncate to target rank
    U_r = U[:, :, :target_rank]  # (N, D, r)
    S_r = S[:, :target_rank]      # (N, r)
    Vh_r = Vh[:, :target_rank, :] # (N, r, D)
    
    return U_r, S_r, Vh_r
```

### 2. Adaptive Rank Selection

**Strategy:** Choose rank based on singular value spectrum

**Criterion:** Keep singular values where σ_i/σ_max > threshold

```python
def adaptive_rank(S, threshold=1e-6):
    """
    Select rank based on relative singular value magnitude.
    """
    sigma_max = S.max(dim=-1, keepdim=True)[0]
    mask = (S / sigma_max) > threshold
    ranks = mask.sum(dim=-1)  # Per-point rank
    return ranks.max().item()  # Conservative: use max rank
```

**Benefits:**
- Automatically adjusts to field complexity
- Preserves accuracy where needed
- Reduces rank in smooth regions

### 3. Hierarchical Compression

**Strategy:** Different compression for different scales

```python
class HierarchicalFieldCompression:
    """
    Multi-scale compression strategy.
    
    - Global structure: Low rank (r ≈ 4-8)
    - Local details: Higher rank (r ≈ 16-32)
    - High-frequency: Full rank (r = D)
    """
    
    def __init__(self, global_rank=8, local_rank=16, detail_rank=32):
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.detail_rank = detail_rank
    
    def compress(self, T):
        # Decompose into scales
        T_global = spatial_average(T)    # Coarse structure
        T_local = T - expand(T_global)   # Medium details
        T_detail = high_frequency(T)     # Fine details
        
        # Compress each scale
        U_g, S_g, V_g = svd_truncate(T_global, self.global_rank)
        U_l, S_l, V_l = svd_truncate(T_local, self.local_rank)
        # Keep detail at full rank or high rank
        
        return (U_g, S_g, V_g), (U_l, S_l, V_l), T_detail
```

### 4. Approximate Energy/Entropy

**Strategy:** Compute physics on low-rank approximation

**Energy with low-rank T:**
```python
def compute_energy_lowrank(U, S, V, H_T_lowrank):
    """
    Compute energy using low-rank approximation.
    
    E = Re[⟨T|H|T⟩] ≈ Re[⟨T_lr|H|T_lr⟩]
    where T_lr = U @ diag(S) @ V†
    """
    # Reconstruct in low-rank space
    # T† H T = (V·S·U†) H (U·S·V†)
    # = V·S·(U† H U)·S·V†
    # Only need U† H U which is (r × r) instead of (D × D)
    
    r = U.shape[-1]
    energy = 0.0
    
    for i in range(U.shape[0]):
        U_i = U[i]  # (D, r)
        S_i = S[i]  # (r,)
        V_i = V[i]  # (r, D)
        H_i = H_T_lowrank[i]  # (D, D)
        
        # U† H U: (r, D) @ (D, D) @ (D, r) = (r, r)
        UHU = U_i.conj().T @ H_i @ U_i  # O(D·r²) instead of O(D³)
        
        # Tr(T† H T) ≈ Tr(V·S·UHU·S·V†) = Tr(S·UHU·S)
        trace = torch.trace(torch.diag(S_i) @ UHU @ torch.diag(S_i))
        energy += torch.real(trace).item()
    
    return energy
```

**Complexity:**
- Full: O(N·D³)
- Low-rank: O(N·D·r²) where r << D
- **Speedup:** (D/r) factor

### 5. Streaming Rank Reduction

**Strategy:** Update low-rank decomposition incrementally

```python
class StreamingSVD:
    """
    Incremental SVD updates without full recomputation.
    
    Based on:
    - Brand (2006): Fast low-rank modifications of the thin SVD
    - Levy & Lindenbaum (2000): Sequential Karhunen-Loeve basis extraction
    """
    
    def __init__(self, target_rank):
        self.U = None
        self.S = None
        self.V = None
        self.target_rank = target_rank
    
    def update(self, new_T):
        """
        Update SVD with new field state.
        
        Complexity: O(D²·r) instead of O(D³)
        """
        if self.U is None:
            # Initial decomposition
            self.U, self.S, Vh = torch.linalg.svd(new_T, full_matrices=False)
            self.V = Vh.conj().T
            self._truncate()
        else:
            # Incremental update
            # Project new data onto current basis
            proj = self.U.conj().T @ new_T @ self.V
            
            # Update singular values
            self.S = 0.9 * self.S + 0.1 * torch.diag(proj)  # Exponential smoothing
            
            # Periodic full recomputation (every N steps)
            if self.step_count % 100 == 0:
                self._full_update(new_T)
```

## Physics Preservation Analysis

### What Can Be Safely Reduced?

**Safe for rank reduction:**
1. **Diagnostics** (energy, entropy, correlation): Approximate values acceptable
2. **Monitoring**: Don't affect training, just tracking
3. **Visualization**: Human interpretation tolerates approximation

**Requires caution:**
1. **Field evolution**: Affects training dynamics
2. **Gradients**: Need accurate backprop through physics
3. **Adaptive parameters**: Entropy gradients drive α, ν, τ

### Accuracy vs Speed Trade-off

**Rank selection guidelines:**

| Use Case | Rank | Accuracy | Speed | Notes |
|----------|------|----------|-------|-------|
| Monitoring | r=4-8 | ~95% | 10-50x | Good for dashboards |
| Training diagnostics | r=8-16 | ~98% | 4-10x | Balance accuracy/speed |
| Gradient computation | r=16-32 | ~99.5% | 2-4x | Conservative |
| Critical physics | Full (D) | 100% | 1x | When precision matters |

### Error Bounds

**Frobenius norm error:**
```
||T - T_r||_F ≤ sqrt(Σ_{i>r} σ_i²)
```

**Energy error bound:**
```
|E_full - E_approx| ≤ ||H||_op · ||T - T_r||_F²
```

**Strategy:** Monitor singular value spectrum, adapt rank dynamically

## Implementation Recommendations

### 1. Diagnostic-Only Rank Reduction (Conservative)

**Apply to:** Energy, entropy, correlation computations  
**Method:** Low-rank approximation with r=16  
**Impact:** 2-4x speedup, minimal accuracy loss  
**Risk:** Low - doesn't affect training

```python
class CognitiveTensorField:
    def compute_energy(self, use_lowrank=False, target_rank=16):
        if not use_lowrank:
            # Original full-rank computation
            return self._compute_energy_full()
        else:
            # Low-rank approximation
            U, S, V = self._compress_field(target_rank)
            return self._compute_energy_lowrank(U, S, V)
```

### 2. Adaptive Rank Selection (Moderate)

**Apply to:** All physics computations  
**Method:** Threshold-based rank selection  
**Impact:** Variable speedup (2-20x) based on field  
**Risk:** Moderate - need validation

```python
def compute_energy(self, auto_rank=True, rank_threshold=1e-4):
    if auto_rank:
        # Analyze field spectrum
        _, S, _ = torch.linalg.svd(self.T[0, 0])  # Sample point
        rank = (S / S[0] > rank_threshold).sum()
        rank = max(8, min(rank, self.config.tensor_dim))
    else:
        rank = self.config.tensor_dim
    
    return self._compute_with_rank(rank)
```

### 3. Hierarchical Compression (Aggressive)

**Apply to:** Field storage and all computations  
**Method:** Multi-scale compression  
**Impact:** 10-50x speedup, storage reduction  
**Risk:** High - affects training dynamics

**Only use if:**
- Field analysis shows low effective rank
- Extensive validation performed
- Accuracy requirements permit

## Validation Strategy

### Before Deploying Rank Reduction

**1. Analyze Field Spectrum:**
```python
def analyze_field_rank(field, n_samples=100):
    """
    Sample field at multiple points, compute rank statistics.
    """
    singular_values = []
    for _ in range(n_samples):
        x, y = random.randint(0, field.shape[0]-1), random.randint(0, field.shape[1]-1)
        T_xy = field.T[x, y]
        _, S, _ = torch.linalg.svd(T_xy)
        singular_values.append(S)
    
    S_mean = torch.stack(singular_values).mean(dim=0)
    effective_rank = (S_mean / S_mean[0] > 1e-3).sum()
    
    return S_mean, effective_rank
```

**2. Compare Accuracy:**
```python
def validate_lowrank_accuracy(field, ranks=[4, 8, 16, 32]):
    """
    Compare full-rank vs low-rank physics computations.
    """
    E_full = field.compute_energy(use_lowrank=False)
    
    results = {'full': E_full}
    for r in ranks:
        E_lr = field.compute_energy(use_lowrank=True, target_rank=r)
        error = abs(E_lr - E_full) / abs(E_full)
        results[f'rank_{r}'] = {'energy': E_lr, 'error': error}
    
    return results
```

**3. Monitor During Training:**
```python
# Track both full and approximate
energy_full = field.compute_energy(use_lowrank=False)
energy_lr = field.compute_energy(use_lowrank=True, target_rank=16)
energy_error = abs(energy_lr - energy_full) / abs(energy_full)

if energy_error > 0.05:  # 5% threshold
    warnings.warn(f"Low-rank approximation error {energy_error:.2%}")
```

## Recommended Action Plan

### Phase 1: Analysis (No Risk)
1. Add field spectrum analysis utilities
2. Measure effective ranks during training
3. Profile computational bottlenecks
4. Identify safe candidates for compression

### Phase 2: Diagnostic Compression (Low Risk)
1. Add `use_lowrank` flag to diagnostic functions
2. Default to False (exact computation)
3. Allow users to enable for dashboards/visualization
4. Validate accuracy on test cases

### Phase 3: Adaptive Compression (Moderate Risk)
1. Implement auto-rank selection
2. Add runtime monitoring of approximation error
3. Fallback to full-rank if error exceeds threshold
4. Document accuracy/speed tradeoffs

### Phase 4: Architecture Integration (High Risk)
1. Only if field analysis shows consistent low rank
2. Extensive validation on multiple tasks
3. Ablation studies comparing full vs compressed
4. Peer review before deployment

## Conclusion

**The D³ complexity is real and addressable.**

**Conservative approach (recommended):**
- Add low-rank option to diagnostics only
- Monitor field spectrum to understand structure
- Document trade-offs for users

**Aggressive approach (requires validation):**
- Compress field in training loop
- Use hierarchical compression
- Full validation suite

**Key insight:** Physics computations are O(D³) but field may have low effective rank. Measure first, compress second, validate always.

---

**Status:** Analysis complete, implementation deferred pending field rank analysis  
**Risk:** Low if limited to diagnostics, high if integrated into training  
**Priority:** Medium - nice optimization but not critical for correctness
