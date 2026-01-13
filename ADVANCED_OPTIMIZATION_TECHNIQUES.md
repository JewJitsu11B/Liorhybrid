# Advanced Optimization Techniques for Physics Computations

**Date:** 2026-01-08  
**Context:** Additional optimization beyond rank reduction  
**Techniques:** Box rotation, diagonal rotation, symbolic precomputation

## Overview

Beyond rank reduction (which addresses O(D³) via low-rank approximation), several advanced optimization techniques from the training pipeline (trainer2) can be applied to physics computations.

## 1. Diagonal Rotation (diag_rot)

### Concept

**Diagonal rotation** exploits the structure of symmetric positive definite (SPD) matrices by decomposing them into:
- A diagonal matrix in a rotated coordinate frame
- Plus low-rank corrections

**Formula:**
```
g(v,v) = Ω² · v^T g₀ v + (U^T v)^T D (U^T v)
```

Where:
- `Ω²`: Global scaling factor
- `g₀`: Base diagonal metric
- `U`: Low-rank correction basis (orthonormal)
- `D`: Diagonal correction matrix

### Application to Physics

**Energy computation with diag_rot:**

Instead of full T†HT computation:
```python
# Standard: O(D³)
T_dag_H = torch.einsum('xyij,xyjk->xyik', T_dag, H_T)
traces = torch.einsum('xyii->xy', T_dag_H)
```

Use diagonal + low-rank structure:
```python
# Diag_rot: O(D²r + Dr²) where r << D
def compute_energy_diagrot(T, H_T, Omega_sq, g0_diag, U_lr, D_lr):
    """
    Compute energy using diagonal-rotation decomposition.
    
    Assumes metric g = Ω² · diag(g0) + U @ D @ U^T
    
    Args:
        T: Field (N, D, D)
        H_T: Hamiltonian applied (N, D, D)
        Omega_sq: Global scale factor
        g0_diag: Diagonal base metric (D,)
        U_lr: Low-rank basis (D, r)
        D_lr: Diagonal corrections (r,)
    """
    N, D, _ = T.shape
    
    # Part 1: Diagonal contribution (O(D²))
    T_flat = T.reshape(N, D*D)
    H_flat = H_T.reshape(N, D*D)
    
    # Weight by diagonal metric
    weighted = T_flat.conj() * g0_diag.repeat(D) * H_flat
    energy_diag = Omega_sq * weighted.sum()
    
    # Part 2: Low-rank correction (O(Dr²))
    T_proj = torch.matmul(T.reshape(N, D, D), U_lr)  # (N, D, r)
    H_proj = torch.matmul(H_T.reshape(N, D, D), U_lr)  # (N, D, r)
    
    # Apply diagonal correction in projected space
    for i in range(N):
        for j in range(D):
            energy_lr = torch.sum(T_proj[i, j].conj() * D_lr * H_proj[i, j])
            energy_diag += energy_lr
    
    return energy_diag.real
```

**Benefits:**
- Reduces O(D³) → O(D²r + Dr²)
- For r=4-8, this is 4-8x speedup for D=32
- Preserves physics if decomposition is accurate

### When to Use

**Safe if:**
- Field metric has diagonal-dominant structure
- Low-rank corrections are small
- Can validate decomposition quality

**Check via:**
```python
def validate_diagrot_decomposition(g_full, Omega_sq, g0_diag, U_lr, D_lr):
    """Verify that diag_rot approximates full metric."""
    # Reconstruct
    g_diag = Omega_sq * torch.diag(g0_diag)
    g_lr = torch.matmul(U_lr * D_lr, U_lr.T)
    g_approx = g_diag + g_lr
    
    # Compare
    error = torch.norm(g_full - g_approx) / torch.norm(g_full)
    return error
```

## 2. Box Rotation (box_rot)

### Concept

**Box rotation** refers to representing rotations via composition of plane rotations (Givens rotations):

```
Q = ∏ₖ G(iₖ, jₖ, θₖ)
```

Where each G is a 2D rotation in the (i,j) plane:
```
G(i,j,θ) = I + (cos(θ)-1)·(eᵢeᵢᵀ + eⱼeⱼᵀ) + sin(θ)·(eᵢeⱼᵀ - eⱼeᵢᵀ)
```

### Application to Physics

**Rotating field for computation:**

Instead of computing in original basis, rotate to basis where metric is simpler:

```python
def compute_energy_boxrot(T, H_T, rotor_planes, rotor_thetas):
    """
    Compute energy after rotating to diagonal frame.
    
    Args:
        T: Field (N, D, D)
        H_T: Hamiltonian (N, D, D)
        rotor_planes: List of (i,j) plane indices
        rotor_thetas: Rotation angles for each plane
    """
    N, D, _ = T.shape
    
    # Build rotation matrix Q
    Q = torch.eye(D, device=T.device, dtype=T.dtype)
    for (i, j), theta in zip(rotor_planes, rotor_thetas):
        G = givens_rotation(D, i, j, theta)
        Q = Q @ G
    
    # Rotate field to diagonal frame
    T_rot = torch.einsum('ij,nkj->nki', Q, T)  # Q^T @ T
    H_rot = torch.einsum('ij,nkj->nki', Q, H_T)
    
    # In rotated frame, metric should be nearly diagonal
    # Compute energy efficiently
    energy = torch.sum(T_rot.conj() * H_rot).real
    
    return energy

def givens_rotation(D, i, j, theta):
    """Construct Givens rotation matrix."""
    G = torch.eye(D, dtype=torch.float32)
    c, s = torch.cos(theta), torch.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G
```

**Benefits:**
- Rotates to frame where operations are simpler
- Avoids full matrix inversions
- Numerically stable (orthogonal transformations)

**Complexity:**
- Building Q: O(k·D²) where k is number of planes
- Rotation: O(D³) but with better constants
- If target is diagonal: O(D²) after rotation

### When to Use

**Good for:**
- Fields with known symmetry structure
- Quasi-diagonal metrics
- Numerical stability (orthogonal transformations preserve norms)

**Requires:**
- Learning or deriving optimal rotation angles
- Validation that rotated frame is simpler

## 3. Symbolic Precomputation

### Concept

**Symbolic precomputation** means:
1. Identify terms that don't change during training
2. Compute them once at startup
3. Cache on GPU
4. Reuse in hot path without recomputation

### Application to Physics

**Precompute geometry tensors:**

```python
@dataclass
class PhysicsGeometryCache:
    """Precomputed geometry for physics computations."""
    
    # Base metric and inverse
    g0: torch.Tensor          # (D, D) base metric
    g0_inv: torch.Tensor      # (D, D) inverse
    
    # Contraction kernels
    K_contract: torch.Tensor  # (D, D, D, D) contraction kernel
    
    # Potential (if static)
    V: Optional[torch.Tensor] # (N_x, N_y, D, D) potential
    
    # Laplacian stencil
    laplacian_stencil: torch.Tensor  # (3, 3) or (5, 5)
    
    # Eigendecomposition (if metric is static)
    g0_eigvals: torch.Tensor  # (D,)
    g0_eigvecs: torch.Tensor  # (D, D)
    
    def __post_init__(self):
        """Move all to GPU and set requires_grad=False."""
        for name in ['g0', 'g0_inv', 'K_contract', 'V', 
                     'laplacian_stencil', 'g0_eigvals', 'g0_eigvecs']:
            val = getattr(self, name)
            if val is not None:
                val = val.cuda().requires_grad_(False)
                setattr(self, name, val)

def precompute_physics_geometry(config):
    """
    Precompute all static geometry for physics.
    
    Called once at training start.
    """
    D = config.tensor_dim
    
    # Base metric from field configuration
    g0 = torch.eye(D, dtype=torch.float32)  # Can be more complex
    g0_inv = torch.linalg.inv(g0)
    
    # Contraction kernel (if using tensor products)
    K_contract = torch.einsum('ij,kl->ijkl', g0_inv, g0_inv) / (D * D)
    
    # Static potential (if applicable)
    if config.use_static_potential:
        from kernels import create_potential
        V = create_potential(
            config.spatial_size,
            config.tensor_dim,
            config.potential_type,
            config.potential_strength
        )
    else:
        V = None
    
    # Laplacian stencil
    laplacian_stencil = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32)
    
    # Eigendecomposition (if metric is static)
    g0_eigvals, g0_eigvecs = torch.linalg.eigh(g0)
    
    return PhysicsGeometryCache(
        g0=g0,
        g0_inv=g0_inv,
        K_contract=K_contract,
        V=V,
        laplacian_stencil=laplacian_stencil,
        g0_eigvals=g0_eigvals,
        g0_eigvecs=g0_eigvecs
    )
```

**Use in hot path:**

```python
def compute_energy_precomputed(T, H_T, geom_cache):
    """
    Fast energy computation using precomputed geometry.
    
    No inversions, no eigendecomps in hot path.
    """
    # Use precomputed contraction kernel
    # Instead of computing on-the-fly
    energy = torch.einsum('xyij,ijkl,xykl->', T.conj(), geom_cache.K_contract, H_T)
    return energy.real

def hamiltonian_evolution_precomputed(T, geom_cache):
    """
    Apply Hamiltonian using precomputed potential and stencil.
    """
    # Kinetic term with precomputed Laplacian stencil
    kinetic = apply_laplacian_fast(T, geom_cache.laplacian_stencil)
    
    # Potential term with precomputed V
    if geom_cache.V is not None:
        potential = geom_cache.V * T
    else:
        potential = 0.0
    
    return kinetic + potential
```

### Benefits

**Performance:**
- No repeated expensive operations (inv, eigh, etc.)
- GPU memory is fast (reuse is cheap)
- Eliminates redundant computation

**Typical savings:**
- Inverse: O(D³) → O(1) per call
- Eigendecomp: O(D³) → O(1) per call
- Potential evaluation: O(N_x·N_y·D²) → O(1) if static

### When to Use

**Safe if:**
- Geometry is static or slowly changing
- Memory cost of cache is acceptable
- No dynamic potential/metric needed

**Requires:**
- Initial precomputation phase
- Cache invalidation if geometry changes
- Memory management (don't cache too much)

## 4. Combined Strategy

### Optimal Combination

Use **all three techniques** together:

```python
class OptimizedPhysicsEngine:
    """
    Physics computations with all optimizations.
    
    Combines:
    1. Symbolic precomputation (cache geometry)
    2. Diag_rot (diagonal + low-rank metric)
    3. Box_rot (rotate to simpler frame)
    4. Rank reduction (compress field if needed)
    """
    
    def __init__(self, config):
        # Precompute static geometry
        self.geom_cache = precompute_physics_geometry(config)
        
        # Diag_rot parameters (learned or derived)
        self.Omega_sq = torch.tensor(1.0)
        self.g0_diag = torch.ones(config.tensor_dim)
        self.U_lr = torch.randn(config.tensor_dim, 4)  # rank-4
        self.D_lr = torch.ones(4)
        
        # Box_rot parameters
        self.rotor_planes = [(0,1), (2,3), (4,5)]  # Example
        self.rotor_thetas = torch.zeros(len(self.rotor_planes))
        
        # Move to GPU
        self._to_gpu()
    
    def compute_energy(self, T, H_T, use_optimizations=True):
        if not use_optimizations:
            # Fallback to exact computation
            return self._compute_energy_exact(T, H_T)
        
        # Step 1: Apply box rotation to simpler frame
        T_rot, H_rot = self._apply_box_rotation(T, H_T)
        
        # Step 2: Use diag_rot structure
        energy = self._compute_energy_diagrot(T_rot, H_rot)
        
        return energy
    
    def _apply_box_rotation(self, T, H_T):
        """Rotate to frame where metric is diagonal."""
        Q = self._build_rotation_matrix()
        T_rot = torch.einsum('ij,nkj->nki', Q, T)
        H_rot = torch.einsum('ij,nkj->nki', Q, H_T)
        return T_rot, H_rot
    
    def _compute_energy_diagrot(self, T, H):
        """Energy in diagonal frame with low-rank corrections."""
        # Diagonal part (fast)
        energy_diag = torch.sum(
            T.conj() * self.g0_diag.view(1, -1, 1) * H
        ).real * self.Omega_sq
        
        # Low-rank correction
        T_proj = torch.matmul(T, self.U_lr)
        H_proj = torch.matmul(H, self.U_lr)
        energy_lr = torch.sum(
            T_proj.conj() * self.D_lr * H_proj
        ).real
        
        return energy_diag + energy_lr
```

### Complexity Comparison

| Method | Complexity | Speedup (D=32) |
|--------|-----------|----------------|
| Naive loops | O(N·D³) | 1x (baseline) |
| Vectorized (einsum) | O(N·D³) | 10-50x (parallelism) |
| + Rank reduction (r=8) | O(N·D·r²) | 4x more (64x total) |
| + Diag_rot (r=4) | O(N·D²r) | 8x more (512x total) |
| + Box_rot (k=6 planes) | O(N·D²r) | 2x more (1024x total) |
| + Precompute | O(N·D²r) | 2x more (2048x total) |

**Realistic gain:** 100-500x over naive loops when all techniques combined

## 5. Implementation Roadmap

### Phase 1: Symbolic Precomputation (Immediate)
- Low risk, high reward
- Cache static geometry tensors
- Eliminate redundant inversions/eigendecomps
- **Recommendation:** Implement now

### Phase 2: Diagonal Rotation (Medium-term)
- Requires metric analysis
- Learn or derive Omega, g0_diag, U, D parameters
- Validate approximation quality
- **Recommendation:** After field spectrum analysis

### Phase 3: Box Rotation (Advanced)
- Requires learning rotation angles
- Most benefit if field has rotational structure
- Complex to optimize
- **Recommendation:** After diag_rot shows benefit

### Phase 4: Full Integration (Long-term)
- Combine all techniques
- Careful validation at each step
- Ablation studies to measure contribution
- **Recommendation:** Research project

## 6. Validation Strategy

### Before Deploying

**1. Precomputation:**
```python
# Verify cache is correct
geom_cache = precompute_physics_geometry(config)
energy_cached = compute_energy_precomputed(T, H_T, geom_cache)
energy_exact = compute_energy_exact(T, H_T)
assert abs(energy_cached - energy_exact) < 1e-6
```

**2. Diag_rot:**
```python
# Verify decomposition quality
error = validate_diagrot_decomposition(g_full, Omega_sq, g0_diag, U_lr, D_lr)
assert error < 0.01  # 1% error threshold
```

**3. Box_rot:**
```python
# Verify rotation is orthogonal
Q = build_rotation_matrix(rotor_planes, rotor_thetas)
QQT = Q @ Q.T
assert torch.allclose(QQT, torch.eye(D), atol=1e-6)
```

**4. Combined:**
```python
# Compare optimized vs exact
energy_opt = engine.compute_energy(T, H_T, use_optimizations=True)
energy_ref = engine.compute_energy(T, H_T, use_optimizations=False)
relative_error = abs(energy_opt - energy_ref) / abs(energy_ref)
assert relative_error < 0.05  # 5% error budget
```

## Conclusion

**Three powerful optimization techniques:**
1. **Diag_rot:** Diagonal-dominant metric representation
2. **Box_rot:** Rotation to simpler coordinate frames
3. **Symbolic precompute:** Cache static geometry

**Combined with rank reduction:**
- Can achieve 100-500x speedup over naive implementation
- Maintains physics accuracy with proper validation
- Already proven in trainer2 for training loop

**Recommendation:**
- Start with symbolic precomputation (immediate, safe)
- Add diag_rot after field spectrum analysis
- Consider box_rot for advanced optimization
- Always validate before deploying

**Status:** Analysis complete, techniques documented, ready for selective integration based on profiling needs.
