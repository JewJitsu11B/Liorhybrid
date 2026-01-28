# Metric Scaling: Isotropic vs Anisotropic

## Current Implementation: ISOTROPIC

The current metric-aware Hamiltonian implementation uses **isotropic (uniform) scaling**.

### Location
`kernels/hamiltonian.py`, function `hamiltonian_evolution_with_metric()`, lines 141-151

### Code
```python
# Weight by inverse metric
# For diagonal metric: scale Laplacian by average metric component
if g_inv_diag.dim() == 1:
    # g_inv_diag is (D,) - take mean as isotropic scaling
    metric_scale = g_inv_diag.mean().item()
else:
    # Shouldn't happen, but handle gracefully
    metric_scale = 1.0

# Metric-weighted Laplacian
lap_T_metric = metric_scale * lap_T
```

## What is Isotropic Scaling?

**Isotropic** means "the same in all directions" (from Greek: *isos* = equal, *tropos* = direction).

In our implementation:
1. We take all diagonal components of the inverse metric: `g^{11}, g^{22}, ..., g^{DD}`
2. We compute their **average**: `λ_avg = mean(g^{ii})`
3. We apply this **single scalar** uniformly: `∇²_g T ≈ λ_avg * ∇²T`

### Example
If the metric components are:
- `g^{11} = 2.0`
- `g^{22} = 1.5`  
- `g^{33} = 1.0`

Isotropic scaling computes: `λ_avg = (2.0 + 1.5 + 1.0) / 3 = 1.5`

Then applies this uniformly to all directions.

## What is Anisotropic Scaling?

**Anisotropic** means "different in different directions" (from Greek: *an-* = not, *isos* = equal, *tropos* = direction).

In a proper anisotropic implementation:
1. Each direction would use its **own** metric component
2. The Laplacian would be: `∇²_g T = Σ_i g^{ii} ∂²T/∂x_i²`
3. Different directions would be weighted differently

### Example
With the same metric:
- x-direction uses: `g^{11} = 2.0` → `∂²T/∂x²` gets weighted by 2.0
- y-direction uses: `g^{22} = 1.5` → `∂²T/∂y²` gets weighted by 1.5
- z-direction uses: `g^{33} = 1.0` → `∂²T/∂z²` gets weighted by 1.0

Each direction respects its own geometry.

## Trade-offs

### Isotropic Scaling (Current)

**Advantages:**
- ✅ Simple to implement (single scalar multiplication)
- ✅ Computationally efficient
- ✅ Backward compatible with flat space
- ✅ Captures overall geometric scale
- ✅ Works even with dimension mismatches

**Disadvantages:**
- ❌ Loses directional information
- ❌ Cannot represent stretched/compressed geometries along specific axes
- ❌ Less physically accurate for anisotropic spaces
- ❌ Averaging may not be appropriate for all metrics

### Anisotropic Scaling (Not Implemented)

**Advantages:**
- ✅ Physically accurate for directional geometries
- ✅ Preserves anisotropic structure
- ✅ Proper Laplace-Beltrami operator
- ✅ Can represent stretched/compressed spaces

**Disadvantages:**
- ❌ More complex to implement
- ❌ Requires careful handling of spatial vs internal dimensions
- ❌ May require restructuring the Laplacian computation
- ❌ Harder to handle dimension mismatches

## Why Isotropic Was Chosen

1. **Simplicity**: First implementation to establish the infrastructure
2. **Backward compatibility**: Easy fallback to flat space (metric_scale = 1.0)
3. **Dimension mismatch**: Coordinate manifold (n=8) vs tensor field (D=16) dimensions don't match
4. **Good approximation**: For nearly isotropic metrics, the error is small

## When Would Anisotropic Be Needed?

Consider switching to anisotropic if:

1. **Strongly directional geometries**: E.g., metric has `g^{11} = 10.0, g^{22} = 0.1` (100x difference)
2. **Physical requirements**: Application requires exact geometric fidelity
3. **Learned metrics are anisotropic**: Training produces strongly directional metrics
4. **Performance is not critical**: Can afford more complex computation

## Implementation Roadmap for Anisotropic

To implement proper anisotropic scaling:

### Step 1: Clarify Dimensions
- Spatial dimensions: (x, y) for the field grid
- Internal dimensions: (D, D) for the tensor at each point
- Metric dimensions: Currently (n,) where n = coord_dim_n

**Question**: Should the metric apply to:
- **Option A**: Spatial directions (x, y)? → Need 2D metric
- **Option B**: Internal tensor space (D, D)? → Need DxD metric
- **Option C**: Both? → Need separate spatial and internal metrics

### Step 2: Modify Laplacian Computation
Current: `lap_T = spatial_laplacian(T, dx=1.0)` computes `∂²T/∂x² + ∂²T/∂y²`

For anisotropic:
```python
# Separate x and y derivatives
lap_x = second_derivative_x(T, dx=1.0)  # ∂²T/∂x²
lap_y = second_derivative_y(T, dx=1.0)  # ∂²T/∂y²

# Weight by metric components
g_xx = g_inv_diag[0]  # Metric for x-direction
g_yy = g_inv_diag[1]  # Metric for y-direction

# Anisotropic Laplacian
lap_T_aniso = g_xx * lap_x + g_yy * lap_y
```

### Step 3: Handle Dimension Mismatches
If metric has dimension n but field has dimension D:
- Pad/project metric to match dimensions?
- Use different metrics for different purposes?
- Separate spatial and internal metrics?

### Step 4: Testing
- Verify anisotropic evolution produces different results than isotropic
- Test with strongly anisotropic metrics
- Validate physics correctness

## Current Status Summary

| Aspect | Status |
|--------|--------|
| **Implementation** | Isotropic (averaging) |
| **Physics** | Approximate, not exact |
| **Performance** | Efficient (scalar multiplication) |
| **Accuracy** | Good for nearly isotropic metrics |
| **Limitation** | Cannot represent anisotropic geometries |
| **Next Step** | Implement anisotropic if needed |

## References

- **Code**: `kernels/hamiltonian.py`, lines 98-159
- **Tests**: `tests/test_metric_aware_hamiltonian.py`
- **Discussion**: PR review comments noted the isotropic approximation

---

**Last Updated**: 2026-01-28  
**Status**: Isotropic implementation is working and tested  
**Decision**: Anisotropic to be implemented if application requires it
