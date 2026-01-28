# Refactored Causal Field Propagation with Context Manager

## Updated Implementation

Based on the expert reviews, here's the refactored code with proper naming and context manager support:

```python
# kernels/causal_propagator.py (renamed from hamiltonian.py)

import torch
import torch.nn.functional as F
from typing import Optional
from .metric_context import MetricContext


def causal_propagator(
    T: torch.Tensor,
    diffusion_coeff: float = 0.1,  # formerly "hbar_cog"
    coupling_strength: float = 1.0,  # formerly "m_cog"
    g_inv_diag: Optional[torch.Tensor] = None,
    V: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Causal field propagator with anisotropic metric.
    
    Computes the causal propagation kernel:
        K[T] = -(D²/2c)∇²_g T + V·T
    
    Where:
        D: Diffusion coefficient (controls propagation speed)
        c: Coupling strength (field interaction strength)
        ∇²_g: Anisotropic Laplace-Beltrami operator
        V: External potential/forcing term
    
    For diagonal spatial metric g_ij = diag(g_xx, g_yy):
        ∇²_g T = g^xx ∂²T/∂x² + g^yy ∂²T/∂y²
    
    This is NOT quantum mechanics - it's a causal diffusion-reaction
    system on a Riemannian manifold with Clifford connection.
    
    Args:
        T: Tensor field (N_x, N_y, D, D) - rank-2 tensor at each point
        diffusion_coeff: Propagation speed parameter (NOT Planck's constant)
        coupling_strength: Interaction strength parameter (NOT mass)
        g_inv_diag: Inverse metric diagonal (n,) where n >= 2
                    First two components used for spatial directions (x, y)
                    If None, uses flat Euclidean metric
        V: Optional potential/forcing term (same shape as T)
    
    Returns:
        K[T]: Propagation kernel (same shape as T)
    
    Mathematical Framework:
        - Geometry: Riemannian manifold with diagonal metric
        - Algebra: Complex octonions (non-associative)
        - Connection: Clifford-Chevalley connection (∇^{(c D^α)})
        - Holomorphic constraint: ∇_μ (Π Γ Φ) = 0
    
    References:
        - docs/Clifford_hodge_Chevally.pdf
        - docs/proof_hierarchy2.pdf
        - models/causal_field.py (associator current)
    """
    # Use context manager for metric validation
    with MetricContext(g_inv_diag, validate=True) as ctx:
        if ctx.is_flat:
            # Flat space: use isotropic Laplacian
            return _causal_propagator_flat(T, diffusion_coeff, coupling_strength, V)
        
        # Anisotropic propagation with metric
        return _causal_propagator_anisotropic(
            T, diffusion_coeff, coupling_strength, ctx, V
        )


def _causal_propagator_flat(
    T: torch.Tensor,
    D: float,
    c: float,
    V: Optional[torch.Tensor]
) -> torch.Tensor:
    """Flat space (Euclidean) propagator."""
    lap_T = spatial_laplacian(T, dx=1.0)
    kinetic = -(D**2 / (2 * c)) * lap_T
    potential = V * T if V is not None else 0.0
    return kinetic + potential


def _causal_propagator_anisotropic(
    T: torch.Tensor,
    D: float,
    c: float,
    ctx: MetricContext,
    V: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Anisotropic propagator respecting Riemannian geometry.
    
    Uses separate directional derivatives weighted by metric components.
    This properly implements the Laplace-Beltrami operator for 
    spatially constant diagonal metrics.
    """
    # Compute directional second derivatives
    d2_dx2 = spatial_laplacian_x(T, dx=1.0)  # ∂²T/∂x²
    d2_dy2 = spatial_laplacian_y(T, dy=1.0)  # ∂²T/∂y²
    
    # Extract spatial metric components (validated by context)
    g_xx, g_yy = ctx.get_spatial_components()
    
    # Anisotropic Laplace-Beltrami: ∇²_g T = g^xx ∂²T/∂x² + g^yy ∂²T/∂y²
    # NOTE: Valid for spatially constant diagonal metrics only
    # For spatially varying metrics, add Christoffel symbol terms
    lap_T_aniso = g_xx * d2_dx2 + g_yy * d2_dy2
    
    # Propagation term (diffusion with coupling)
    propagation = -(D**2 / (2 * c)) * lap_T_aniso
    
    # Potential/forcing term
    potential = V * T if V is not None else 0.0
    
    return propagation + potential


# Keep old name as alias for backward compatibility
def hamiltonian_evolution_with_metric(
    T: torch.Tensor,
    hbar_cog: float,
    m_cog: float,
    g_inv_diag: Optional[torch.Tensor] = None,
    V: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    DEPRECATED: Use causal_propagator() instead.
    
    This is a misnomer - it's not a quantum Hamiltonian.
    Kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "hamiltonian_evolution_with_metric() is deprecated and misleading. "
        "This is NOT quantum mechanics. Use causal_propagator() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return causal_propagator(T, hbar_cog, m_cog, g_inv_diag, V)
```

## Usage Examples

### Basic Usage with Context Manager

```python
from kernels.causal_propagator import causal_propagator
from kernels.metric_context import MetricContext

# Field state
T = torch.randn(28, 28, 16, 16, dtype=torch.complex64)

# Anisotropic metric (x-direction 5x stronger than y)
g_inv_diag = torch.tensor([5.0, 1.0, 1.0, 1.0, ...])

# Context manager automatically validates metric
with MetricContext(g_inv_diag, validate=True, track_perf=True) as ctx:
    K_T = causal_propagator(
        T,
        diffusion_coeff=0.1,
        coupling_strength=1.0,
        g_inv_diag=ctx.g_inv
    )
    
    print(f"Isotropic: {ctx.is_isotropic}")
    print(f"Elapsed: {ctx.elapsed_time:.3f}s")
```

### Integration with Field Evolution

```python
# core/tensor_field.py - Updated evolve_step

def evolve_step(
    self,
    evidence: Optional[torch.Tensor] = None,
    external_input: Optional[torch.Tensor] = None,
    g_inv_diag: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Single timestep of causal field evolution.
    
    Uses causal propagator (NOT quantum Hamiltonian).
    """
    from ..kernels.causal_propagator import causal_propagator
    from ..kernels.metric_context import MetricContext
    
    # Validate and track metric usage
    with MetricContext(g_inv_diag, validate=True) as ctx:
        # Causal propagation term
        K_T = causal_propagator(
            self.T,
            diffusion_coeff=self.config.hbar_cog,  # TODO: Rename config
            coupling_strength=self.config.m_cog,
            g_inv_diag=ctx.g_inv
        )
        
        # Rest of evolution logic...
        # (Bayesian recursive term, memory, etc.)
```

### Batch Processing

```python
from kernels.metric_context import MetricBatchContext

# Multiple fields with different metrics
fields = [field1, field2, field3]
metrics = [metric1, metric2, metric3]

with MetricBatchContext(metrics) as contexts:
    results = []
    for field, ctx in zip(fields, contexts):
        K_T = causal_propagator(
            field.T,
            diffusion_coeff=0.1,
            coupling_strength=1.0,
            g_inv_diag=ctx.g_inv
        )
        results.append(K_T)
```

## Migration Guide

### Step 1: Update Imports (5 minutes)

```python
# Old
from kernels.hamiltonian import hamiltonian_evolution_with_metric

# New
from kernels.causal_propagator import causal_propagator
from kernels.metric_context import MetricContext
```

### Step 2: Update Function Calls (10 minutes)

```python
# Old
H_T = hamiltonian_evolution_with_metric(
    T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=metric
)

# New (with context manager)
with MetricContext(metric) as ctx:
    K_T = causal_propagator(
        T,
        diffusion_coeff=0.1,
        coupling_strength=1.0,
        g_inv_diag=ctx.g_inv
    )
```

### Step 3: Update Config Names (1 hour)

Rename configuration parameters to reflect causal field theory:

```python
# core/config.py
class FieldConfig:
    # OLD names (misleading)
    hbar_cog: float = 0.1  # "cognitive Planck constant" 
    m_cog: float = 1.0     # "effective mass"
    
    # NEW names (accurate)
    diffusion_coeff: float = 0.1  # Diffusion coefficient D
    coupling_strength: float = 1.0  # Coupling strength c
    
    # Or even better: describe what they control
    propagation_speed: float = 0.1
    field_interaction: float = 1.0
```

### Step 4: Update Documentation (2 hours)

Replace all "quantum" and "Hamiltonian" language with proper causal field theory terminology.

## Benefits of Context Manager

1. **Automatic Validation** ✅
   - Checks positive definiteness
   - Catches NaN/Inf early
   - Clear error messages

2. **Performance Tracking** 📊
   - Measure operation time
   - GPU synchronization handled
   - Profile bottlenecks

3. **Resource Management** 🧹
   - Automatic cleanup
   - Exception safety
   - State restoration

4. **Better API** 🎯
   - Clear intent (using context)
   - Impossible to forget validation
   - Type-safe access

5. **Future-Proof** 🔮
   - Easy to add caching
   - Can track metric history
   - Extensible for monitoring

## Testing

```python
# tests/test_context_manager.py

def test_metric_validation():
    """Test that invalid metrics are caught."""
    # Negative component (invalid)
    bad_metric = torch.tensor([1.0, -0.5])
    
    with pytest.raises(ValueError, match="positive definite"):
        with MetricContext(bad_metric):
            pass

def test_performance_tracking():
    """Test performance tracking."""
    metric = torch.tensor([2.0, 1.0])
    
    with MetricContext(metric, track_perf=True) as ctx:
        # Do some work
        time.sleep(0.1)
    
    assert ctx.elapsed_time >= 0.1
    assert ctx.elapsed_time < 0.2

def test_isotropic_detection():
    """Test isotropic metric detection."""
    iso = torch.ones(8) * 2.0
    aniso = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    with MetricContext(iso) as ctx:
        assert ctx.is_isotropic
    
    with MetricContext(aniso) as ctx:
        assert not ctx.is_isotropic
```

## Summary

**What Changed:**
1. ✅ Added `MetricContext` for validation and tracking
2. ✅ Renamed `hamiltonian_evolution_with_metric` → `causal_propagator`
3. ✅ Renamed parameters: `hbar_cog` → `diffusion_coeff`, `m_cog` → `coupling_strength`
4. ✅ Added proper documentation explaining causal field theory (not quantum)
5. ✅ Deprecated old names with warnings
6. ✅ Backward compatible via aliases

**Impact:**
- 🎯 Accurate naming reflects actual physics
- 🔒 Automatic validation prevents errors
- 📊 Performance tracking built-in
- 🧹 Resource management handled
- 📚 Documentation matches implementation

**Recommendation:**
Migrate gradually using deprecation warnings, then remove old names in v2.0.
