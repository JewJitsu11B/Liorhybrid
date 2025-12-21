# Bayesian Cognitive Field

A PyTorch implementation of rank-2 tensor field evolution under Bayesian recursive dynamics with fractional memory.

## Overview

This package implements the mathematical framework described in the companion paper `bayesian_recursive_operator.tex`. The cognitive tensor field $T_{ij}(x,t) \in \mathbb{C}^{D \times D}$ evolves according to:

```
iℏ_cog ∂_t T = H[T] + Λ_QR[T] - Λ_F[T] + J
```

where:
- **H[T]**: Hamiltonian evolution (kinetic + potential)
- **Λ_QR[T]**: Bayesian recursive update (belief revision)
- **Λ_F[T]**: Fractional memory (power-law damping)
- **J**: External input (stimulus)

### Key Features

- **Zero free parameters**: All operators derived from first principles
- **Self-tokenization**: Categories emerge from correlation structure
- **Reversible collapse**: Decisions are one-way in time but informationally reversible
- **Fractional memory**: Long-range temporal effects via power-law kernels
- **GPU acceleration**: Full PyTorch implementation with CUDA support

## Installation

### From source

```bash
git clone <repository-url>
cd bayesian_cognitive_field
pip install -e .
```

### Dependencies

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy
- SciPy
- Matplotlib (for visualization)
- pytest (for testing)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Evolution

```python
from bayesian_cognitive_field import CognitiveTensorField, FAST_TEST_CONFIG

# Create field with default configuration
field = CognitiveTensorField(FAST_TEST_CONFIG)

# Run evolution
for step in range(100):
    field.evolve_step()

    if step % 20 == 0:
        print(f"Step {step}: ||T||² = {field.get_norm_squared():.6f}")
```

### With External Evidence

```python
import torch

# Create evidence tensor (same shape as field)
evidence = torch.randn(8, 8, 8, 8, dtype=torch.complex64)

# Evolve with Bayesian update toward evidence
for _ in range(100):
    field.evolve_step(evidence=evidence)
```

### Custom Configuration

```python
from bayesian_cognitive_field import FieldConfig, CognitiveTensorField

config = FieldConfig(
    spatial_size=(16, 16),      # Grid size
    tensor_dim=16,               # D×D tensor at each point
    hbar_cog=0.1,                # Cognitive Planck constant
    lambda_QR=0.3,               # Bayesian update strength
    lambda_F=0.05,               # Memory damping strength
    alpha=0.5,                   # Fractional order
    tau=0.5,                     # Bayesian temperature
    dt=0.005,                    # Timestep
    device='cuda'                # Use GPU
)

field = CognitiveTensorField(config)
```

## Examples

See `examples/` directory:

- **simple_evolution.py**: Basic field evolution with diagnostics
- **mnist_clustering.py**: Self-tokenization on MNIST (stub, in progress)

Run examples:
```bash
python examples/simple_evolution.py
```

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Test specific components:
```bash
pytest tests/test_conservation.py  # Norm conservation
pytest tests/test_bayesian.py      # Bayesian updates
pytest tests/test_memory.py        # Fractional memory
pytest tests/test_integration.py   # Full evolution
```

## Architecture

```
bayesian_cognitive_field/
├── core/
│   ├── config.py           # Configuration and parameters
│   └── tensor_field.py     # Main CognitiveTensorField class
├── kernels/
│   ├── hamiltonian.py      # H[T] operator
│   ├── bayesian.py         # Λ_QR[T] operator
│   └── fractional_memory.py # Λ_F[T] operator
├── operators/
│   └── collapse.py         # Collapse and measurement
├── utils/
│   ├── metrics.py          # Diagnostics and conservation laws
│   └── visualization.py    # Plotting utilities
├── tests/                  # Test suite
└── examples/               # Usage examples
```

## Mathematical Background

### Master Equation

The field evolves according to (Paper Equation 1):

```
iℏ_cog ∂_t T_ij = [H + Λ_QR - Λ_F + J]_ij
```

### Hamiltonian (Paper Eq. 2)

```
H[T]_ij = -(ℏ²_cog/2m_cog) ∇²T_ij + V_ij T_ij
```

Implemented via 2D convolution with discrete Laplacian kernel.

### Bayesian Update (Paper Eq. 4-6)

```
Λ_QR[T]_ij = λ_QR (B[T(t-Δt)]_ij - T_ij(t-Δt))

B[T]_ij = (w_ij T_ij) / Z
w_ij = exp(-|T_ij - E_ij|²/τ)
```

Drives field toward evidence-weighted posterior.

### Fractional Memory (Paper Eq. 7-8)

```
Λ_F[T]_ij = λ_F ∫₀ᵗ K(t-τ) T_ij(τ) dτ
K(τ) = τ^(α-1) / Γ(α)
```

Power-law kernel creates long-range temporal correlations.

## Key Parameters

| Symbol | Name | Default | Range | Description |
|--------|------|---------|-------|-------------|
| ℏ_cog | Cognitive Planck constant | 0.1 | 0.01-1.0 | Sets quantum-like scale |
| m_cog | Effective mass | 1.0 | 0.1-10.0 | Controls diffusion rate |
| λ_QR | Bayesian update gain | 0.3 | 0.1-0.5 | Belief revision strength |
| λ_F | Memory damping | 0.05 | 0.01-0.1 | Memory effect strength |
| α | Fractional order | 0.5 | 0.3-0.7 | Memory decay rate |
| τ | Bayesian temperature | 0.5 | 0.1-1.0 | Evidence sharpness |
| D | Tensor dimension | 16 | ≥16 | Internal DOF (must be ≥16 for overdetermination) |

## Current Status

### Implemented ✓
- Core field evolution (Algorithm 1)
- All three kernel operators (H, Λ_QR, Λ_F)
- Configuration system with validation
- Basic metrics (norm, local correlation)
- Test suite with conservation tests
- GPU support via PyTorch

### In Progress ⚠
- Token-based clustering (replacing naive outer product)
- Collapse operators (soft projection)
- Visualization utilities
- MNIST self-tokenization example

### Planned 📋
- Metric tensor and Christoffel symbols for semantic addressing
- BCH error correction for route hashing
- Neighbor heap structures (32 NN, 16 min/max)
- Full self-tokenization pipeline
- Active inference examples

## Theory Papers

For complete mathematical derivations, see:
- **bayesian_recursive_operator.tex**: Full formalism and implementation notes
- **cdgt_three_key_derivations.tex**: High-impact predictions (cosmological constant, Hubble tension, strong CP)
- **cdgt_parameter_derivations.tex**: 45-parameter validation set

## Citation

```bibtex
@article{leizerman2025bayesian,
  title={Bayesian Recursive Operator for Cognitive Field Dynamics},
  author={Leizerman, Sam},
  year={2025}
}
```

## License

[To be determined]

## Contact

For questions, issues, or contributions, please [open an issue](repository-issues-url).
