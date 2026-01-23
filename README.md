# LIoRHybrid: Physics-Based AI That Explains Itself

A geometric deep learning architecture that uses **quantum-inspired field theory**, **LIoR memory kernel**, and **Riemannian geometry** to build AI systems where physics explains *why* they work, not just benchmarks showing *that* they work.

## Overview

LIoRHybrid implements a complete physics-based AI framework with interpretable geometric attention and non-Markovian memory. The cognitive tensor field $T_{ij}(x,t) \in \mathbb{C}^{D \times D}$ evolves according to:

```
iℏ_cog ∂_t T = H[T] + Λ_QR[T] - Λ_F[T] + J
```

where:
- **H[T]**: Hamiltonian evolution (kinetic + potential)
- **Λ_QR[T]**: Bayesian recursive update (belief revision)
- **Λ_F[T]**: Fractional memory via **LIoR kernel** (power-law with O(1) recurrence)
- **J**: External input (stimulus)

### Core Architecture

**LIoRHybrid** combines five key components:

1. **Cognitive Tensor Field (T_ij)** - Quantum-inspired state representation
2. **LIoR Memory Kernel** - **L**earnable **I**ntegral **o**f **R**esilience with O(1) recurrence for non-Markovian dynamics
3. **Causal Field Encoder** - O(N log N) parallel processing via FFT convolution
4. **Learned Riemannian Geometry** - Metric emerges from field physics
5. **Geodesic Training** - Optimization follows least-action paths through curved space

### Key Features

- **O(1) Memory Recurrence**: LIoR kernel provides power-law memory without sequential cost
- **O(N log N) Processing**: Causal field layers use FFT for parallel convolution
- **Physics-Grounded**: All operators derived from first principles with mathematical guarantees
- **Interpretable**: Geometric products (wedge/tensor/spinor) with clear physical meaning
- **Non-Markovian Memory**: Power-law kernels for long-range temporal correlations
- **Field-Contracted Attention**: Memory-efficient geometric products via field contractions
- **GPU Acceleration**: Full PyTorch implementation with CUDA support

## Installation

### From Source

```bash
git clone https://github.com/JewJitsu11B/Liorhybrid.git
cd Liorhybrid
pip install -e .
```

### Dependencies

Core requirements:
- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.5.0 (for visualization)

Optional (for DPR integration):
- transformers (Hugging Face)

Development tools:
- pytest ≥ 7.0.0 (for testing)
- pytest-cov (for coverage)

Install all dependencies:
```bash
pip install -r requirements.txt
```
```

## Quick Start

### Interactive Training Interface

The easiest way to get started is with the interactive training interface:

```bash
python main.py
```

This launches an interactive menu where you can:
1. **Quick Start** - Train with CausalField + LIoR on sample or custom data
2. **Full Training** - End-to-end training with all components
3. **Resume from Checkpoint** - Continue previous training runs
4. **Generate Sample Dataset** - Create test data for experimentation

### Architecture Options

When starting training, you'll select between:

**Standard Transformer** (O(N²)):
- Traditional self-attention
- Field-contracted geometric products
- Best for shorter sequences

**CausalField with LIoR** (O(N log N)) - **RECOMMENDED**:
- Parallel FFT-based convolution: O(N log N)
- LIoR memory kernel: O(1) recurrence for power-law memory
- Complex octonion state space with geometric operators
- Address-based attention (no dense K/V matmul)
- Best for long sequences with temporal dependencies

### Quick Training Example

```bash
python main.py
# Select: 1 (Quick Start) → 'sample' → 2 (CausalField+LIoR) → 1 (Quick config) → Y
```

### Educational Demo

To understand how geometric operators work:

```bash
python demo_geometric_mamba.py
```

Note: This demo file retains the "mamba" name for historical reasons but demonstrates **geometric operators** (ComplexOctonion, Trinor/Wedge/Spinor) that are used in the CausalField+LIoR architecture.

### Programmatic Usage

For custom training pipelines:

```python
from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.inference import GeometricTransformer
from Liorhybrid.training import CognitiveTrainer

# Configure field
config = FieldConfig(
    spatial_size=(16, 16),
    tensor_dim=16,
    hbar_cog=0.1,
    lambda_QR=0.3,
    lambda_F=0.05,
    alpha=0.5,
    device='cuda'
)

# Create field
field = CognitiveTensorField(config)

# Create model with CausalField + LIoR
model = GeometricTransformer(
    d_model=512,
    n_layers=6,
    n_heads=8,
    field_dim=16,
    use_lior=True  # Enable LIoR memory kernel
)

# Train
trainer = CognitiveTrainer(model, field, trainer_config)
trainer.train(train_loader)
```

## Documentation

For detailed information, see:

- **[QUICK_START.md](QUICK_START.md)** - Comprehensive quick start guide with examples
- **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Project overview, use cases, and engineering spec
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[TRAINING.md](TRAINING.md)** - Training procedures and configuration
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## Examples

See `examples/` directory:

- **demo_geometric_mamba.py**: Educational demo of geometric operators (ComplexOctonion, Trinor/Wedge/Spinor)
- **simple_evolution.py**: Basic field evolution with diagnostics  
- **geometric_inference.py**: Inference with geometric attention
- **potential_evolution.py**: Field evolution with external potentials

Note: `demo_geometric_mamba.py` demonstrates geometric operators used in the CausalField+LIoR architecture.

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Test specific components:
```bash
pytest tests/test_conservation.py    # Norm conservation
pytest test_training.py              # Training pipeline
pytest test_torch_compile.py         # Torch compilation
```

## Architecture

```
Liorhybrid/
├── core/                       # Core field dynamics
│   ├── config.py              # Configuration and parameters
│   └── tensor_field.py        # CognitiveTensorField class
├── models/                     # Physics models
│   ├── biquaternion.py        # Biquaternion algebra
│   ├── causal_field.py        # CausalField with LIoR kernel
│   ├── complex_metric.py      # Riemannian metric learning
│   ├── lior_kernel.py         # LIoR memory kernel (O(1) recurrence)
│   └── manifold.py            # Manifold operations
├── inference/                  # Inference and attention
│   ├── geometric_attention.py # Field-contracted attention
│   ├── geometric_products.py  # Wedge/tensor/spinor products
│   ├── geometric_stack.py     # Full transformer stack with CausalField
│   └── composite_k.py         # Address-based K structure
├── training/                   # Training infrastructure
│   ├── trainer.py             # Main training loop
│   ├── lior_trainer.py        # LIoR-specific training
│   ├── lior_optimizer.py      # Geodesic optimization
│   ├── losses.py              # Loss functions
│   ├── metrics.py             # Comprehensive logging
│   └── datasets.py            # Data loading (text/image/video)
├── kernels/                    # Field evolution operators
│   ├── hamiltonian.py         # H[T] operator
│   ├── bayesian.py            # Λ_QR[T] operator
│   └── fractional_memory.py   # Λ_F[T] operator
├── operators/                  # Additional operators
│   └── collapse.py            # Collapse and measurement
├── utils/                      # Utilities
│   ├── metrics.py             # Diagnostics and conservation laws
│   └── visualization.py       # Plotting utilities
├── tests/                      # Test suite
├── examples/                   # Usage examples
└── main.py                     # Interactive training interface
```

## Mathematical Background

### Master Equation

The field evolves according to:

```
iℏ_cog ∂_t T_ij = [H + Λ_QR - Λ_F + J]_ij
```

### Hamiltonian

```
H[T]_ij = -(ℏ²_cog/2m_cog) ∇²T_ij + V_ij T_ij
```

Implemented via 2D convolution with discrete Laplacian kernel.

### Bayesian Update

```
Λ_QR[T]_ij = λ_QR (B[T(t-Δt)]_ij - T_ij(t-Δt))

B[T]_ij = (w_ij T_ij) / Z
w_ij = exp(-|T_ij - E_ij|²/τ)
```

Drives field toward evidence-weighted posterior.

### LIoR Memory Kernel (Lior Integral of Resilience)

The key innovation that replaces traditional sequential memory:

```
Λ_F[T]_ij = λ_F ∫₀ᵗ K_L(t-τ) T_ij(τ) dτ

K_L(τ; J_H) = Θ(τ) [
    α exp(-β τ)                      # Exponential (Markovian)
  - γ τ^(-δ) exp(-ξ τ)               # Power-law (Fractional)
  + η cos(ω τ + φ) exp(-ζ τ)         # Phasic (Oscillatory)
]
```

**Key insight**: The full path integral memory can be computed via **finite-pole recurrence in O(1) time**:

```
m_t = ρ m_{t-1} + η x_t - ξ x_{t-p_eff}
```

This is "Non-Markovian physics with O(1) Bayesian filter update."

### CausalField Architecture

State-space model with complex octonion structure and FFT-based parallel processing:

```
J^{μν}_{ρσ}(x) = associator(ψ)  # Non-associativity current
T^{μν}_{ρσ}(x) = α J - (1-α) ∫ k(τ) Π·Γ·Φ·J(x') dx'
```

- **Complexity**: O(N log N) via FFT convolution (parallel, not sequential)
- **Memory**: O(1) update via LIoR kernel recurrence
- **Address-based attention**: No dense K/V matmul, only 64 neighbor probing

### Riemannian Geometry

Learned metric tensor g_ij emerges from field:

```
g_ij = ⟨T_ik, T_jk⟩  (field-induced metric)
Geodesic cost = ∫ √(g_ij dx^i dx^j)
```

Training follows geodesics (least-action paths) through learned geometry.

## Key Parameters

| Symbol | Name | Default | Range | Description |
|--------|------|---------|-------|-------------|
| ℏ_cog | Cognitive Planck constant | 0.1 | 0.01-1.0 | Sets quantum-like scale |
| m_cog | Effective mass | 1.0 | 0.1-10.0 | Controls diffusion rate |
| λ_QR | Bayesian update gain | 0.3 | 0.1-0.5 | Belief revision strength |
| λ_F | Memory damping | 0.05 | 0.01-0.1 | Memory effect strength |
| α | Fractional order | 0.5 | 0.3-0.7 | Memory decay rate |
| τ | Bayesian temperature | 0.5 | 0.1-1.0 | Evidence sharpness |
| D | Tensor dimension | 16 | ≥16 | Internal DOF (≥16 for overdetermination) |

### LIoR Kernel Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| α | Exponential weight | 0.5 |
| β | Exponential decay | 1.0 |
| γ | Power-law weight | 0.3 |
| δ | Power-law exponent | 0.5 |
| η | Oscillatory amplitude | 0.2 |
| ω | Oscillatory frequency | 1.0 |
| p_eff | Effective pole count for O(1) recurrence | 4 |

### Model Architecture Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| d_model | Model dimension | 512-1024 |
| n_layers | CausalField layers | 6-12 |
| n_attention_layers | Geometric attention layers | 2-4 |
| n_heads | Attention heads | 8-16 |
| field_dim | Field tensor dimension | 16-32 |

## Features and Status

### Core Components ✓

- **Cognitive Tensor Field**: Full field evolution with all operators
- **LIoR Memory Kernel**: O(1) recurrence for non-Markovian power-law memory
- **CausalField Encoder**: O(N log N) parallel processing via FFT convolution
- **Field-Contracted Attention**: Memory-efficient geometric products
- **Riemannian Metric Learning**: Geometry emerges from field
- **Geodesic Optimization**: Physics-guided training
- **Comprehensive Logging**: All metrics tracked (timing, complexity, field state, etc.)

### Training Infrastructure ✓

- Interactive training interface (main.py)
- Multi-modal data loading (text/image/video)
- Checkpoint management and resume
- Mixed precision training (AMP)
- GPU acceleration with CUDA
- Extensive metrics and diagnostics

### Applications

**Ideal Use Cases**:
- Scientific computing with learned dynamics
- Geometric NLP with interpretable attention
- Multi-modal learning with unified geometry
- Time series with long-range dependencies
- Domains requiring interpretability and uncertainty quantification

**Key Advantages**:
- O(1) memory recurrence (vs O(T) for RNNs, O(T²) for standard transformers)
- O(N log N) processing via parallel FFT (vs O(N²) for attention)
- Physics-based interpretability
- Mathematical guarantees and conservation laws
- Non-Markovian memory without sequential cost
- Emergent geometry from data

## Performance

### Complexity Comparison

| Component | Traditional | LIoRHybrid | Improvement |
|-----------|-------------|------------|-------------|
| **Memory Recurrence** | O(T) sequential (RNN/LSTM) | **O(1)** via LIoR | **T× faster** |
| **Sequence Processing** | O(N²) attention (Transformer) | **O(N log N)** via FFT | **N/log N × faster** |
| **Memory Type** | Exponential decay (Markovian) | **Power-law** (non-Markovian) | True long-range |
| **Attention** | O(N²) dense matmul | **O(N × 64)** address probing | **N/64 × less** |

### Key Innovations

1. **LIoR Kernel**: Replaces sequential memory accumulation with O(1) finite-pole recurrence
2. **Parallel FFT**: CausalField uses FFT convolution instead of sequential processing
3. **Address-Based Attention**: 64 neighbor probing instead of dense K/V matmul
4. **Non-Markovian Physics**: Power-law memory without quadratic cost

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Testing requirements
- Documentation expectations
- Pull request process

## Theory and Papers

For complete mathematical derivations:
- **EXECUTIVE_SUMMARY.md**: Engineering specification and use cases
- **PHYSICS_AUDIT_FINAL.md**: Physics validation
- **IMPLEMENTATION_SUMMARY.md**: Technical details

## Citation

```bibtex
@software{leizerman2026liorhybrid,
  title={LIoRHybrid: Physics-Based AI with Geometric Deep Learning},
  author={Leizerman, Sam},
  year={2026},
  url={https://github.com/JewJitsu11B/Liorhybrid}
}
```

## License

[To be determined]

## Contact

For questions, issues, or contributions:
- Open an issue: https://github.com/JewJitsu11B/Liorhybrid/issues
- Repository: https://github.com/JewJitsu11B/Liorhybrid
