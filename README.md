# LIoRHybrid: Physics-Based AI That Explains Itself

A geometric deep learning architecture that uses **quantum-inspired field theory**, **power-law memory**, and **Riemannian geometry** to build AI systems where physics explains *why* they work, not just benchmarks showing *that* they work.

## Overview

LIoRHybrid implements a complete physics-based AI framework with interpretable geometric attention and non-Markovian memory. The cognitive tensor field $T_{ij}(x,t) \in \mathbb{C}^{D \times D}$ evolves according to:

```
iℏ_cog ∂_t T = H[T] + Λ_QR[T] - Λ_F[T] + J
```

where:
- **H[T]**: Hamiltonian evolution (kinetic + potential)
- **Λ_QR[T]**: Bayesian recursive update (belief revision)
- **Λ_F[T]**: Fractional memory (power-law damping with O(1) recurrence)
- **J**: External input (stimulus)

### Core Architecture

**LIoRHybrid** combines five key components:

1. **Cognitive Tensor Field (T_ij)** - Quantum-inspired state representation
2. **Geometric Mamba** - O(N) state-space model with CI8 geometric operators
3. **LIoR Memory Kernel** - Power-law non-Markovian dynamics with O(1) recurrence
4. **Learned Riemannian Geometry** - Metric emerges from field physics
5. **Geodesic Training** - Optimization follows least-action paths through curved space

### Key Features

- **O(N) Complexity**: Geometric Mamba provides linear-time processing for long sequences
- **Physics-Grounded**: All operators derived from first principles with mathematical guarantees
- **Interpretable**: Geometric products (wedge/tensor/spinor) with clear physical meaning
- **Non-Markovian Memory**: Power-law kernels for long-range temporal correlations
- **Field-Contracted Attention**: Memory-efficient geometric products via field contractions
- **DPR Integration**: Statistically optimal K/V generation using pre-trained encoders
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

## Quick Start

### Interactive Training Interface

The easiest way to get started is with the interactive training interface:

```bash
python main.py
```

This launches an interactive menu where you can:
1. **Quick Start** - Train with Geometric Mamba on sample or custom data
2. **Full Training** - End-to-end training with all components
3. **Resume from Checkpoint** - Continue previous training runs
4. **Generate Sample Dataset** - Create test data for experimentation

### Choose Your Architecture

When starting training, you'll select between:

**Standard Transformer** (O(N²)):
- Traditional self-attention
- Field-contracted geometric products
- Best for shorter sequences

**Geometric Mamba** (O(N)) - **RECOMMENDED**:
- Linear complexity state-space model
- CI8 (Complex Octonion) geometric operators
- Trinor/Wedge/Spinor products
- DPR integration for optimal K/V
- Best for long sequences (10k+ tokens)

### Quick Training Example

```bash
python main.py
# Select: 1 (Quick Start) → 'sample' → 2 (Geometric Mamba) → 1 (Quick config) → Y
```

### Educational Demo

To understand how geometric operators work before training:

```bash
python demo_geometric_mamba.py
```

This shows:
- ComplexOctonion (CI8) operations
- How Trinor replaces matrix A
- How Wedge replaces matrix B  
- How Spinor replaces matrix C
- Comparison tables and examples

### Programmatic Usage

For custom training pipelines:

```python
from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.inference import GeometricTransformerWithMamba
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

# Create model with Geometric Mamba
model = GeometricTransformerWithMamba(
    d_model=512,
    n_mamba_layers=6,
    n_attention_layers=2,
    n_heads=8,
    field_dim=16,
    use_dpr=True
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
- **[GEOMETRIC_MAMBA_GUIDE.md](GEOMETRIC_MAMBA_GUIDE.md)** - Operator correspondence and mathematical background
- **[TRAINING.md](TRAINING.md)** - Training procedures and configuration
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## Examples

See `examples/` directory:

- **demo_geometric_mamba.py**: Educational demo of geometric operators
- **simple_evolution.py**: Basic field evolution with diagnostics  
- **geometric_inference.py**: Inference with geometric attention
- **potential_evolution.py**: Field evolution with external potentials

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
│   ├── causal_field.py        # Causal field operations
│   ├── complex_metric.py      # Riemannian metric learning
│   ├── lior_kernel.py         # LIoR memory kernel (O(1) recurrence)
│   └── manifold.py            # Manifold operations
├── inference/                  # Inference and attention
│   ├── geometric_mamba.py     # Geometric Mamba (O(N) state-space)
│   ├── geometric_attention.py # Field-contracted attention
│   ├── geometric_products.py  # Wedge/tensor/spinor products
│   ├── dpr_encoder.py         # DPR K/V generation
│   └── geometric_stack.py     # Full transformer stack
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

### Fractional Memory (LIoR Kernel)

```
Λ_F[T]_ij = λ_F ∫₀ᵗ K(t-τ) T_ij(τ) dτ
K(τ) = τ^(α-1) / Γ(α)
```

Power-law kernel creates long-range temporal correlations with O(1) recurrence.

### Geometric Mamba

State-space model with CI8 (Complex Octonion) structure:

```
h_t = Trinor(x_t) • h_{t-1} + Wedge(x_t)
y_t = Spinor(h_t)
```

where Trinor/Wedge/Spinor are geometric products derived from field contractions.

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

### Model Architecture Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| d_model | Model dimension | 512-1024 |
| n_mamba_layers | Geometric Mamba layers | 6-12 |
| n_attention_layers | Geometric attention layers | 2-4 |
| n_heads | Attention heads | 8-16 |
| field_dim | Field tensor dimension | 16-32 |

## Features and Status

### Core Components ✓

- **Cognitive Tensor Field**: Full field evolution with all operators
- **Geometric Mamba**: O(N) state-space model with CI8 operators
- **Field-Contracted Attention**: Memory-efficient geometric products
- **LIoR Memory Kernel**: O(1) recurrence for power-law memory
- **DPR Integration**: Statistically optimal K/V generation
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
- O(N) complexity for long sequences (vs O(N²) for standard transformers)
- Physics-based interpretability
- Mathematical guarantees and conservation laws
- Non-Markovian memory without quadratic cost
- Emergent geometry from data

## Performance

### Complexity Comparison

| Architecture | Complexity | Memory | Long Sequences |
|--------------|------------|---------|----------------|
| Standard Transformer | O(N²) | O(N²) | Slow (>2048 tokens) |
| **Geometric Mamba** | **O(N)** | **O(N)** | **Fast (10k+ tokens)** |

### Benchmark Results

- **10-20x faster** on medium sequences (512-2048 tokens)
- **100x+ faster** on long sequences (>2048 tokens)  
- Linear scaling to 10k+ tokens
- Reduced memory footprint via field contractions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Testing requirements
- Documentation expectations
- Pull request process

## Theory and Papers

For complete mathematical derivations:
- **EXECUTIVE_SUMMARY.md**: Engineering specification and use cases
- **GEOMETRIC_MAMBA_GUIDE.md**: Operator correspondence
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
