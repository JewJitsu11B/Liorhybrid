# Liorhybrid

A PyTorch implementation of physics-inspired AI combining Bayesian cognitive field dynamics with geometric algebra for interpretable, efficient learning.

## Overview

Liorhybrid is a research framework that bridges theoretical physics and modern machine learning. It implements a **Cognitive Tensor Field** $T_{ij}(x,t)$ that evolves under physics-inspired dynamics, integrated with transformer architectures via geometric attention mechanisms.

### Core Components

**1. Cognitive Tensor Field**
- Rank-2 complex tensor field $T_{ij}(x,t) \in \mathbb{C}^{D \times D}$ at each spatial location
- Evolves via Bayesian recursive dynamics with fractional memory
- Governed by master equation: $i\hbar_{cog} \partial_t T = H[T] + \Lambda_{QR}[T] - \Lambda_F[T] + J$
- Hamiltonian evolution + Bayesian updates + power-law memory kernel

**2. LIoR (Learning in Operator Regime)**
- Geodesic-based optimization through field Hamiltonians
- Parameters update via entropy gradients $\nabla H$ rather than loss gradients $\nabla L$
- Geodesic cost measures deviation from physics-guided optimal paths
- O(1) recurrence for efficient memory kernel computation

**3. Geometric Attention**
- Replaces standard dot-product attention with geometric products
- **Wedge product**: Antisymmetric (captures orthogonality between concepts)
- **Tensor product**: Symmetric (captures correlations)
- **Spinor product**: Rotational invariants (captures phase structure)
- Field-contracted operations avoid OOM on large tensors

**4. Biquaternion Algebra**
- 16-DOF state space: two complex quaternions (Q_M for present, Q_H for memory)
- Pure real arithmetic (fp16/bf16 compatible, avoids ComplexHalf bugs)
- SL(2,C) transformations represent Lorentz rotations + boosts in cognitive spacetime

### Key Features

- **Physics-Guided Learning**: Evolution driven by physical principles, not just gradient descent
- **Interpretable Representations**: Field dynamics have clear mathematical/physical meaning
- **Memory Efficient**: Field contractions reduce O(d²) outer products to O(d) operations
- **Adaptive Parameters**: Field parameters (α, ν, τ) learn optimal values during training
- **Multi-modal**: Supports text, images, and video through field encoding
- **GPU Accelerated**: Full PyTorch with CUDA support

## Installation

### From source

```bash
git clone https://github.com/JewJitsu11B/Liorhybrid.git
cd Liorhybrid
pip install -e .
```

### Dependencies

**Required:**
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0

**Optional (for DPR K/V generation):**
- transformers (HuggingFace)

**Development:**
- pytest ≥ 7.0.0
- matplotlib ≥ 3.5.0 (for visualization)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive Training

Launch the interactive training interface:

```bash
python main.py
```

Available options:
1. Quick Start (Geometric Training - Recommended)
2. Full Training (Train Everything End-to-End)
3. Resume from Checkpoint
4. Generate Sample Dataset
5. Inference/Chat Mode
6. Inspect Checkpoint
7. Evaluate Checkpoint (Run Validation)
8. Config Cost Calculator
9. Exit

### Basic Field Evolution

```python
from Liorhybrid import CognitiveTensorField, FAST_TEST_CONFIG

# Create field with default configuration
field = CognitiveTensorField(FAST_TEST_CONFIG)

# Run evolution
for step in range(100):
    field.evolve_step()
    
    if step % 20 == 0:
        print(f"Step {step}: ||T||² = {field.get_norm_squared():.6f}")
```

### Transformer Training with LIoR

```python
from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.inference import GeometricTransformer
from Liorhybrid.training import CognitiveTrainer, TextDataset, CognitiveTokenizer

# Initialize field
field_config = FieldConfig(
    spatial_size=(16, 16),
    tensor_dim=16,
    adaptive_learning=True  # Enable adaptive α, ν, τ
)
field = CognitiveTensorField(field_config)

# Initialize tokenizer
tokenizer = CognitiveTokenizer()

# Create geometric transformer
model = GeometricTransformer(
    d_model=512,
    n_layers=6,
    n_heads=8,
    field_dim=16,
    field=field  # Connect to cognitive field
)

# Load dataset
dataset = TextDataset("path/to/data.txt", tokenizer, max_length=512)

# Train with LIoR
trainer = CognitiveTrainer(
    model=model,
    field=field,
    tokenizer=tokenizer,
    use_lior=True,           # Enable geodesic optimization
    lior_loss_weights={
        'lm': 1.0,            # Language modeling
        'geodesic': 0.1,      # Geodesic cost
        'field_entropy': 0.001  # Field regularization
    },
    max_epochs=10
)

trainer.train(dataset)
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Cognitive Tensor Field                    │
│   T_ij(x,t) evolves via Bayesian recursive dynamics         │
│   • Hamiltonian evolution (kinetic + potential)             │
│   • Bayesian updates (belief revision)                      │
│   • Fractional memory (power-law kernel)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │ Provides metric & key/value states
                  ↓
┌─────────────────────────────────────────────────────────────┐
│               Geometric Transformer Layer                    │
│   • Input → Embeddings                                      │
│   • Query generation from input                             │
│   • Key/Value extraction from field                         │
│   • Geometric attention (wedge/tensor/spinor products)      │
│   • Output generation                                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                   LIoR Training Loop                        │
│   Loss = CrossEntropy + w_geo * GeodesicCost               │
│   • Standard gradients → model parameters                   │
│   • Entropy gradients → field parameters (α, ν, τ)         │
│   • Geodesic cost guides optimization through field         │
└─────────────────────────────────────────────────────────────┘
```

### Field Evolution Equation

The cognitive tensor field evolves according to:

```
iℏ_cog ∂_t T_ij = H[T]_ij + Λ_QR[T]_ij - Λ_F[T]_ij + J_ij
```

Where:
- **H[T]**: Hamiltonian operator (kinetic + potential energy)
  ```
  H[T]_ij = -(ℏ²_cog/2m_cog) ∇²T_ij + V_ij T_ij
  ```

- **Λ_QR[T]**: Bayesian recursive update (drives toward evidence)
  ```
  Λ_QR[T]_ij = λ_QR (B[T(t-Δt)]_ij - T_ij(t-Δt))
  B[T]_ij = (w_ij T_ij) / Z    where w_ij = exp(-|T_ij - E_ij|²/τ)
  ```

- **Λ_F[T]**: Fractional memory kernel (long-range temporal correlations)
  ```
  Λ_F[T]_ij = λ_F ∫₀ᵗ K(t-τ) T_ij(τ) dτ
  K(τ) = τ^(α-1) / Γ(α)
  ```

- **J**: External input/stimulus

### LIoR Memory Kernel

Efficient O(1) recurrence for non-Markovian dynamics:

```
K_L(dt) = α·exp(-β·dt)                      # Exponential (Markov)
        - γ·dt^(-δ)·exp(-ξ·dt)               # Power-law (Fractional)
        + η·cos(ω·dt + φ)·exp(-ζ·dt)         # Oscillatory (Phase)
```

State update: `m_t = ρ·m_{t-1} + η·x_t - ξ·x_{t-p}`

### Geometric Products

Field-contracted attention products (memory-efficient):

**Wedge Product** (antisymmetric):
```python
score(i,j) = Σ_μν T_μν (Q_i^μ K_j^ν - K_j^μ Q_i^ν)
```
- High score = orthogonal concepts (Q⊥K)
- Captures novelty and complementarity

**Tensor Product** (symmetric):
```python
score(i,j) = ||Q_i|| × ||K_j|| × Tr(T)
```
- Captures signal strength and correlations
- Modulated by field magnitude

**Spinor Product** (rotational):
```python
score(i,j) = Re(Q_i^† σ_μ K_j) T^μ
```
- Extracts rotational invariants
- Captures phase structure and orientation

### Directory Structure

```
Liorhybrid/
├── core/
│   ├── config.py              # Field configuration parameters
│   └── tensor_field.py        # CognitiveTensorField implementation
├── models/
│   ├── biquaternion.py        # Biquaternion algebra (16-DOF state)
│   ├── lior_kernel.py         # LIoR memory kernel (O(1) recurrence)
│   ├── causal_field.py        # Causal field dynamics
│   ├── complex_metric.py      # Metric tensor computations
│   └── manifold.py            # Geometric manifold operations
├── inference/
│   ├── geometric_attention.py # Geometric attention mechanisms
│   ├── geometric_products.py  # Wedge/tensor/spinor products
│   ├── field_extraction.py    # Extract K/V from field
│   └── dpr_encoder.py         # DPR K/V generation (optional)
├── training/
│   ├── trainer.py             # Standard training loop
│   ├── lior_trainer.py        # LIoR geodesic training
│   ├── lior_optimizer.py      # Entropy-based optimization
│   ├── biquat_optimizer.py    # Biquaternion-specific optimizer
│   ├── losses.py              # Loss functions (geodesic, entropy)
│   ├── tokenizer.py           # CognitiveTokenizer
│   └── datasets.py            # Text/Image/Video datasets
├── kernels/
│   ├── hamiltonian.py         # Hamiltonian operator H[T]
│   ├── bayesian.py            # Bayesian update Λ_QR[T]
│   └── fractional_memory.py   # Fractional memory Λ_F[T]
├── operators/
│   └── collapse.py            # Field collapse and measurement
├── utils/
│   ├── metrics.py             # Training metrics and diagnostics
│   └── visualization.py       # Plotting utilities
├── tests/                     # Test suite
├── examples/                  # Usage examples
│   ├── geometric_inference.py # Field-based inference demo
│   └── mnist_clustering.py    # Self-tokenization (WIP)
└── main.py                    # Interactive training interface
```

## Key Parameters

| Symbol | Name | Default | Range | Description |
|--------|------|---------|-------|-------------|
| ℏ_cog | Cognitive Planck constant | 0.1 | 0.01-1.0 | Sets quantum-like evolution scale |
| m_cog | Effective mass | 1.0 | 0.1-10.0 | Controls diffusion rate |
| λ_QR | Bayesian update gain | 0.3 | 0.1-0.5 | Belief revision strength |
| λ_F | Memory damping | 0.05 | 0.01-0.1 | Fractional memory strength |
| α | Fractional order | 0.5 | 0.3-0.7 | Memory decay rate (power-law exponent) |
| τ | Bayesian temperature | 0.5 | 0.1-1.0 | Evidence sharpness |
| ν | Geodesic coupling | 1.0 | 0.1-10.0 | Field-embedding coupling strength |
| D | Tensor dimension | 16 | ≥16 | Internal DOF (must be ≥16 for overdetermination) |

### Adaptive Learning

When `adaptive_learning=True`, parameters α, ν, and τ become learnable spatial fields that optimize via entropy gradients:

```python
∂α/∂t = -η_α ∂H/∂α    # Minimize field entropy
∂ν/∂t = -η_ν ∂H/∂ν    # Optimize coupling
∂τ/∂t = -η_τ ∂H/∂τ    # Adjust temperature
```

where H = -Tr(T log T) is the field entropy.

## Features & Capabilities

### Implemented ✓

**Core Field Dynamics:**
- Complete tensor field evolution (master equation)
- All three kernel operators (H, Λ_QR, Λ_F)
- O(1) LIoR memory kernel with multi-mode recurrence
- Adaptive parameter learning (α, ν, τ)
- Biquaternion state representation (16-DOF)

**Geometric Attention:**
- Field-contracted geometric products (wedge/tensor/spinor)
- Memory-efficient attention (O(seq²) instead of O(seq²·d²))
- Multiple attention modes with learned mixing weights

**Training Infrastructure:**
- LIoR geodesic optimization
- Entropy-based parameter updates
- Standard trainer with LM/contrastive/alignment losses
- Comprehensive metrics and logging
- Checkpoint management

**Data Support:**
- Text datasets with cognitive tokenization
- Image/video dataset interfaces
- Multi-modal data loading

### Current Limitations ⚠

- MNIST self-tokenization example incomplete
- Visualization utilities basic
- DPR integration optional (requires transformers library)
- Some geometric inference examples need updating

### Research Directions 📋

- **Semantic Addressing**: Metric tensor + Christoffel symbols for navigating concept space
- **Route Hashing**: BCH error correction for stable addressing
- **Neighbor Structures**: Efficient k-NN in field space
- **Active Inference**: Integrate free energy principle
- **Multi-scale Fields**: Hierarchical field resolutions

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test specific components:
```bash
pytest tests/test_conservation.py  # Norm conservation
pytest tests/test_bayesian.py      # Bayesian updates
pytest tests/test_memory.py        # Fractional memory
```

## Examples

### Field Evolution Demo

```bash
python examples/geometric_inference.py
```

Shows how the field evolves and connects to transformer inference.

### MNIST Clustering (WIP)

```bash
python examples/mnist_clustering.py
```

Demonstrates emergent clustering via field dynamics (work in progress).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{liorhybrid2025,
  title={Liorhybrid: Physics-Inspired AI with Bayesian Cognitive Fields},
  author={Leizerman, Sam},
  year={2025},
  url={https://github.com/JewJitsu11B/Liorhybrid}
}
```

## License

[To be determined]

## Contributing

Contributions are welcome! Please open an issue to discuss major changes.

## Contact

For questions or collaboration:
- Open an issue on [GitHub](https://github.com/JewJitsu11B/Liorhybrid/issues)
- See documentation files: `QUICK_START.md`, `IMPLEMENTATION_SUMMARY.md`, `TRAINING.md`

