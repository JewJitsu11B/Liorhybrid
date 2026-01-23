# Multimodal Head Modules

Standalone modules for audio, image, and cross-modal processing using the Liorhybrid physics framework.

## Overview

These modules leverage the core physics components (ComplexMetricTensor, LiorKernel, CognitiveManifold, CausalFieldLayer) to provide specialized processing for different modalities and tasks.

## Modules

### 1. AudioCausalHead
**Purpose**: Audio processing with temporal memory and phase structure

**Key Features**:
- LiorKernel for O(1) temporal recurrence
- ComplexMetricTensor for frequency phase structure  
- CausalFieldLayer for parallel processing (replaces RNN)
- Phase orthogonality ensures frequency stability
- Power-law memory captures long-range audio dependencies

**Usage**:
```python
from multimodal_heads import AudioCausalHead

audio_head = AudioCausalHead(
    n_mels=80,
    d_model=512,
    output_type='embedding'
)

# Process spectrogram
embeddings, phase_field = audio_head(spectrogram)
```

### 2. ImageManifoldHead
**Purpose**: Image processing with geodesic spatial geometry

**Key Features**:
- CognitiveManifold for spatial structure (8D geodesic space)
- SpinorBilinears for K0→K1→K2 hierarchical mapping
- ComplexMetricTensor A (Riemannian) for spatial relationships
- ComplexMetricTensor B (symplectic) for color/texture phase
- Geodesic distances for semantic grouping

**Usage**:
```python
from multimodal_heads import ImageManifoldHead

image_head = ImageManifoldHead(
    img_size=224,
    patch_size=16,
    d_model=512
)

# Process images
embeddings, coords = image_head(images, return_manifold_coords=True)
```

### 3. CrossModalFusion
**Purpose**: Multimodal fusion via complex metrics

**Key Features**:
- A_{mu nu} (symmetric) for semantic alignment
- B_{mu nu} (antisymmetric) for cross-modal interference
- Phase orthogonality prevents modality collapse
- Supports concat, add, and attention fusion

**Usage**:
```python
from multimodal_heads import CrossModalFusion

fusion = CrossModalFusion(
    d_model=512,
    num_modalities=3,
    fusion_type='attention'
)

# Fuse modalities
modalities = {
    'text': text_embeddings,
    'audio': audio_embeddings,
    'image': image_embeddings
}
fused, alignment = fusion(modalities, return_alignment=True)
```

### 4. RetrievalHead
**Purpose**: Geodesic-based retrieval on manifold

**Key Features**:
- Embeddings stored on learned Riemannian manifold
- Retrieval via geodesic distance (not cosine similarity)
- LIoR-weighted effective metric for importance
- Supports up to 10K items by default

**Usage**:
```python
from multimodal_heads import RetrievalHead

retrieval = RetrievalHead(
    d_model=512,
    max_items=10000
)

# Add items
retrieval.add_items(embeddings, metadata, resilience_scores)

# Retrieve
distances, indices, metadata = retrieval.retrieve(query, top_k=5)
```

### 5. TimeSeriesHead
**Purpose**: Time series with full LiorKernel capabilities

**Key Features**:
- Exponential mode: Short-term patterns
- Power-law mode: Long-range dependencies
- Oscillatory mode: Periodic/seasonal effects
- O(1) recurrence: Real-time streaming
- Autoregressive forecasting

**Usage**:
```python
from multimodal_heads import TimeSeriesHead

ts_head = TimeSeriesHead(
    input_dim=10,
    d_model=512,
    init_omega=0.5  # For daily patterns
)

# Process time series
output, memory = ts_head(timeseries)

# Forecast
forecast = ts_head.forecast(timeseries, horizon=24)
```

### 6. GraphReasoningHead
**Purpose**: Graph reasoning with parallel transport

**Key Features**:
- Nodes as points in CognitiveManifold
- Edges via ParallelTransport + CliffordConnection
- AssociatorCurrent for non-associative reasoning
- CausalFieldLayer message passing
- Covariant graph neural network

**Usage**:
```python
from multimodal_heads import GraphReasoningHead

graph_head = GraphReasoningHead(
    d_model=512,
    num_layers=3
)

# Process graph
node_output, edge_output = graph_head(
    node_features,
    edge_index,
    edge_attr
)
```

### 7. ControlHead
**Purpose**: RL/robotics control with Hamiltonian structure

**Key Features**:
- State space on CognitiveManifold (configuration)
- Action space from symplectic form B (momentum)
- Hamiltonian energy function
- LiorKernel for temporal credit assignment
- Policy and value heads

**Usage**:
```python
from multimodal_heads import ControlHead

control = ControlHead(
    state_dim=17,
    action_dim=6,
    use_hamiltonian=True
)

# Get action
outputs = control(state, return_value=True)
action = outputs['action']
value = outputs['value']
energy = outputs['energy']  # Hamiltonian
```

## Physics Integration

All modules use the same core physics:

- **ComplexMetricTensor (A+iB)**: Separates semantic (Riemannian) from phase (symplectic)
- **LiorKernel**: O(1) recurrence with exponential, power-law, and oscillatory modes
- **CognitiveManifold**: Geodesic geometry for optimal paths and distances
- **CausalFieldLayer**: Parallel field evolution with Pi, Gamma, Phi tensors
- **Phase Orthogonality**: Guarantees stability and prevents collapse

## CUDA Safety

All modules are CUDA-safe:
- Pre-allocated buffers (no dynamic allocation in hot paths)
- Masked operations (no CPU-GPU synchronization)
- Compatible with `torch.compile` and CUDA graphs
- Fixed-size loops where possible

## Integration with Main Pipeline

These modules are standalone and can be used:
1. As drop-in replacements for standard layers
2. As specialized heads on top of your main model
3. Combined with the MoE framework for expert specialization
4. In multimodal architectures with CrossModalFusion

## Examples

See `examples/multimodal_demo.py` for complete examples of using each head module.

## Requirements

- PyTorch >= 2.0
- Access to Liorhybrid models (ComplexMetricTensor, LiorKernel, etc.)

## License

Same as Liorhybrid main project.
