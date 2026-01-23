# Quick Start: MoE Framework

This guide will help you get started with the Mixture-of-Experts (MoE) framework in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/JewJitsu11B/Liorhybrid.git
cd Liorhybrid

# Install dependencies
pip install torch>=2.0.0
pip install -e .
```

## Basic Usage

```python
from moe_framework import MoEConfig, FullyOptimizedMoESystem
import torch

# Configure MoE system
config = MoEConfig(
    input_dim=512,
    hidden_dim=2048,
    output_dim=512,
    num_experts=32,
    top_k_experts=3,
)

# Initialize model
model = FullyOptimizedMoESystem(config).cuda()
model.eval()

# Run inference
with torch.no_grad():
    x = torch.randn(8, 128, 512).cuda()
    output = model(x)  # (8, 128, 512)

print(f"Output shape: {output.shape}")
```

## Training

```python
# Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# Training loop
model.train()
for batch in dataloader:
    x, targets = batch
    loss = model.training_step((x, targets), optimizer, loss_fn)
    print(f'Loss: {loss:.4f}')
```

## Query Knowledge Graph

```python
# Query for similar past insights
query = torch.randn(512).cuda()
results = model.query_knowledge_graph(query, top_k=5)

for result in results:
    print(f"Node {result['node_id']}: Similarity={result['similarity']:.3f}")
```

## Run Examples

```bash
# Run comprehensive demo
python examples/moe_framework_demo.py

# Run tests
pytest tests/test_moe_framework.py -v
```

## Documentation

- **[Complete Implementation Guide](MOE_FRAMEWORK_IMPLEMENTATION.md)**: Full architecture and code
- **[Optimization Guide](MOE_OPTIMIZATION_GUIDE.md)**: CUDA-safe optimizations
- **[Implementation Summary](MOE_IMPLEMENTATION_SUMMARY.md)**: Executive overview
- **[API Reference](moe_framework/README.md)**: Quick reference

## Configuration

### Production Inference (Maximum Speed)

```python
config = MoEConfig(
    use_compile=True,
    compile_mode='max-autotune',
    use_amp=True,
    use_gradient_checkpointing=False,
)
```

### Production Training (Memory Efficient)

```python
config = MoEConfig(
    use_compile=False,
    use_amp=True,
    use_gradient_checkpointing=True,
)
```

## Performance

Expected performance on A100 GPU:

| Configuration | Throughput | Memory |
|---------------|------------|--------|
| Baseline | 100 samples/s | 24GB |
| Optimized | 450-500 samples/s | 10-12GB |

**Speedup**: 3-5x  
**Memory reduction**: 50-85%

## Troubleshooting

### CUDA Out of Memory

```python
config.use_gradient_checkpointing = True
config.max_batch_size = 16
config.use_amp = True
```

### Compilation Too Slow

```python
config.compile_mode = 'reduce-overhead'  # Faster compilation
# or
config.use_compile = False  # Disable compilation
```

## Next Steps

1. Read [Complete Implementation Guide](MOE_FRAMEWORK_IMPLEMENTATION.md)
2. Explore [Optimization Techniques](MOE_OPTIMIZATION_GUIDE.md)
3. Check [examples/](examples/) for more scenarios
4. Run [tests/](tests/) to verify installation

## Support

For issues or questions, see [MOE_IMPLEMENTATION_SUMMARY.md](MOE_IMPLEMENTATION_SUMMARY.md) for detailed information.
