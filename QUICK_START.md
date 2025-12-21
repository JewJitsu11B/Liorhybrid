# Quick Start Guide

## Demo vs Production

### `demo_geometric_mamba.py` - Educational Only

This is **just for understanding** how geometric operators work:
- Shows ComplexOctonion operations
- Demonstrates Trinor/Wedge/Spinor individually
- Prints comparison tables
- **NOT for training** - just educational

**Run it to learn:**
```bash
python demo_geometric_mamba.py
```

You'll see:
- How CI8 operations work
- How Trinor replaces matrix A
- How Wedge replaces matrix B
- How Spinor replaces matrix C
- Comparison tables

### `main.py` - Production Training

This is your **actual training interface**. The new Geometric Mamba is already integrated.

**Run it for real training:**
```bash
python main.py
```

You'll get an interactive menu:

```
================================================================================
  BAYESIAN COGNITIVE FIELD - Training System
  Advanced Physics-Based Multimodal AI
================================================================================

┌─ MAIN MENU ─────────────────────────────────────────────────┐
│                                                              │
│  1. Quick Start (Geometric Training - Recommended)          │
│  2. Full Training (Train Everything End-to-End)             │
│  3. Resume from Checkpoint                                  │
│  4. Generate Sample Dataset                                 │
│  5. Exit                                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘

▶ Select option [1-5]:
```

## Choosing Your Architecture

When you select option 1 or 2, you'll be asked:

```
┌─ ARCHITECTURE ───────────────────────────────────────────────┐
│  1. Standard Transformer (O(N²) attention)                   │
│  2. Geometric Mamba (O(N) state-space) - RECOMMENDED         │
└──────────────────────────────────────────────────────────────┘

▶ Select architecture [1-2]:
```

### Option 1: Standard Transformer (Old)
- Uses `GeometricTransformer`
- O(N²) complexity (slow for long sequences)
- Field-contracted geometric products (memory-efficient)
- Works with existing checkpoints

### Option 2: Geometric Mamba (New) ⭐ RECOMMENDED
- Uses `GeometricTransformerWithMamba`
- **O(N) complexity** (much faster)
- CI8 state space (16D: 8 amplitudes + 8 phases)
- Trinor/Wedge/Spinor geometric operators
- DPR integration for statistically optimal K/V
- SBERT pooling for sequence aggregation

## What Happens Behind the Scenes

When you choose **Geometric Mamba**, main.py creates:

```python
model = GeometricTransformerWithMamba(
    d_model=config['d_model'],
    n_mamba_layers=config['n_layers'],    # O(N) base processing
    n_attention_layers=2,                  # Geometric attention on top
    n_heads=config['n_heads'],
    field_dim=config['field_dim'],
    use_dpr=True,                          # Enable DPR K/V generation
    use_positional_encoding=True,
    use_temporal_encoding=True
)
```

This gives you the full stack:
```
Input → Embeddings → Geometric Mamba (O(N))
                          ↓
                    SBERT Pooling
                          ↓
                    DPR K/V Generation
                          ↓
              Geometric Attention (field-contracted)
                          ↓
                      Output
```

## Quick Start Examples

### 1. Test with Sample Data (Geometric Mamba)

```bash
python main.py
```

Then select:
- `1` - Quick Start
- `sample` - Use sample data
- `2` - Geometric Mamba (O(N))
- `1` - Quick config (defaults)
- `Y` - Start training

### 2. Train on Your Data (Geometric Mamba)

```bash
python main.py
```

Then select:
- `1` - Quick Start
- `path/to/your/data.txt` - Your data path
- `2` - Geometric Mamba (O(N))
- `2` - Custom config
- Configure your parameters
- `Y` - Start training

### 3. Full Training with LIoR

Edit the trainer config in `main.py` or modify after initial setup:

```python
trainer_config = {
    'training_mode': 'full',
    'use_lior': True,                    # Enable geodesic learning
    'lior_loss_weights': {
        'lm': 1.0,                       # Language modeling
        'geodesic': 0.1,                 # Geodesic cost (LIoR)
        'contrastive': 0.01,             # Contrastive (multimodal)
        'field_entropy': 0.001           # Field regularization
    },
    'max_epochs': 10,
    'use_amp': True,
    'output_dir': './checkpoints/lior'
}
```

## Performance Expectations

### Standard Transformer (O(N²))
- Short sequences (<512 tokens): Fast
- Medium sequences (512-2048): Manageable
- Long sequences (>2048): **Very slow**

### Geometric Mamba (O(N))
- Short sequences: Slightly faster
- Medium sequences: **Much faster** (~10-20x)
- Long sequences: **Dramatically faster** (~100x+)
- Scales to 10k+ tokens

## Logging Output

With comprehensive logging enabled, you'll see:

```
================================================================================
EPOCH 1 | BATCH 42 | STEP 168
================================================================================

TIMING:
  Batch time:    0.234s
  Step time:     0.189s
  Throughput:    68.4 samples/s

COMPUTATIONAL COMPLEXITY:
  Complexity:    O(N)                    ← Shows O(N) for Mamba!
  Est. GFLOPs:   15.23

LOSSES:
  Total:         2.345678
  LM Loss:       2.123456
  Geodesic:      0.156789                ← LIoR geodesic cost

FIELD STATE:
  Alpha:         0.123456
  Nu (mean):     0.234567
  Tau (mean):    0.345678
  Hamiltonian:   1.234567
  |∇H|:          0.012345                ← Entropy gradient

GEOMETRIC WEIGHTS:
  Wedge:         0.3421
  Tensor:        0.4123
  Spinor:        0.2456
```

## Files You Care About

### For Training (Production):
- `main.py` - **Start here** - interactive training interface
- `IMPLEMENTATION_SUMMARY.md` - Complete technical details
- `GEOMETRIC_MAMBA_GUIDE.md` - Operator correspondence guide

### For Learning (Educational):
- `demo_geometric_mamba.py` - **Run this first** to understand the math
- Example outputs showing how operators work

## Common Questions

### Q: Should I use Standard or Geometric Mamba?
**A**: Use **Geometric Mamba** unless you have a specific reason not to. It's:
- Faster (O(N) vs O(N²))
- More interpretable (CI8 has physical meaning)
- Causally structured (non-associative algebra)
- Memory efficient (field contractions)

### Q: Can I switch between them?
**A**: Yes! Just change your selection in the menu. Both use the same field and training infrastructure.

### Q: Do I need DPR?
**A**: It's recommended but optional. DPR provides statistically optimal K/V vectors. If you don't have `transformers` installed, it will use a fallback.

### Q: What's the difference between geometric and full training modes?
**A**:
- **Geometric mode**: Freezes embeddings, trains only geometric weights + field params
- **Full mode**: Trains everything end-to-end

Choose geometric mode for faster experimentation, full mode for production.

### Q: How do I enable LIoR training?
**A**: Set `use_lior: True` in the trainer config. This enables:
- Geodesic cost (path optimization through field)
- Adaptive parameter updates via ∇H (not ∇L)
- Physics-guided learning

## Next Steps

1. **Run the demo** to understand the math:
   ```bash
   python demo_geometric_mamba.py
   ```

2. **Train on sample data** to test:
   ```bash
   python main.py  # Select: 1 → sample → 2 → 1 → Y
   ```

3. **Read the guides**:
   - `GEOMETRIC_MAMBA_GUIDE.md` - Operator correspondence
   - `IMPLEMENTATION_SUMMARY.md` - Full technical details

4. **Train on your data**:
   ```bash
   python main.py  # Select: 1 → your_data_path → 2 → configure → Y
   ```

## Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `demo_geometric_mamba.py` | Education | Learn how operators work |
| `main.py` | Production | Actually train models |
| `GEOMETRIC_MAMBA_GUIDE.md` | Reference | Understand math |
| `IMPLEMENTATION_SUMMARY.md` | Reference | Technical details |

**Start with the demo to learn, then use main.py to train.**
