# Troubleshooting Guide

## Common Warnings (Safe to Ignore)

When you run `python main.py`, you may see these warnings:

### 1. Tensor Dimension Warning
```
UserWarning: tensor_dim=8 < 16 may have insufficient DOF for overdetermination.
```

**What it means**: The default tensor dimension (8) is smaller than the recommended 16 for full overdetermination.

**Is it a problem?**: No, this is just informational. The system will work fine.

**How to fix** (optional): Set `tensor_dim=16` in field config if you want full DOF.

### 2. Transformers Library Warning
```
Warning: transformers library not available. DPR integration will use fallback.
```

**What it means**: HuggingFace transformers is not installed, so DPR will use a simpler fallback.

**Is it a problem?**: No, training will work. DPR fallback uses basic projections instead of pre-trained encoders.

**How to fix** (optional): Install transformers if you want pre-trained DPR:
```bash
pip install transformers
```

### 3. Torchvision Warning
```
Warning: torchvision not available (operator torchvision::nms does not exist). Image/video datasets will not work.
```

**What it means**: Torchvision has a compatibility issue with your PyTorch version.

**Is it a problem?**: No for text training. Yes if you need image/video datasets.

**How to fix** (if needed):
```bash
# Upgrade torchvision to match your PyTorch version
pip install --upgrade torchvision

# Or downgrade PyTorch to match torchvision
pip install torch==2.0.1 torchvision==0.15.2
```

### 4. TorchInductor Complex Operator Warning
```
UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager.
```

**What it means**: Parts of the graph that use `torch.complex` ops cannot be fully code-generated/fused by TorchInductor, so `torch.compile` may fall back to less-optimized kernels or create more graph partitions.

**Is it a problem?**: Not a correctness problem by itself. It can be a performance problem if complex ops land on the hot path.

**How to improve**:
- Keep complex-only work in the field evolution / diagnostics, not in the compiled token-forward path.
- Prefer the biquaternion (pure real) blocks for compiled training runs.

## torch.compile Safety Rules (No Surprise Sync)

If you enable `torch.compile`, keep these invariants or you will get graph breaks, device-mismatch errors, or slow steps:

1. **No `.to(device)` inside `forward()`** (move tensors via buffers or preflight)
2. **No tokenization / Python text processing inside `forward()`** (DPR seeding happens before training)
3. **No `.item()` inside the compiled region** (convert to floats only for logging/checkpointing)
4. **No module/Parameter creation after optimizer construction** (all trainables must exist before optimizer)

## Verified Working Configuration

Your system NOW works for **text-only training** with:
- ✅ TextDataset
- ✅ Geometric Mamba (O(N))
- ✅ Standard Transformer (O(N²))
- ✅ LIoR geodesic learning
- ✅ Comprehensive logging
- ✅ Field evolution

Not currently working (due to torchvision):
- ❌ ImageTextDataset
- ❌ VideoTextDataset

## Quick Test

Run this to verify everything works:

```bash
cd T:\claudepdf\bayesian_cognitive_field
python main.py
```

Select:
1. Option `1` (Quick Start)
2. Type `sample` (use sample data)
3. Option `2` (Geometric Mamba)
4. Option `1` (Quick config)
5. Type `Y` (start training)

You should see training start without errors!

## Expected Output

```
======================================================================
  BAYESIAN COGNITIVE FIELD - Training System
  Advanced Physics-Based Multimodal AI
======================================================================

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

## Still Having Issues?

1. **Check Python version**: Should be 3.9+
   ```bash
   python --version
   ```

2. **Check PyTorch installation**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. **Verify imports work**:
   ```bash
   python -c "from bayesian_cognitive_field.inference import GeometricMambaEncoder; print('OK')"
   ```

4. **Clear Python cache**:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   # On Windows:
   del /s /q __pycache__
   ```

## What to Do Next

Once `python main.py` loads the menu successfully:

1. **Learn the math** first:
   ```bash
   python demo_geometric_mamba.py
   ```

2. **Train on sample data**:
   ```bash
   python main.py
   # Select: 1 → sample → 2 → 1 → Y
   ```

3. **Train on your data**:
   ```bash
   python main.py
   # Select: 1 → path/to/your/data.txt → 2 → configure → Y
   ```

## Architecture Choices

When prompted for architecture:

- **Option 1: Standard Transformer (O(N²))**
  - Traditional attention mechanism
  - Good for short sequences (<512 tokens)
  - Existing implementation

- **Option 2: Geometric Mamba (O(N)) ⭐ RECOMMENDED**
  - CI8 state space (8 amplitudes + 8 phases)
  - Trinor/Wedge/Spinor geometric operators
  - O(N) complexity - much faster for long sequences
  - Novel causal structure enforcement

## Error Messages

### "ImportError: cannot import name 'X'"

**Solution**: Clear cache and reinstall:
```bash
pip install -e . --force-reinstall --no-cache-dir
```

### "RuntimeError: CUDA out of memory"

**Solution**: Reduce batch size in config or use CPU:
```python
config = {
    'batch_size': 4,  # Reduce from 16
    'device': 'cpu'   # Use CPU if GPU OOM
}
```

### "FileNotFoundError: data not found"

**Solution**: Use absolute path or generate sample data:
```bash
python main.py
# Select: 4 (Generate Sample Dataset)
```

## Performance Tips

1. **Use Geometric Mamba** for long sequences (>512 tokens)
2. **Enable AMP** (automatic mixed precision) for GPU:
   ```python
   config['use_amp'] = True
   ```
3. **Increase batch size** if you have GPU memory:
   ```python
   config['batch_size'] = 32  # Or higher
   ```
4. **Reduce logging interval** for faster training:
   ```python
   config['log_interval'] = 500  # From 50
   ```

## Summary

**Your system is working** for text training with both:
- ✅ Standard Transformer (O(N²))
- ✅ Geometric Mamba (O(N))

Warnings about transformers/torchvision are **safe to ignore** for text-only training.

Ready to train!
