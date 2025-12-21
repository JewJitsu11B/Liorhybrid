# Bayesian Cognitive Field - Examples

This directory contains example scripts demonstrating usage of the Bayesian cognitive field framework.

## Available Examples

### 1. Simple Evolution (`simple_evolution.py`)

Basic field evolution with diagnostic output.

**Run:**
```bash
python examples/simple_evolution.py
```

**Demonstrates:**
- Field initialization with configuration
- Evolution loop
- Norm conservation tracking
- Basic diagnostics

**Status:** ✓ Ready to run

---

### 2. MNIST Self-Tokenization (`mnist_clustering.py`)

Emergent clustering on MNIST digits without pre-defined labels.

**Run:**
```bash
python examples/mnist_clustering.py
```

**Demonstrates:**
- Image encoding as field initial conditions
- Self-organization via evolution
- Token extraction from correlation structure
- Cluster visualization

**Status:** ⚠ Stub only - requires implementation of:
- MNIST data loading
- Image → field encoding scheme
- Token-based distance clustering
- Visualization utilities

---

## Configuration

All examples use pre-defined configurations from `core/config.py`:

- **FAST_TEST_CONFIG**: Small 8×8 grid, D=8, for quick tests
- **MNIST_CONFIG**: 28×28 grid, D=16, for image tasks

To customize, modify the config object before creating the field:

```python
from bayesian_cognitive_field.core import FAST_TEST_CONFIG, CognitiveTensorField

config = FAST_TEST_CONFIG
config.lambda_QR = 0.5  # Increase Bayesian update strength
config.alpha = 0.3      # Decrease memory decay rate

field = CognitiveTensorField(config)
```

## Dependencies

- PyTorch
- NumPy
- Matplotlib (for visualization examples)
- pytest (for running tests)

See `requirements.txt` in the root directory.

## Next Steps

After running the basic examples, see:

- `tests/` for validation tests
- `bayesian_recursive_operator.tex` for mathematical background
- Main README for installation and theory overview
