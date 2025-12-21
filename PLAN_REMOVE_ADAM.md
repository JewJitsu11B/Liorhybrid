# Plan: Replace AdamW with BiquatOptimizer

## Summary
Remove AdamW optimizer entirely and replace with a custom `BiquatOptimizer` that uses pure SGD with optional weight decay and theta clamping for Lie algebra parameters.

## Memory Benefit
AdamW stores **2 extra tensors per parameter** (first moment `m` and second moment `v`), meaning it uses ~3x the memory of the model parameters for optimizer state alone. Switching to pure SGD eliminates this overhead entirely.

---

## Files to Modify

### 1. Create: `training/biquat_optimizer.py` (NEW FILE)

```python
import torch

class BiquatOptimizer:
    """
    Pure SGD optimizer with optional weight decay and theta clamping.

    Memory-efficient alternative to AdamW - stores NO optimizer state.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0, theta_clip=8.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.theta_clip = theta_clip

        # For compatibility with lr schedulers
        self.param_groups = [{'lr': lr, 'params': self.params}]

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            # Weight decay (decoupled, like AdamW)
            if self.weight_decay > 0:
                p.mul_(1 - self.lr * self.weight_decay)

            # SGD update
            p -= self.lr * p.grad

            # Theta clamping for Lie algebra stability
            if p.shape[-1] == 24:  # generator projection layer
                p[..., 16:24].clamp_(-self.theta_clip, self.theta_clip)

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'theta_clip': self.theta_clip
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict.get('lr', self.lr)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.theta_clip = state_dict.get('theta_clip', self.theta_clip)
        # Update param_groups for scheduler compatibility
        self.param_groups[0]['lr'] = self.lr
```

---

### 2. Modify: `main.py` (lines 1474-1478)

**Before:**
```python
optimizer = torch.optim.AdamW(
    params,
    lr=config['lr'],
    weight_decay=config['weight_decay']
)
```

**After:**
```python
from training.biquat_optimizer import BiquatOptimizer

optimizer = BiquatOptimizer(
    params,
    lr=config['lr'],
    weight_decay=config['weight_decay'],
    theta_clip=config.get('theta_clip', 8.0)
)
```

Also update line 1036 comment about Adam memory.

---

### 3. Modify: `test_training.py` (lines 105 and 231)

**Before:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
```

**After:**
```python
from training.biquat_optimizer import BiquatOptimizer
optimizer = BiquatOptimizer(model.parameters(), lr=config['lr'])
```

---

### 4. Modify: `tests/test_exponential_form.py` (lines ~105, ~327)

Same pattern - replace AdamW with BiquatOptimizer.

---

### 5. Update: `training/__init__.py`

Add export:
```python
from .biquat_optimizer import BiquatOptimizer
```

---

### 6. Update: `TRAINING.md` (line 117)

Change documentation from AdamW weight decay reference to BiquatOptimizer.

---

## Additional Memory Management Tools

The codebase already has:
1. **AMP (FP16)** via `GradScaler` in `trainer.py:109` - halves activation memory
2. **Gradient accumulation** via `grad_accum_steps` config

Could add (not in scope but good options):
1. **Gradient checkpointing** (`torch.utils.checkpoint`) - trades compute for memory
2. **`torch.cuda.empty_cache()`** calls between batches if OOM persists
3. **Mixed precision for specific layers** - keep critical ops in FP32

---

## Learning Rate Consideration

Pure SGD typically needs a **smaller learning rate** than AdamW. Common starting points:
- AdamW: 1e-4 to 3e-4
- SGD: 1e-3 to 1e-2 (but test with current lr first)

Consider adding a config option or starting with current lr and adjusting if training diverges.

---

## Checksum of Changes

| File | Action | Lines Changed |
|------|--------|---------------|
| `training/biquat_optimizer.py` | CREATE | ~60 lines |
| `main.py` | EDIT | ~5 lines |
| `test_training.py` | EDIT | ~4 lines |
| `tests/test_exponential_form.py` | EDIT | ~4 lines |
| `training/__init__.py` | EDIT | 1 line |
| `TRAINING.md` | EDIT | 1 line |

**Total: ~75 lines of changes**
