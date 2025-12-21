# Training Guide

Complete guide for training the Bayesian Cognitive Field with Geometric Attention.

## Overview

The system supports **two training modes**:

### 1. Geometric-Only Training
- **Trains**: Geometric weights (wedge, tensor, spinor) + field parameters (α, ν, τ)
- **Freezes**: All embeddings
- **Use case**: Fine-tuning geometric attention on pre-trained embeddings
- **Faster**: Less parameters to optimize

### 2. Full Training
- **Trains**: Everything (embeddings + geometric + field + transformer)
- **Use case**: Training from scratch or full fine-tuning
- **Powerful**: Full end-to-end optimization

## Quick Start

### Geometric-Only Training
```bash
python -m bayesian_cognitive_field.main \
    --mode geometric \
    --data-path ./data/text/train.txt \
    --val-data-path ./data/text/val.txt \
    --data-type text \
    --config configs/train_geometric.yaml
```

### Full Training
```bash
python -m bayesian_cognitive_field.main \
    --mode full \
    --data-path ./data/multimodal/train \
    --val-data-path ./data/multimodal/val \
    --data-type text \
    --config configs/train_full.yaml
```

## Data Formats

### Text Data
**Format**: Plain text files (one document per line) or JSONL

**Example** (`train.txt`):
```
This is a sample document about cognitive fields.
Another document with different content.
...
```

**JSONL format** (`train.jsonl`):
```json
{"text": "First document content"}
{"text": "Second document content"}
```

### Image-Text Data
**Format**: JSONL with image paths

**Example** (`train.jsonl`):
```json
{"image": "images/img001.jpg", "text": "A cat sitting on a mat"}
{"image": "images/img002.jpg", "text": "A dog playing in the park"}
```

### Video-Text Data
**Format**: JSONL with frame directories

**Example** (`train.jsonl`):
```json
{"frames_dir": "videos/video001/", "text": "A person walking"}
{"frames_dir": "videos/video002/", "text": "Cars driving on highway"}
```

Frame directories should contain: `frame_0000.jpg`, `frame_0001.jpg`, ...

## Configuration

### Using Config Files
```bash
python -m bayesian_cognitive_field.main --config configs/train_full.yaml
```

### Command Line Arguments
Override any config value:
```bash
python -m bayesian_cognitive_field.main \
    --mode full \
    --data-path ./data \
    --batch-size 32 \
    --lr 0.0003 \
    --max-epochs 10 \
    --d-model 768 \
    --n-layers 12
```

### Key Parameters

#### Model Architecture
```yaml
field_dim: 4              # Tensor dimension D (4, 8, 16)
spatial_size: [8, 8]      # Field grid size
d_model: 512              # Transformer hidden dimension
n_heads: 8                # Attention heads
n_layers: 6               # Transformer depth
vocab_size: 32000         # Vocabulary size
```

#### Training
```yaml
batch_size: 32            # Batch size
max_epochs: 10            # Training epochs
lr: 0.0001                # Learning rate (BiquatOptimizer, pure SGD)
theta_clip: 8.0           # Clamp rotation params for stability
warmup_steps: 1000        # LR warmup
grad_accum_steps: 1       # Gradient accumulation
max_seq_len: 512          # Maximum sequence length
```

#### Field Evolution
```yaml
adaptive_field: true      # Enable α, ν, τ adaptation
evolve_field: true        # Evolve field during training
```

## Training Pipeline

### Architecture Flow

```
Input (text/image/video)
    ↓
Multimodal Embeddings
    ↓ (Q: Query)
Cognitive Field T_ij(x,t)
    ↓ (K, V: Memory)
Geometric Attention
    - Wedge product (antisymmetric span)
    - Tensor product (full correlation)
    - Spinor product (rotational flow)
    ↓
Softmax Normalization
    ↓
Transformer Layers
    ↓
Language Model Head
    ↓
Vocabulary Logits
```

### Loss Functions

**Language Modeling** (always):
```
L_LM = CrossEntropy(logits, labels)
```

**Contrastive** (multimodal):
```
L_contrast = InfoNCE(text_emb, image_emb)
```

**Alignment** (multimodal):
```
L_align = TripletMargin(text, image/video)
```

**Field Regularization**:
```
L_field = (H[T] - target_entropy)²
```

**Combined**:
```
L_total = w_LM * L_LM + w_contrast * L_contrast + w_align * L_align + w_field * L_field
```

## Checkpointing

### Save Checkpoints
Checkpoints are saved automatically:
- Every N steps (configured by `save_interval`)
- End of each epoch
- Best validation loss

Location: `{output_dir}/checkpoint_*.pt`

### Resume Training
```bash
python -m bayesian_cognitive_field.main \
    --mode full \
    --resume ./checkpoints/best_model.pt
```

### Checkpoint Contents
```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': {...},
    'field_state': {
        'T': tensor,  # Field state
        'alpha': tensor,
        'nu': tensor,
        'tau': tensor,
        't': float,
        'step_count': int
    },
    'optimizer_state_dict': {...},
    'train_losses': [...],
    'val_losses': [...],
    'best_val_loss': float,
    'config': {...}
}
```

## Example Training Scripts

### Text-Only (English corpus)
```bash
#!/bin/bash
python -m bayesian_cognitive_field.main \
    --mode full \
    --data-path ./data/english/train.txt \
    --val-data-path ./data/english/val.txt \
    --data-type text \
    --batch-size 64 \
    --max-epochs 20 \
    --lr 0.0003 \
    --d-model 768 \
    --n-layers 12 \
    --adaptive-field \
    --output-dir ./checkpoints/english_lm
```

### Multimodal (Image + Text)
```bash
#!/bin/bash
python -m bayesian_cognitive_field.main \
    --mode full \
    --data-path ./data/coco/train.jsonl \
    --val-data-path ./data/coco/val.jsonl \
    --data-type image-text \
    --batch-size 32 \
    --max-epochs 50 \
    --lr 0.0002 \
    --d-model 768 \
    --n-heads 12 \
    --n-layers 12 \
    --adaptive-field \
    --output-dir ./checkpoints/multimodal
```

### Geometric Fine-Tuning
```bash
#!/bin/bash
# Fine-tune only geometric weights
python -m bayesian_cognitive_field.main \
    --mode geometric \
    --data-path ./data/reasoning/train.txt \
    --val-data-path ./data/reasoning/val.txt \
    --data-type text \
    --batch-size 32 \
    --max-epochs 5 \
    --lr 0.0001 \
    --adaptive-field \
    --output-dir ./checkpoints/reasoning_geometric
```

## Monitoring Training

### Loss Tracking
Training losses are printed during training:
```
Epoch 1/10:
  Train loss: 3.4521
  Val loss: 3.6234
```

### Field State Monitoring
When `adaptive_field=true`, field parameters are logged:
```
Field parameters:
  α: 0.4523
  ν (mean): 0.1234
  τ (mean): 0.5678
```

## Advanced Usage

### Custom Loss Weights
Modify loss weights in config:
```yaml
loss_weights:
  lm: 1.0              # Language modeling
  contrastive: 0.5     # Contrastive (image-text)
  alignment: 0.3       # Multimodal alignment
  field_reg: 0.01      # Field regularization
```

### Multi-GPU Training
(Future feature - placeholder for DDP)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    -m bayesian_cognitive_field.main --mode full ...
```

### RLHF (Future)
```bash
python -m bayesian_cognitive_field.main \
    --mode full \
    --data-path ./data/rlhf/preferences.jsonl \
    --data-type rlhf \
    --rlhf-mode ppo
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Increase `grad_accum_steps`
- Reduce `d_model`, `n_layers`, or `max_seq_len`
- Use `--device cpu` for debugging

### Training Not Converging
- Check learning rate (try 1e-4 to 3e-4)
- Increase `warmup_steps`
- Check data quality
- Verify loss weights are balanced

### Field Entropy Issues
- Adjust `field_reg` weight
- Check adaptive learning is enabled
- Monitor α, ν, τ values

## Next Steps

1. **Prepare Data**: Format your corpus according to the specifications above
2. **Configure**: Choose geometric or full mode, set hyperparameters
3. **Train**: Run training with appropriate config
4. **Evaluate**: Monitor losses and field evolution
5. **Deploy**: Use trained model for inference (see `examples/geometric_inference.py`)

## References

- Paper: "Entropy-Gated Cognitive Field Collapse"
- Geometric Products: Wedge (Grassmann), Tensor, Spinor (Clifford)
- Contrastive Learning: CLIP-style InfoNCE loss
- Field Evolution: Fractional calculus + Bayesian updates
