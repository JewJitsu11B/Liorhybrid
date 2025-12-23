"""
Checkpoint Inspection and Management Utilities

Tools for inspecting, comparing, and managing saved checkpoints.

Features:
- Inspect checkpoint contents
- Compare multiple checkpoints
- Recreate train/val/test splits from saved split_info
- Run validation on checkpointed models
"""

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


def inspect_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load and inspect checkpoint contents.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata and statistics
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model config (prefer new model_config, fallback to config)
    model_config = checkpoint.get('model_config', {})
    general_config = checkpoint.get('config', {})

    # Extract field config from field_state
    field_state = checkpoint.get('field_state', {})

    info = {
        'path': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'global_step': checkpoint.get('global_step', 'N/A'),
        'train_loss': checkpoint.get('train_loss', 'N/A'),
        'val_loss': checkpoint.get('val_loss', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
        'lior_loss': checkpoint.get('lior_loss', 'N/A'),
        'field_norm': checkpoint.get('field_norm', 'N/A'),
        'config': general_config,
        'd_model': model_config.get('d_model') or general_config.get('d_model', 'N/A'),
        'n_layers': model_config.get('n_layers') or general_config.get('n_layers', 'N/A'),
        'field_dim': model_config.get('field_dim') or general_config.get('field_dim', 'N/A'),
        'spatial_size': field_state.get('spatial_size', 'N/A'),
        'tensor_dim': field_state.get('tensor_dim', 'N/A'),
    }

    # Extract model parameter count and breakdown
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
        info['model_params'] = total_params

        # Count parameters by layer type
        param_breakdown = {}
        for name, param in model_state.items():
            if isinstance(param, torch.Tensor):
                layer_type = name.split('.')[0]  # Get top-level module name
                param_breakdown[layer_type] = param_breakdown.get(layer_type, 0) + param.numel()
        info['param_breakdown'] = param_breakdown
    else:
        info['model_params'] = 'N/A'
        info['param_breakdown'] = {}

    # Extract field state shape
    if 'field_state_dict' in checkpoint:
        field_state = checkpoint['field_state_dict']
        if 'T' in field_state:
            info['field_shape'] = tuple(field_state['T'].shape)
        else:
            info['field_shape'] = 'N/A'
    else:
        info['field_shape'] = 'N/A'

    # Extract split info for validation set recreation
    split_info = checkpoint.get('split_info', None)
    if split_info:
        info['split_info'] = {
            'has_split_info': True,
            'split_seed': split_info.get('split_seed', 'N/A'),
            'val_split_ratio': split_info.get('val_split_ratio', 'N/A'),
            'test_split_ratio': split_info.get('test_split_ratio', 'N/A'),
            'dataset_length': split_info.get('dataset_length', 'N/A'),
            'train_size': len(split_info.get('train_indices', [])),
            'val_size': len(split_info.get('val_indices', [])),
            'test_size': len(split_info.get('test_indices', [])),
            'has_exact_indices': split_info.get('train_indices') is not None,
        }
    else:
        info['split_info'] = {'has_split_info': False}

    return info


def print_checkpoint_summary(checkpoint_path: str):
    """
    Print formatted summary of checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    info = inspect_checkpoint(checkpoint_path)

    print("=" * 70)
    print(f"CHECKPOINT SUMMARY: {Path(checkpoint_path).name}")
    print("=" * 70)
    print(f"Epoch:              {info['epoch']}")
    print(f"Global Step:        {info['global_step']}")
    print(f"Train Loss:         {info['train_loss']}")
    print(f"Val Loss:           {info['val_loss']}")
    print(f"Best Val Loss:      {info['best_val_loss']}")
    print(f"LIoR Loss:          {info['lior_loss']}")
    print(f"Field Norm:         {info['field_norm']}")
    print(f"Model Parameters:   {info['model_params']:,}" if isinstance(info['model_params'], int) else f"Model Parameters:   {info['model_params']}")
    print(f"Field Shape:        {info['field_shape']}")

    # Print parameter breakdown by layer
    if info.get('param_breakdown'):
        print("\nPARAMETER BREAKDOWN:")
        breakdown = info['param_breakdown']
        for layer_name, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / info['model_params'] * 100) if isinstance(info['model_params'], int) else 0
            print(f"  {layer_name:<20} {count:>12,} ({percentage:>5.1f}%)")

    print("=" * 70)

    # Print key config parameters
    config = info['config']
    if config:
        print("CONFIG:")
        print(f"  Spatial Size:     {config.get('spatial_size', 'N/A')}")
        print(f"  Tensor Dim:       {config.get('tensor_dim', 'N/A')}")
        print(f"  d_model:          {config.get('d_model', 'N/A')}")
        print(f"  n_layers:         {config.get('n_layers', 'N/A')}")
        print(f"  lambda_QR:        {config.get('lambda_QR', 'N/A')}")
        print(f"  lambda_F:         {config.get('lambda_F', 'N/A')}")
        print(f"  alpha:            {config.get('alpha', 'N/A')}")
        print("=" * 70)


def compare_checkpoints(checkpoint_paths: List[str]):
    """
    Compare multiple checkpoints side by side.

    Args:
        checkpoint_paths: List of checkpoint file paths
    """
    if not checkpoint_paths:
        print("No checkpoints provided for comparison.")
        return

    infos = [inspect_checkpoint(path) for path in checkpoint_paths]

    print("=" * 100)
    print("CHECKPOINT COMPARISON")
    print("=" * 100)

    # Print header
    print(f"{'Metric':<20}", end="")
    for i, path in enumerate(checkpoint_paths):
        name = Path(path).name[:20]
        print(f"{name:>20}", end="")
    print()
    print("-" * 100)

    # Print comparison rows
    metrics = ['epoch', 'global_step', 'train_loss', 'val_loss', 'best_val_loss',
               'lior_loss', 'field_norm', 'model_params']

    for metric in metrics:
        print(f"{metric:<20}", end="")
        for info in infos:
            value = info.get(metric, 'N/A')
            if isinstance(value, float):
                print(f"{value:>20.6f}", end="")
            elif isinstance(value, int):
                print(f"{value:>20,}", end="")
            else:
                print(f"{str(value):>20}", end="")
        print()

    print("=" * 100)


def export_checkpoint_info(checkpoint_path: str, output_path: str):
    """
    Export checkpoint information to JSON file.

    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Path to output JSON file
    """
    info = inspect_checkpoint(checkpoint_path)

    # Convert tensors to serializable format
    serializable_info = {}
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            serializable_info[key] = value.tolist()
        elif isinstance(value, tuple):
            serializable_info[key] = list(value)
        else:
            serializable_info[key] = value

    with open(output_path, 'w') as f:
        json.dump(serializable_info, f, indent=2)

    print(f"Checkpoint info exported to: {output_path}")


def find_best_checkpoint(checkpoint_dir: str, metric: str = 'val_loss') -> Optional[str]:
    """
    Find checkpoint with best metric value in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to optimize ('val_loss', 'train_loss', etc.)

    Returns:
        Path to best checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return None

    checkpoint_files = list(checkpoint_dir.glob('*.pt')) + list(checkpoint_dir.glob('*.pth'))

    if not checkpoint_files:
        print(f"No checkpoints found in: {checkpoint_dir}")
        return None

    best_checkpoint = None
    best_value = float('inf')

    for ckpt_path in checkpoint_files:
        try:
            info = inspect_checkpoint(str(ckpt_path))
            value = info.get(metric)

            if value != 'N/A' and value < best_value:
                best_value = value
                best_checkpoint = str(ckpt_path)
        except Exception as e:
            print(f"Error reading {ckpt_path.name}: {e}")
            continue

    if best_checkpoint:
        print(f"Best checkpoint ({metric}={best_value:.6f}): {Path(best_checkpoint).name}")
    else:
        print(f"No valid checkpoints with metric '{metric}' found.")

    return best_checkpoint


# =============================================================================
# Split Recreation and Validation Functions
# =============================================================================

def get_split_info(checkpoint_path: str) -> Optional[Dict]:
    """
    Extract split information from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dict with split info or None if not present
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('split_info', None)


def recreate_splits(
    full_dataset: Dataset,
    checkpoint_path: str
) -> Tuple[Subset, Subset, Subset]:
    """
    Recreate exact train/val/test splits from checkpoint.

    Uses saved indices for exact reproducibility. Falls back to
    seed-based recreation if indices not saved.

    Args:
        full_dataset: The full dataset to split (must be same as original)
        checkpoint_path: Path to checkpoint with split info

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Raises:
        ValueError: If checkpoint has no split_info or dataset length mismatch
    """
    split_info = get_split_info(checkpoint_path)

    if split_info is None:
        raise ValueError(
            f"Checkpoint has no split_info. Cannot recreate splits. "
            f"This checkpoint was saved before split_info was implemented."
        )

    # Verify dataset length matches
    saved_length = split_info.get('dataset_length')
    if saved_length and len(full_dataset) != saved_length:
        raise ValueError(
            f"Dataset length mismatch: checkpoint expected {saved_length}, "
            f"but got {len(full_dataset)}. Use the same dataset as training."
        )

    # Prefer exact indices if saved (100% reproducibility)
    if split_info.get('train_indices') is not None:
        train_dataset = Subset(full_dataset, split_info['train_indices'])
        val_dataset = Subset(full_dataset, split_info['val_indices'])
        test_dataset = Subset(full_dataset, split_info['test_indices'])
        print(f"✓ Recreated splits from exact indices: "
              f"{len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    else:
        # Fallback: recreate from seed (should produce same split)
        from torch.utils.data import random_split

        val_ratio = split_info.get('val_split_ratio', 0.1)
        test_ratio = split_info.get('test_split_ratio', 0.1)
        seed = split_info.get('split_seed', 42)

        total_size = len(full_dataset)
        val_size = int(total_size * val_ratio)
        test_size = int(total_size * test_ratio)
        train_size = total_size - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        print(f"✓ Recreated splits from seed {seed}: "
              f"{train_size} train, {val_size} val, {test_size} test")

    return train_dataset, val_dataset, test_dataset


def run_validation_from_checkpoint(
    checkpoint_path: str,
    full_dataset: Dataset,
    model_class,
    field_class,
    device: str = 'cuda',
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Load checkpoint, recreate validation set, run validation, return metrics.

    This is the main entry point for validating checkpointed models.

    Args:
        checkpoint_path: Path to checkpoint file
        full_dataset: Full dataset (same as used in training)
        model_class: Model class to instantiate
        field_class: Field class to instantiate
        device: Device to run validation on
        batch_size: Batch size for validation

    Returns:
        Dict with validation metrics (val_loss, perplexity, etc.)

    Example:
        from Liorhybrid.inference.geometric_transformer import GeometricTransformerWithMamba
        from Liorhybrid.core.field import CognitiveTensorField
        from Liorhybrid.training.datasets import TextDataset

        dataset = TextDataset("training_data/", tokenizer, max_length=512)
        metrics = run_validation_from_checkpoint(
            checkpoint_path="checkpoints/best_model.pt",
            full_dataset=dataset,
            model_class=GeometricTransformerWithMamba,
            field_class=CognitiveTensorField,
        )
        print(f"Val loss: {metrics['val_loss']:.4f}")
    """
    import torch.nn.functional as F

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Recreate splits
    _, val_dataset, _ = recreate_splits(full_dataset, checkpoint_path)

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Instantiate model and field
    model_config = checkpoint.get('model_config', {})
    d_model = model_config.get('d_model') or config.get('d_model', 256)
    n_layers = model_config.get('n_layers') or config.get('n_layers', 4)

    model = model_class(
        d_model=d_model,
        n_layers=n_layers,
    ).to(device)

    field = field_class().to(device)

    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'field_state' in checkpoint:
        field_state = checkpoint['field_state']
        field.T = field_state['T'].to(device)

    model.eval()

    # Run validation
    total_loss = 0.0
    total_tokens = 0
    all_losses = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = model(input_ids, T_field=field.T)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )

            num_tokens = labels.numel()
            total_loss += loss.item()
            total_tokens += num_tokens
            all_losses.append(loss.item() / num_tokens)

    avg_loss = total_loss / total_tokens
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e10)

    metrics = {
        'val_loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': len(val_loader),
        'num_tokens': total_tokens,
        'checkpoint': checkpoint_path,
    }

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS: {Path(checkpoint_path).name}")
    print(f"{'='*50}")
    print(f"Val Loss:    {avg_loss:.4f}")
    print(f"Perplexity:  {perplexity:.2f}")
    print(f"Tokens:      {total_tokens:,}")
    print(f"{'='*50}\n")

    return metrics


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python checkpoint_utils.py <checkpoint_path>         # Inspect single checkpoint")
        print("  python checkpoint_utils.py <path1> <path2> ...       # Compare multiple checkpoints")
        sys.exit(1)

    if len(sys.argv) == 2:
        print_checkpoint_summary(sys.argv[1])
    else:
        compare_checkpoints(sys.argv[1:])
