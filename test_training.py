"""
Test Training Script

Programmatically tests both architectures:
1. Standard Transformer (O(N²))
2. Geometric Mamba (O(N))

Run this to verify everything works without interactive prompts.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time

from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.inference import (
    GeometricTransformer,
    GeometricTransformerWithMamba
)
from Liorhybrid.training import (
    CognitiveTokenizer,
    TextDataset,
    CognitiveTrainer
)

def test_standard_transformer():
    """Test Standard Transformer (O(N²))."""
    print("\n" + "="*80)
    print("TEST 1: Standard Transformer (O(N²))")
    print("="*80)

    # Config
    config = {
        'field_dim': 4,
        'spatial_size': [8, 8],
        'd_model': 128,  # Small for testing
        'n_heads': 4,
        'n_layers': 2,
        'batch_size': 4,
        'max_epochs': 1,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'vocab_size': 256,
        'max_seq_len': 64,
        'num_workers': 0
    }

    print(f"Device: {config['device']}")
    print(f"Model: d={config['d_model']}, layers={config['n_layers']}")

    # Create field
    field_config = FieldConfig(
        spatial_size=tuple(config['spatial_size']),
        tensor_dim=config['field_dim'],
        adaptive_learning=True,
        device=config['device']
    )
    field = CognitiveTensorField(field_config)

    # Create model
    model = GeometricTransformer(
        field_dim=config['field_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        use_positional_encoding=True,
        use_temporal_encoding=True
    )

    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create tiny test dataset
    tokenizer = CognitiveTokenizer(vocab_size=config['vocab_size'])

    # Use sample data
    data_path = Path('./data/sample/train.txt')
    if not data_path.exists():
        print(f"Creating sample data at {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, 'w') as f:
            f.write("The cognitive field evolves according to differential equations.\n")
            f.write("Geometric attention uses wedge and tensor products for reasoning.\n")

    dataset = TextDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        max_length=config['max_seq_len']
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,          # Parallel data loading (4 CPU workers)
        pin_memory=True,        # Faster CPU→GPU transfer
        prefetch_factor=4,      # Each worker prefetches 4 batches
        persistent_workers=True # Keep workers alive between epochs
    )

    print(f"✓ Dataset: {len(dataset)} examples")

    # Create optimizer
    from Liorhybrid.training.biquat_optimizer import BiquatOptimizer
    optimizer = BiquatOptimizer(model.parameters(), lr=config['lr'])

    # Create trainer
    trainer_config = {
        'training_mode': 'geometric',
        'max_epochs': 1,
        'grad_accum_steps': 1,
        'use_amp': False,  # Disable for testing
        'clip_grad_norm': 1.0,
        'log_interval': 1,
        'eval_interval': 1000,
        'save_interval': 1000,
        'output_dir': './checkpoints/test_standard',
        'vocab_size': config['vocab_size'],
        'max_seq_len': config['max_seq_len']
    }

    trainer = CognitiveTrainer(
        model=model,
        field=field,
        train_loader=loader,
        val_loader=None,
        optimizer=optimizer,
        device=config['device'],
        config=trainer_config,
        tokenizer=tokenizer  # For inference
    )

    # Run one epoch
    print("\n▶ Training 1 epoch...")
    start = time.time()

    try:
        metrics = trainer.train_epoch()
        elapsed = time.time() - start

        print(f"✓ Training completed in {elapsed:.2f}s")
        print(f"✓ Average loss: {metrics['train_loss']:.4f}")
        print("✓ Standard Transformer test PASSED")
        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_mamba():
    """Test Geometric Mamba (O(N))."""
    print("\n" + "="*80)
    print("TEST 2: Geometric Mamba (O(N))")
    print("="*80)

    # Config
    config = {
        'field_dim': 4,
        'spatial_size': [8, 8],
        'd_model': 128,  # Small for testing
        'n_heads': 4,
        'n_layers': 2,
        'batch_size': 4,
        'max_epochs': 1,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'vocab_size': 256,
        'max_seq_len': 64,
        'num_workers': 0
    }

    print(f"Device: {config['device']}")
    print(f"Model: d={config['d_model']}, Mamba layers={config['n_layers']}")

    # Create field
    field_config = FieldConfig(
        spatial_size=tuple(config['spatial_size']),
        tensor_dim=config['field_dim'],
        adaptive_learning=True,
        device=config['device']
    )
    field = CognitiveTensorField(field_config)

    # Create Geometric Mamba model
    model = GeometricTransformerWithMamba(
        d_model=config['d_model'],
        n_mamba_layers=config['n_layers'],
        n_attention_layers=1,  # Small for testing
        n_heads=config['n_heads'],
        field_dim=config['field_dim'],
        use_dpr=False,  # Disable DPR for testing
        use_positional_encoding=True,
        use_temporal_encoding=True
    )

    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset
    tokenizer = CognitiveTokenizer(vocab_size=config['vocab_size'])

    data_path = Path('./data/sample/train.txt')
    if not data_path.exists():
        print(f"Creating sample data at {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, 'w') as f:
            f.write("The cognitive field evolves according to differential equations.\n")
            f.write("Geometric attention uses wedge and tensor products for reasoning.\n")

    dataset = TextDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        max_length=config['max_seq_len']
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,          # Parallel data loading (4 CPU workers)
        pin_memory=True,        # Faster CPU→GPU transfer
        prefetch_factor=4,      # Each worker prefetches 4 batches
        persistent_workers=True # Keep workers alive between epochs
    )

    print(f"✓ Dataset: {len(dataset)} examples")

    # Create optimizer
    from Liorhybrid.training.biquat_optimizer import BiquatOptimizer
    optimizer = BiquatOptimizer(model.parameters(), lr=config['lr'])

    # Create trainer
    trainer_config = {
        'training_mode': 'geometric',
        'max_epochs': 1,
        'grad_accum_steps': 1,
        'use_amp': False,
        'clip_grad_norm': 1.0,
        'log_interval': 1,
        'eval_interval': 1000,
        'save_interval': 1000,
        'output_dir': './checkpoints/test_mamba',
        'vocab_size': config['vocab_size'],
        'max_seq_len': config['max_seq_len']
    }

    trainer = CognitiveTrainer(
        model=model,
        field=field,
        train_loader=loader,
        val_loader=None,
        optimizer=optimizer,
        device=config['device'],
        config=trainer_config,
        tokenizer=tokenizer  # For inference
    )

    # Run one epoch
    print("\n▶ Training 1 epoch...")
    start = time.time()

    try:
        metrics = trainer.train_epoch()
        elapsed = time.time() - start

        print(f"✓ Training completed in {elapsed:.2f}s")
        print(f"✓ Average loss: {metrics['train_loss']:.4f}")
        print("✓ Geometric Mamba test PASSED")
        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BAYESIAN COGNITIVE FIELD - TRAINING TESTS")
    print("="*80)

    results = {}

    # Test 1: Standard Transformer
    results['standard'] = test_standard_transformer()

    # Test 2: Geometric Mamba
    results['mamba'] = test_geometric_mamba()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Standard Transformer (O(N²)): {'PASSED ✓' if results['standard'] else 'FAILED ✗'}")
    print(f"Geometric Mamba (O(N)):       {'PASSED ✓' if results['mamba'] else 'FAILED ✗'}")
    print("="*80)

    if all(results.values()):
        print("\n✓ ALL TESTS PASSED - System is working!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - See errors above")
        return 1


if __name__ == "__main__":
    exit(main())
