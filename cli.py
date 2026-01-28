#!/usr/bin/env python3
"""
CLI wrapper for Liorhybrid training and inference.

Usage:
    liorhybrid train --config configs/train_geometric.yaml
    liorhybrid inference --checkpoint checkpoints/model.pt
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parent
if str(repo_root.parent) not in sys.path:
    sys.path.insert(0, str(repo_root.parent))


def train_entrypoint():
    """Training entrypoint with absolute imports.
    
    For full training with the interactive interface, use:
        python -m Liorhybrid.main
    
    This CLI provides a simplified training interface.
    """
    parser = argparse.ArgumentParser(
        description='Train a Liorhybrid model',
        epilog='For full training options, use: python -m Liorhybrid.main'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config YAML file (optional)')
    parser.add_argument('--data', type=str, default='data/sample/train.txt',
                        help='Path to training data (default: data/sample/train.txt)')
    parser.add_argument('--val-data', type=str, default='data/sample/val.txt',
                        help='Path to validation data (default: data/sample/val.txt)')
    parser.add_argument('--output', type=str, default='./checkpoints',
                        help='Output directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    
    args = parser.parse_args()
    
    # Import after arg parsing to avoid slow imports for --help
    import torch
    from Liorhybrid.training.trainer2 import TrainConfig
    
    print("=" * 60)
    print("Liorhybrid Training CLI")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Validation: {args.val_data}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    print()
    print("Note: For full interactive training with all options, use:")
    print("    python -m Liorhybrid.main")
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for training but not available.")
        print("Please install a CUDA-enabled PyTorch build.")
        sys.exit(1)
    
    # Check data files exist
    if not Path(args.data).exists():
        print(f"ERROR: Training data not found: {args.data}")
        print()
        print("To generate sample data, run:")
        print("    python -m Liorhybrid.data.sample.generate_sample_data")
        sys.exit(1)
    
    # NOTE: trainer2 requires complex setup with hooks, model, field, memory, etc.
    # For now, direct users to main.py which has the full infrastructure.
    print("ERROR: Direct CLI training is not yet fully implemented.")
    print()
    print("Please use the interactive training interface:")
    print("    python -m Liorhybrid.main")
    print()
    print("Or use the programmatic API. See QUICK_START.md for examples.")
    sys.exit(1)


def inference_entrypoint():
    """Inference entrypoint with absolute imports."""
    parser = argparse.ArgumentParser(description='Run inference with a trained Liorhybrid model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt for generation (if not provided, enters interactive mode)')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, defaults to auto-detect)')
    
    args = parser.parse_args()
    
    # Import after arg parsing to avoid slow imports for --help
    from Liorhybrid.inference.inference import InferenceEngine
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Initialize inference engine
    print(f"Loading model from: {checkpoint_path}")
    import torch
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        engine = InferenceEngine(str(checkpoint_path), device=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Run inference
    if args.prompt:
        # Single generation
        print(f"\nPrompt: {args.prompt}")
        response = engine.generate(
            args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"Response: {response}")
    else:
        # Interactive chat
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.")
        engine.chat()


def main():
    """Main CLI entrypoint."""
    # Handle no arguments
    if len(sys.argv) < 2:
        parser = argparse.ArgumentParser(
            description='Liorhybrid: Physics-Based AI',
            usage='liorhybrid <command> [options]'
        )
        parser.add_argument('command', choices=['train', 'inference'],
                            help='Command to run (train or inference)')
        parser.print_help()
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Remove the program name and command from sys.argv for subcommand parsing
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'train':
        train_entrypoint()
    elif command == 'inference':
        inference_entrypoint()
    else:
        parser = argparse.ArgumentParser(
            description='Liorhybrid: Physics-Based AI',
            usage='liorhybrid <command> [options]'
        )
        parser.add_argument('command', choices=['train', 'inference'],
                            help='Command to run (train or inference)')
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
