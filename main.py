"""
Bayesian Cognitive Field - Interactive Training Interface

Run without arguments for interactive mode:
    python main.py

Or use command line arguments for automation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
import sys
from pathlib import Path

# Allow running as a script from inside the repo.
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent
    parent_dir = repo_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.inference import (
    GeometricTransformer,
    GeometricTransformerWithMamba
)
from Liorhybrid.training import (
    CognitiveTokenizer,
    TextDataset,
    ChunkedTextDataset,
    ImageTextDataset,
    VideoTextDataset,
    CognitiveTrainer,
    open_file_dialog,
    open_multiple_files_dialog,
    UniversalFileReader,
)


def count_parameters(model):
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total, trainable, frozen)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def print_parameter_summary(model, prefix=""):
    """Print formatted parameter summary."""
    total, trainable, frozen = count_parameters(model)
    print(f"{prefix}Total parameters: {total:,}")
    print(f"{prefix}Trainable: {trainable:,}")
    if frozen > 0:
        print(f"{prefix}Frozen: {frozen:,}")


def run_preflight_checklist_or_die(model, field, tokenizer, config):
    """
    Preflight checklist gate to ensure:
    - all required trainables exist before optimizer
    - everything lives on the target device
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("preflight", __file__)

    target_device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Model/device invariants
    param_devices = {p.device for p in model.parameters()}
    if len(param_devices) != 1:
        raise RuntimeError(f"Model parameters must be on a single device, found: {sorted({str(d) for d in param_devices})}")

    param_device = next(iter(param_devices))
    if param_device.type != target_device.type:
        raise RuntimeError(f"Model parameters must be on {target_device}, found: {sorted({str(d) for d in param_devices})}")

    # Allow "cuda" (no index) to match any specific CUDA device like "cuda:0".
    if target_device.type == "cuda" and target_device.index is None:
        pass
    elif param_device != target_device:
        raise RuntimeError(f"Model parameters must be on {target_device}, found: {sorted({str(d) for d in param_devices})}")

    if not hasattr(model, "lm_head") or model.lm_head is None:
        raise RuntimeError("Preflight failed: model.lm_head is missing (must exist before optimizer).")

    if tokenizer is not None:
        if not hasattr(model, "input_embedding") or model.input_embedding is None:
            raise RuntimeError("Preflight failed: model.input_embedding is missing (must exist before optimizer).")


def interactive_menu():
    """Interactive menu for training configuration."""

    print("=" * 70)
    print("  BAYESIAN COGNITIVE FIELD - Training System")
    print("  Advanced Physics-Based Multimodal AI")
    print("=" * 70)

    # Main menu
    print("┌─ MAIN MENU ─────────────────────────────────────────────────┐")
    print("│                                                             │")
    print("│  1. Quick Start (Geometric Training - Recommended)          │")
    print("│  2. Full Training (Train Everything End-to-End)             │")
    print("│  3. Resume from Checkpoint                                  │")
    print("│  4. Generate Sample Dataset                                 │")
    print("│  5. Inference/Chat Mode                                     │")
    print("│  6. Inspect Checkpoint                                      │")
    print("│  7. Evaluate Checkpoint (Run Validation)                    │")
    print("│  8. Config Cost Calculator                                  │")
    print("│  9. Exit                                                    │")
    print("│                                                             │")
    print("└─────────────────────────────────────────────────────────────┘")

    choice = input("\n▶ Select option [1-9]: ").strip()

    if choice == '1':
        return configure_geometric_training()
    elif choice == '2':
        return configure_full_training()
    elif choice == '3':
        return configure_resume_training()
    elif choice == '4':
        generate_sample_data()
        return interactive_menu()
    elif choice == '5':
        start_inference_mode()
        return interactive_menu()
    elif choice == '6':
        inspect_checkpoint_menu()
        return interactive_menu()
    elif choice == '7':
        evaluate_checkpoint_menu()
        return interactive_menu()
    elif choice == '8':
        config_cost_calculator_menu()
        return interactive_menu()
    elif choice == '9':
        print("\nExiting. Goodbye!")
        return None  # Return None instead of sys.exit to allow clean exit
    else:
        print("\n✗ Invalid choice. Please try again.")
        return interactive_menu()


def config_cost_calculator_menu():
    """Interactive one-shot config cost calculator (params/memory/compute)."""
    from Liorhybrid.utils.cost_estimator import print_estimate

    while True:
        print("\n" + "=" * 70)
        print("  CONFIG COST CALCULATOR")
        print("=" * 70)
        print("\nPARAM IMPACT GUIDE:")
        print("  d_model:    MAJOR - 256=~5M, 512=~20M, 1024=~80M total params")
        print("  n_layers:   MAJOR - BiQuatCausal blocks, linear scaling")
        print("  vocab_size: MAJOR - embedding=V*d, LM_head=d*V+V")
        print("  n_attn:     MODERATE - GeometricAttention layers")
        print("  max_seq:    MEMORY - positional embed, activations")
        print("  batch_size: MEMORY - linear GPU memory scaling")
        print("  spatial:    MEMORY - field grid, 16^2 complex per point")
        print()

        def _ask_int(prompt: str, default: int) -> int:
            raw = input(f"{prompt} [{default}]: ").strip()
            return int(raw) if raw else int(default)

        d_model = _ask_int("d_model", 256)
        n_layers = _ask_int("n_layers (BiQuatCausal blocks)", 4)
        n_attention_layers = _ask_int("n_attention_layers (GeomAttn)", 2)
        vocab_size = _ask_int("vocab_size", 32000)
        max_seq_len = _ask_int("max_seq_len", 512)
        batch_size = _ask_int("batch_size", 64)
        spatial_x = _ask_int("spatial_size x", 8)
        spatial_y = _ask_int("spatial_size y", 8)

        dtype = input("param dtype [fp32/fp16/bf16] (fp32): ").strip() or "fp32"
        optimizer = input("optimizer [biquat/adamw] (biquat): ").strip() or "biquat"

        cfg = {
            "use_causal_field": True,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_attention_layers": n_attention_layers,
            "n_heads": 4,           # FIXED - use_attention=False in BiQuatCausalBlock
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "field_dim": 16,        # FIXED - d_field hardcoded in GeometricStack
            "spatial_size": (spatial_x, spatial_y),
            "dtype": dtype,
            "optimizer": optimizer,
        }

        print_estimate(cfg)

        print("\n  1) Another calculation")
        print("  2) Return to main menu")
        nxt = input("\n▶ Choice [1-2]: ").strip()
        if nxt != "1":
            return


def configure_geometric_training():
    """Configure geometric-only training."""
    print("\n" + "=" * 70)
    print("  GEOMETRIC TRAINING MODE")
    print("  (Trains: Geometric weights + Field parameters)")
    print("  (Freezes: Embeddings)")
    print("=" * 70)

    config = {'mode': 'geometric'}

    # Data path
    print("┌─ DATASET ────────────────────────────────────────────────────┐")
    print("│  Select your training data                                   │")
    print("│  [1] Single file (browse)                                    │")
    print("│  [2] Multiple files (add to dataset)                         │")
    print("│  [3] MNIST (standard benchmark)                              │")
    print("│  [4] Generate sample text data                               │")
    print("└──────────────────────────────────────────────────────────────┘")

    choice = input("\n▶ Choice [1-4]: ").strip()

    if choice == '2':
        # Multi-file upload - offer GUI or manual
        print("\n▶ How would you like to add files?")
        print("  [1] GUI multi-select (fast, one directory)")
        print("  [2] Add files manually (flexible, paste paths)")

        multi_choice = input("\n▶ Choice [1-2]: ").strip()

        data_paths = []

        if multi_choice == '1':
            # GUI multi-select
            selected = open_multiple_files_dialog("Select Training Data Files")
            if selected:
                data_paths.extend(selected)
                print(f"\nSelected {len(selected)} files")

                # Option to add more from another directory
                while True:
                    add_more = input("\n▶ Add more files from another directory? [y/N]: ").strip().lower()
                    if add_more == 'y':
                        more_files = open_multiple_files_dialog("Select Additional Training Data Files")
                        if more_files:
                            data_paths.extend(more_files)
                            print(f"✓ Added {len(more_files)} more files (total: {len(data_paths)})")
                        else:
                            print("✗ No files selected")
                    else:
                        break

        else:
            # Manual adding (original functionality)
            print("\n▶ Add multiple files to your dataset")
            while True:
                print(f"\n   Current files: {len(data_paths)}")
                print("   [1] Browse for file")
                print("   [2] Enter path manually")
                print("   [3] Done adding files")

                file_choice = input("\n▶ Choice [1-3]: ").strip()

                if file_choice == '3':
                    break
                elif file_choice == '1':
                    file_path = open_file_dialog(f"Select Training Data File #{len(data_paths)+1}")
                    if file_path and Path(file_path).exists():
                        data_paths.append(file_path)
                        print(f"✓ Added: {file_path}")
                elif file_choice == '2':
                    file_path = input("▶ File path: ").strip()
                    if file_path and Path(file_path).exists():
                        data_paths.append(file_path)
                        print(f"✓ Added: {file_path}")
                    else:
                        print(f"✗ File not found: {file_path}")

        if not data_paths:
            print("\n✗ No files selected. Using sample data instead.")
            data_path = generate_sample_data()
            config['data_path'] = data_path
        else:
            print(f"\n✓ Total files: {len(data_paths)}")
            for path in data_paths:
                print(f"  - {Path(path).name}")
            config['data_paths'] = data_paths

        config['data_type'] = 'text'

    elif choice == '3':
        # MNIST dataset
        config['data_type'] = 'mnist'
        config['data_path'] = './data/mnist'
        print("✓ Using MNIST dataset")

    elif choice == '4':
        data_path = generate_sample_data()
        config['data_path'] = data_path
        config['data_type'] = 'text'

    else:
        # Single file
        print("\n▶ Opening file picker...")
        data_path = open_file_dialog("Select Training Data")

        if not data_path:
            print("✗ No file selected. Using sample data instead.")
            data_path = generate_sample_data()
        else:
            print(f"✓ Selected: {data_path}")

        config['data_path'] = data_path
        config['data_type'] = 'text'

    # Architecture choice
    print("┌─ ARCHITECTURE ───────────────────────────────────────────────┐")
    print("│  1. Standard Transformer (O(N^2) attention)                  │")
    print("│  2. Causal Field (O(N log N) parallel) - RECOMMENDED         │")
    print("└──────────────────────────────────────────────────────────────┘")

    arch_choice = input("▶ Select architecture [1-2]: ").strip()
    config['use_causal_field'] = (arch_choice == '2')

    # Quick or custom settings
    print("┌─ CONFIGURATION ──────────────────────────────────────────────┐")
    print("│  1. Quick (Use defaults - good for testing)                  │")
    print("│  2. Custom (Configure model size & training)                 │")
    print("└──────────────────────────────────────────────────────────────┘")

    mode_choice = input("▶ Select [1-2]: ").strip()

    if mode_choice == '2':
        config.update(get_custom_config())
    else:
        # Interactive quick config with param impact guide
        print("\n" + "=" * 70)
        print("  MODEL CONFIGURATION")
        print("=" * 70)
        print("\nPARAM IMPACT GUIDE (from actual model code):")
        print("  d_model:    MAJOR - 256=~5M, 512=~20M, 1024=~80M total params")
        print("              Each layer adds ~12*d^2 params (FFN=8/3*d + projections)")
        print("  n_layers:   MAJOR - BiQuatCausal blocks, linear scaling")
        print("  vocab_size: MAJOR - embedding=V*d, LM_head=d*V+V")
        print("  n_attn:     MODERATE - GeometricAttention O(N^2) layers")
        print("  max_seq:    MEMORY - positional embed T*d, activations O(B*T*d)")
        print("  batch_size: MEMORY/THROUGHPUT - linear GPU memory scaling")
        print("  spatial:    MEMORY - field grid, 16^2 complex per grid point")
        print()

        # Only REAL configurable params
        try:
            d_model = int(input("d_model [256/512/1024] (256): ").strip() or "256")
            n_layers = int(input("n_layers [2/4/8] (4): ").strip() or "4")
            n_attention_layers = int(input("n_attention_layers [1/2/4] (2): ").strip() or "2")
            vocab_size = int(input("vocab_size [8000/16000/32000] (32000): ").strip() or "32000")
            max_seq_len = int(input("max_seq_len [256/512/1024] (512): ").strip() or "512")
            batch_size = int(input("batch_size [16/32/64/128] (64): ").strip() or "64")
            spatial_x = int(input("spatial_size X [4/8/16] (8): ").strip() or "8")
            spatial_y = int(input("spatial_size Y [4/8/16] (8): ").strip() or "8")
            max_epochs = int(input("max_epochs [5/10/20] (5): ").strip() or "5")
            lr = float(input("learning_rate [0.0001/0.0003/0.001] (0.0001): ").strip() or "0.0001")
            dropout = float(input("dropout [0.0/0.1/0.2] (0.0): ").strip() or "0.0")
        except ValueError:
            print("Invalid input, using defaults")
            d_model, n_layers, n_attention_layers = 256, 4, 2
            vocab_size, max_seq_len, batch_size = 32000, 512, 64
            spatial_x, spatial_y = 8, 8
            max_epochs, lr, dropout = 5, 0.0001, 0.0

        config.update({
            'd_model': d_model,
            'n_layers': n_layers,
            'n_mamba_layers': n_layers,  # For backwards compatibility
            'n_attention_layers': n_attention_layers,
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'spatial_size': [spatial_x, spatial_y],
            'max_epochs': max_epochs,
            'lr': lr,
            'dropout': dropout,
            # FIXED VALUES (not user configurable - hardcoded in model)
            'n_heads': 4,
            'field_dim': 16,
            'adaptive_field': True,
            'output_dir': './checkpoints/geometric',
            'log_interval': 10,
        })

        # Show estimated cost
        try:
            from Liorhybrid.utils.cost_estimator import estimate_cost, format_bytes
            est = estimate_cost(config)
            print("\n" + "-" * 40)
            print("ESTIMATED COST:")
            print(f"  Total params:     {est.total_params:,}")
            print(f"  Memory required:  {format_bytes(est.total_bytes_est)}")
            print("-" * 40)
            proceed = input("\nProceed with training? [Y/n]: ").strip().lower()
            if proceed == 'n':
                return None
        except Exception as e:
            print(f"Cost estimation failed: {e}")

    return config


def configure_full_training():
    """Configure full end-to-end training."""
    print("\n" + "=" * 70)
    print("  FULL TRAINING MODE")
    print("  (Trains: Everything - Embeddings + Geometric + Transformer)")
    print("=" * 70)

    config = {'mode': 'full'}

    # Data type
    print("┌─ DATA TYPE ──────────────────────────────────────────────────┐")
    print("│  1. Text only                                                 │")
    print("│  2. Image + Text (multimodal)                                │")
    print("│  3. Video + Text (multimodal)                                │")
    print("└──────────────────────────────────────────────────────────────┘")

    data_type_choice = input("\n▶ Select data type [1-3]: ").strip()

    data_type_map = {'1': 'text', '2': 'image-text', '3': 'video-text'}
    config['data_type'] = data_type_map.get(data_type_choice, 'text')

    # Data path
    print("\n┌─ DATASET ────────────────────────────────────────────────────┐")
    print("│  Select your training data                                   │")
    print("│  Supports: PDF, DOCX, TXT, MD, PY, CPP, JSON, CSV, etc.      │")
    print("│  [1] Single file (browse)                                    │")
    print("│  [2] Multiple files (add to dataset)                         │")
    print("│  [3] Generate sample data                                    │")
    print("└──────────────────────────────────────────────────────────────┘")

    choice = input("\n▶ Choice [1-3]: ").strip()

    if choice == '2':
        # Multi-file upload - offer GUI or manual
        print("\n▶ How would you like to add files?")
        print("  [1] GUI multi-select (fast, one directory)")
        print("  [2] Add files manually (flexible, paste paths)")

        multi_choice = input("\n▶ Choice [1-2]: ").strip()

        data_paths = []

        if multi_choice == '1':
            # GUI multi-select
            selected = open_multiple_files_dialog("Select Training Data Files")
            if selected:
                data_paths.extend(selected)
                print(f"\nSelected {len(selected)} files")

                # Option to add more from another directory
                while True:
                    add_more = input("\n▶ Add more files from another directory? [y/N]: ").strip().lower()
                    if add_more == 'y':
                        more_files = open_multiple_files_dialog("Select Additional Training Data Files")
                        if more_files:
                            data_paths.extend(more_files)
                            print(f"✓ Added {len(more_files)} more files (total: {len(data_paths)})")
                        else:
                            print("✗ No files selected")
                    else:
                        break

        else:
            # Manual adding
            print("\n▶ Add multiple files to your dataset")
            while True:
                print(f"\n   Current files: {len(data_paths)}")
                print("   [1] Browse for file")
                print("   [2] Enter path manually")
                print("   [3] Done adding files")

                file_choice = input("\n▶ Choice [1-3]: ").strip()

                if file_choice == '3':
                    break
                elif file_choice == '1':
                    file_path = open_file_dialog(f"Select Training Data File #{len(data_paths)+1}")
                    if file_path and Path(file_path).exists():
                        data_paths.append(file_path)
                        print(f"✓ Added: {file_path}")
                elif file_choice == '2':
                    file_path = input("▶ File path: ").strip()
                    if file_path and Path(file_path).exists():
                        data_paths.append(file_path)
                        print(f"✓ Added: {file_path}")
                    else:
                        print(f"✗ File not found: {file_path}")

        if not data_paths:
            print("\n✗ No files selected. Using sample data instead.")
            data_path = generate_sample_data()
            config['data_path'] = data_path
        else:
            print(f"\n✓ Total files: {len(data_paths)}")
            for path in data_paths:
                print(f"  - {Path(path).name}")
            config['data_paths'] = data_paths

    elif choice == '3':
        data_path = generate_sample_data()
        config['data_path'] = data_path

    else:
        # Single file
        print("\n▶ Opening file picker...")
        data_path = open_file_dialog("Select Training Data")

        if not data_path:
            print("✗ No file selected. Using sample data instead.")
            data_path = generate_sample_data()
        else:
            print(f"✓ Selected: {data_path}")

        config['data_path'] = data_path

    # Defaults
    d_model, n_layers, n_attention_layers = 256, 4, 2
    vocab_size, max_seq_len, batch_size = 32000, 512, 64
    field_dim, spatial_x, spatial_y = 16, 8, 8
    lr, dropout = 0.0003, 0.0

    # Configuration subpaths
    print("\n" + "=" * 70)
    print("  CONFIGURATION OPTIONS")
    print("=" * 70)
    print("\n  1. Training/Performance (safe)")
    print("  2. Physics (changes model, won't break)")
    print("  3. Architecture (WARNING: can break if wrong)")
    print("  4. Skip - use all defaults")

    choice = input("\nSelect [1-4]: ").strip()

    if choice == '1':
        # Training/Perf - safe params
        print("\n-- Training/Performance --")
        try:
            batch_size = int(input("batch_size (64): ").strip() or "64")
            max_seq_len = int(input("max_seq_len (512): ").strip() or "512")
            lr = float(input("learning_rate (0.0003): ").strip() or "0.0003")
            dropout = float(input("dropout (0.0): ").strip() or "0.0")
        except ValueError:
            print("Invalid, using defaults")

    elif choice == '2':
        # Physics - significant but safe
        print("\n-- Physics Parameters --")
        try:
            d_model = int(input("d_model (256): ").strip() or "256")
            n_layers = int(input("n_layers (4): ").strip() or "4")
            n_attention_layers = int(input("n_attention_layers (2): ").strip() or "2")
            field_dim = int(input("field_dim (16): ").strip() or "16")
            spatial_x = int(input("spatial_size X (8): ").strip() or "8")
            spatial_y = int(input("spatial_size Y (8): ").strip() or "8")
        except ValueError:
            print("Invalid, using defaults")

    elif choice == '3':
        # Breaking params - needs warning
        print("\n" + "!" * 50)
        print("  WARNING: These params can break training if wrong!")
        print("  - vocab_size must match tokenizer")
        print("  - Mismatched sizes cause shape errors")
        print("!" * 50)
        confirm = input("\nContinue? [y/N]: ").strip().lower()
        if confirm == 'y':
            try:
                vocab_size = int(input("vocab_size (32000): ").strip() or "32000")
                d_model = int(input("d_model (256): ").strip() or "256")
                n_layers = int(input("n_layers (4): ").strip() or "4")
            except ValueError:
                print("Invalid, using defaults")

    config.update({
        'd_model': d_model,
        'n_layers': n_layers,
        'n_mamba_layers': n_layers,
        'n_attention_layers': n_attention_layers,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'field_dim': field_dim,
        'spatial_size': [spatial_x, spatial_y],
        'lr': lr,
        'dropout': dropout,
    })

    # Architecture choice
    print("\n┌─ ARCHITECTURE ───────────────────────────────────────────────┐")
    print("│  1. Standard Transformer (O(N^2) attention)                  │")
    print("│  2. Causal Field (O(N log N) parallel) - RECOMMENDED         │")
    print("└──────────────────────────────────────────────────────────────┘")

    arch_choice = input("\n▶ Select architecture [1-2]: ").strip()
    config['use_causal_field'] = (arch_choice == '2')

    # Training epochs override
    default_epochs = 10
    epochs_input = input(f"\n▶ Max epochs [{default_epochs}]: ").strip()
    max_epochs = int(epochs_input) if epochs_input else default_epochs

    # Timing debug toggle
    timing_input = input("\n▶ Enable timing debug? [y/N]: ").strip().lower()
    timing_debug = (timing_input == 'y')

    # NaN diagnostic toggle
    diagnose_input = input("▶ Enable NaN diagnostics? [y/N]: ").strip().lower()
    diagnose_nan = (diagnose_input == 'y')

    # Step progress logging (within each window)
    step_progress_input = input("▶ Log progress every N steps within window (0=off) [50]: ").strip()
    step_progress_every = int(step_progress_input) if step_progress_input else 50

    # Training
    config.update({
        'max_epochs': max_epochs,
        'adaptive_field': True,
        'output_dir': './checkpoints/full',
        'log_interval': 1,  # Log every step
        'timing_debug': timing_debug,
        'diagnose_nan': diagnose_nan,
        'step_progress_every': step_progress_every
    })

    return config


def configure_resume_training():
    """Configure resuming from checkpoint."""
    print("\n┌─ RESUME TRAINING ────────────────────────────────────────────┐")
    print("│  Select checkpoint to resume from                            │")
    print("└──────────────────────────────────────────────────────────────┘")

    print("\n  [1] Browse with GUI")
    print("  [2] Enter path manually")
    print("  [3] Use latest checkpoint (./checkpoints/best_model.pt)")

    choice = input("\n▶ Choice [1-3]: ").strip()

    if choice == '1':
        print("\n▶ Opening file picker...")
        checkpoint_path = open_file_dialog("Select Checkpoint (.pt)")
        if not checkpoint_path:
            print("\n✗ No checkpoint selected.")
            return interactive_menu()
    elif choice == '3':
        checkpoint_path = './checkpoints/best_model.pt'
        if not Path(checkpoint_path).exists():
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
            print("   Train a model first to create a checkpoint.")
            return interactive_menu()
    else:
        checkpoint_path = input("\n▶ Checkpoint path: ").strip()

    if not Path(checkpoint_path).exists():
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        return interactive_menu()

    print(f"✓ Selected checkpoint: {checkpoint_path}")

    # Ask for training data
    print("\n▶ Select training data:")
    print("  [1] Single file (browse)")
    print("  [2] Multiple files (add to dataset)")
    print("  [3] Use sample data")
    print("  [4] Use default (./data/train.txt)")

    data_choice = input("\n▶ Choice [1-4]: ").strip()

    data_paths = []

    if data_choice == '1':
        print("\n▶ Opening file picker...")
        data_path = open_file_dialog("Select Training Data")
        if not data_path:
            print("✗ No file selected. Using default.")
            data_paths = ['./data/train.txt']
        else:
            print(f"✓ Selected: {data_path}")
            data_paths = [data_path]

    elif data_choice == '2':
        # Multi-file upload with GUI
        print("\n▶ Select multiple training files (hold Ctrl/Cmd to select multiple)")
        selected_paths = open_multiple_files_dialog("Select Training Data Files")

        if not selected_paths:
            print("\n✗ No files selected. Using default.")
            data_paths = ['./data/train.txt']
        else:
            data_paths = list(selected_paths)
            print(f"\n✓ Selected {len(data_paths)} files:")
            for path in data_paths:
                print(f"  - {Path(path).name}")

    elif data_choice == '3':
        data_paths = [generate_sample_data()]
    else:
        data_paths = ['./data/train.txt']

    # If multiple files, we'll merge them later
    # For now, pass the list of paths
    if len(data_paths) == 1:
        return {'resume': checkpoint_path, 'mode': 'full', 'data_path': data_paths[0]}
    else:
        return {'resume': checkpoint_path, 'mode': 'full', 'data_paths': data_paths}


def start_inference_mode():
    """Launch inference/chat mode."""
    from Liorhybrid.inference.inference import InferenceEngine, load_checkpoint_with_gui

    print("")
    print("INFERENCE MODE")
    print("=" * 70)
    print("Select checkpoint to load for inference")
    print("")
    print("  [1] Browse with GUI")
    print("  [2] Enter path manually")

    while True:
        choice = input("\nChoice [1-2]: ").strip()
        if choice in ("1", "2"):
            break
        print("Invalid choice. Please try again.")

    if choice == "1":
        checkpoint_path = load_checkpoint_with_gui()
        if not checkpoint_path:
            print("")
            print("No checkpoint selected.")
            return
    else:
        checkpoint_path = input("\nCheckpoint path: ").strip()

    if not checkpoint_path:
        print("")
        print("No checkpoint path provided.")
        return

    if not Path(checkpoint_path).exists():
        print("")
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    try:
        engine = InferenceEngine(checkpoint_path)
        engine.chat()
    except Exception as e:
        print("")
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()


def evaluate_checkpoint_menu():
    """Evaluate a checkpoint on validation data."""
    import math
    from Liorhybrid.training import open_file_dialog

    print("\n┌─ EVALUATE CHECKPOINT ────────────────────────────────────────┐")
    print("│  Run validation on a saved checkpoint                        │")
    print("└──────────────────────────────────────────────────────────────┘")

    # Select checkpoint
    print("\n  [1] Browse for checkpoint")
    print("  [2] Enter path manually")
    print("  [3] Use latest (./checkpoints/full/)")

    choice = input("\n▶ Choice [1-3]: ").strip()

    if choice == '1':
        checkpoint_path = open_file_dialog("Select Checkpoint (.pt)")
        if not checkpoint_path:
            print("\n✗ No checkpoint selected.")
            return
    elif choice == '3':
        # Find latest checkpoint in default dir
        checkpoint_dir = Path('./checkpoints/full')
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.pt'))
            if checkpoints:
                checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
                print(f"✓ Found: {checkpoint_path}")
            else:
                print("\n✗ No checkpoints found in ./checkpoints/full/")
                return
        else:
            print("\n✗ Directory ./checkpoints/full/ does not exist")
            return
    else:
        checkpoint_path = input("\n▶ Checkpoint path: ").strip()

    if not Path(checkpoint_path).exists():
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        return

    # Select validation data
    print("\n" + "="*70)
    print("VALIDATION DATA")
    print("="*70)

    print("\n  [1] Single file (browse with GUI)")
    print("  [2] Multiple files (add to dataset)")
    print("  [3] Enter path manually")

    data_choice = input("\nChoice [1-3]: ").strip()

    data_paths = []

    if data_choice == '2':
        # Multiple files mode
        print("\nHow would you like to add files?")
        print("  [1] GUI multi-select (Ctrl/Shift to select multiple)")
        print("  [2] Add files manually (flexible, paste paths)")

        multi_choice = input("\nChoice [1-2]: ").strip()

        if multi_choice == '1':
            selected = open_multiple_files_dialog("Select Validation Data Files")
            if selected:
                data_paths.extend(selected)
                print(f"\nSelected {len(selected)} file(s)")
            else:
                print("\nNo files selected.")
                return
        else:
            # Manual mode with loop
            while True:
                print("\nAdd a file:")
                print("  [1] Browse with GUI")
                print("  [2] Paste file path")
                print("  [3] Done adding files")

                file_choice = input("\nChoice [1-3]: ").strip()

                if file_choice == '1':
                    file_path = open_file_dialog("Select Validation Data File")
                    if file_path:
                        data_paths.append(file_path)
                        print(f"   Added: {Path(file_path).name}")
                elif file_choice == '2':
                    file_path = input("\nFile path: ").strip()
                    if Path(file_path).exists():
                        data_paths.append(file_path)
                        print(f"   Added: {Path(file_path).name}")
                    else:
                        print(f"   File not found: {file_path}")
                else:
                    break

                if not data_paths:
                    print("\nNo files added.")
                    return

    elif data_choice == '1':
        # Single file with GUI
        data_path = open_file_dialog("Select Validation Data")
        if not data_path:
            print("\nNo data file selected.")
            return
        data_paths = [data_path]

    else:
        # Manual path entry
        data_path = input("\nValidation data path: ").strip()
        if not Path(data_path).exists():
            print(f"\nData file not found: {data_path}")
            return
        data_paths = [data_path]

    print(f"\nCheckpoint: {checkpoint_path}")
    if len(data_paths) == 1:
        print(f"Validation data: {data_paths[0]}")
    else:
        print(f"Validation data: {len(data_paths)} files")
    print("\nLoading checkpoint and running evaluation...")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        config = checkpoint.get('config', {})

        print(f"   Step: {checkpoint.get('global_step', 'N/A')}")
        print(f"   Train losses recorded: {len(checkpoint.get('train_losses', []))}")

        # Create tokenizer from checkpoint
        if 'tokenizer' in checkpoint:
            tokenizer = CognitiveTokenizer(vocab_size=checkpoint['tokenizer']['vocab_size'])
            tokenizer.vocab = checkpoint['tokenizer']['vocab']
            tokenizer.inverse_vocab = checkpoint['tokenizer']['inverse_vocab']
        else:
            print("   Warning: No tokenizer in checkpoint, creating new one")
            tokenizer = CognitiveTokenizer(vocab_size=config.get('vocab_size', 32000))

        # Load and tokenize validation data
        from Liorhybrid.training import UniversalFileReader
        reader = UniversalFileReader()

        # Handle multiple files
        if len(data_paths) > 1:
            print(f"   Loading {len(data_paths)} validation files...")
            merged_content = []

            for i, data_path in enumerate(data_paths, 1):
                if not Path(data_path).exists():
                    print(f"   Warning: File {i} not found: {data_path}")
                    continue

                print(f"   Loading file {i}/{len(data_paths)}: {Path(data_path).name}")

                try:
                    content = reader.read_file(data_path)
                    merged_content.append(content)
                    print(f"   Loaded {len(content)} characters")
                except Exception as e:
                    print(f"   Error reading file: {e}")

            val_text = '\n\n'.join(merged_content)
            print(f"   Merged {len(merged_content)} files -> {len(val_text)} characters")
        else:
            val_text = reader.read_file(data_paths[0])

        # Create validation dataset
        val_dataset = TextDataset(
            texts=[val_text],
            tokenizer=tokenizer,
            max_length=config.get('max_seq_len', 512),
            stride=config.get('max_seq_len', 512) // 2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )

        print(f"   Validation samples: {len(val_dataset)}")

        # Recreate model
        d_model = config.get('d_model', 256)
        n_layers = config.get('n_layers', 4)
        n_attention_layers = config.get('n_attention_layers', 2)

        model = GeometricTransformerWithMamba(
            d_model=d_model,
            n_mamba_layers=n_layers,
            n_attention_layers=n_attention_layers,
            vocab_size=config.get('vocab_size', 32000),
            max_seq_len=config.get('max_seq_len', 4096)
        )

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Create field
        field_config = FieldConfig(
            tensor_dim=config.get('field_dim', 16),
            spatial_size=tuple(config.get('spatial_size', [8, 8]))
        )
        field = CognitiveTensorField(field_config)
        if 'field_state' in checkpoint:
            field.T = checkpoint['field_state']['T'].to(device)

        # Create LM head if needed
        if 'lm_head_state_dict' in checkpoint:
            lm_head = nn.Linear(d_model, config.get('vocab_size', 32000)).to(device)
            lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
        else:
            lm_head = nn.Linear(d_model, config.get('vocab_size', 32000)).to(device)

        # Create embedding
        from Liorhybrid.training.embeddings import MultimodalEmbedding
        input_embedding = MultimodalEmbedding(
            vocab_size=config.get('vocab_size', 32000),
            d_model=d_model,
            max_seq_len=config.get('max_seq_len', 512)
        ).to(device)

        if 'input_embedding_state_dict' in checkpoint:
            input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])

        # Run evaluation
        print("\n" + "="*60)
        print("  RUNNING VALIDATION")
        print("="*60)

        val_losses = []
        from tqdm import tqdm

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                Q_input = input_embedding(batch['input_ids'], modality='text')
                output, _ = model(Q_input, field.T, time=field.t)
                logits = lm_head(output)

                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['input_ids'][..., 1:].contiguous()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0
                )
                val_losses.append(loss.item())

        # Print results
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        perplexity = math.exp(min(avg_loss, 20))

        print("\n" + "="*60)
        print("  EVALUATION RESULTS")
        print("="*60)
        print(f"  Validation Loss:       {avg_loss:.4f}")
        print(f"  Validation Perplexity: {perplexity:.2f}")
        print(f"  Samples evaluated:     {len(val_losses) * config.get('batch_size', 32)}")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def inspect_checkpoint_menu():
    """Inspect checkpoint statistics."""
    from Liorhybrid.training.checkpoint_utils import (
        print_checkpoint_summary,
        compare_checkpoints,
        find_best_checkpoint
    )
    from Liorhybrid.inference.inference import load_checkpoint_with_gui

    print("\n┌─ CHECKPOINT INSPECTION ──────────────────────────────────────┐")
    print("│  1. Inspect single checkpoint                                │")
    print("│  2. Compare multiple checkpoints                             │")
    print("│  3. Find best checkpoint in directory                        │")
    print("└──────────────────────────────────────────────────────────────┘")

    choice = input("\n▶ Choice [1-3]: ").strip()

    if choice == '1':
        # Single checkpoint inspection
        print("\n  [1] Browse with GUI")
        print("  [2] Enter path manually")
        method = input("\n▶ Choice [1-2]: ").strip()

        if method == '1':
            checkpoint_path = load_checkpoint_with_gui()
            if not checkpoint_path:
                print("\n✗ No checkpoint selected.")
                return
        else:
            checkpoint_path = input("\n▶ Checkpoint path: ").strip()

        if not Path(checkpoint_path).exists():
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
            return

        print_checkpoint_summary(checkpoint_path)

    elif choice == '2':
        # Compare multiple checkpoints
        print("\nEnter checkpoint paths (comma-separated):")
        paths_input = input("▶ Paths: ").strip()
        paths = [p.strip() for p in paths_input.split(',')]

        # Validate paths
        valid_paths = [p for p in paths if Path(p).exists()]
        if not valid_paths:
            print("\n✗ No valid checkpoint paths provided.")
            return

        if len(valid_paths) < len(paths):
            print(f"\n⚠ Warning: {len(paths) - len(valid_paths)} paths not found, skipping.")

        compare_checkpoints(valid_paths)

    elif choice == '3':
        # Find best checkpoint
        checkpoint_dir = input("\n▶ Checkpoint directory: ").strip()

        if not Path(checkpoint_dir).exists():
            print(f"\n✗ Directory not found: {checkpoint_dir}")
            return

        metric = input("▶ Metric to optimize [val_loss]: ").strip() or 'val_loss'

        best_path = find_best_checkpoint(checkpoint_dir, metric)

        if best_path:
            print(f"\nShow details? [y/n]: ", end="")
            if input().strip().lower() == 'y':
                print_checkpoint_summary(best_path)


def get_custom_config():
    """Get custom configuration from user."""
    print("\n┌─ CUSTOM CONFIGURATION ───────────────────────────────────────┐")

    config = {}

    # Model dimensions
    try:
        d_model = int(input("▶ Model dimension (128-2048) [512]: ").strip() or "512")
        n_layers = int(input("▶ CausalField layers (non-attention) (1-12) [4]: ").strip() or "4")
        n_attention_layers = int(input("▶ Attention layers (0-6) [2]: ").strip() or "2")
        n_heads = int(input("▶ Attention heads (4-16) [8]: ").strip() or "8")
        batch_size = int(input("▶ Batch size (4-128) [16]: ").strip() or "16")
        max_epochs = int(input("▶ Training epochs (1-100) [10]: ").strip() or "10")
        max_seq_len = int(input("▶ Max sequence length (128-2048) [512]: ").strip() or "512")
        timing_input = input("▶ Enable timing debug? [y/N]: ").strip().lower()
        timing_debug = (timing_input == 'y')
        diagnose_input = input("▶ Enable NaN diagnostics? [y/N]: ").strip().lower()
        diagnose_nan = (diagnose_input == 'y')

        # BPTT window configuration
        print("\n▶ Memory gradient flow (BPTT window):")
        print("  [0] Fully detached (no BPTT, fastest, lowest memory)")
        print("  [1] 50-step window (moderate BPTT)")
        print("  [2] 100-step window (longer BPTT)")
        print("  [3] Fully attached (full BPTT, slowest, highest memory)")
        bptt_choice = input("  Choice [0-3, default=0]: ").strip() or "0"
        bptt_window_map = {'0': 0, '1': 50, '2': 100, '3': -1}  # -1 = never detach
        bptt_window = bptt_window_map.get(bptt_choice, 0)

        config.update({
            'd_model': d_model,
            'n_layers': n_layers,
            'n_mamba_layers': n_layers,  # For backwards compatibility
            'n_attention_layers': n_attention_layers,
            'n_heads': n_heads,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'max_seq_len': max_seq_len,
            'field_dim': 16,
            'spatial_size': [8, 8],
            'lr': 0.0001,
            'timing_debug': timing_debug,
            'diagnose_nan': diagnose_nan,
            'bptt_window': bptt_window
        })

    except ValueError:
        print("\n✗ Invalid input. Using defaults.")
        config = {
            'd_model': 512,
            'n_layers': 4,
            'n_mamba_layers': 4,
            'n_attention_layers': 2,
            'n_heads': 8,
            'batch_size': 16,
            'max_epochs': 10,
            'max_seq_len': 512,
            'field_dim': 16,
            'spatial_size': [8, 8],
            'lr': 0.0001,
            'timing_debug': False
        }

    return config


def generate_sample_data():
    """Generate sample dataset for testing."""
    print("\n┌─ GENERATING SAMPLE DATA ─────────────────────────────────────┐")

    data_dir = Path('./data/sample')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Training data
    train_data = [
        "The cognitive field evolves according to differential equations.",
        "Geometric attention uses wedge and tensor products for reasoning.",
        "Bayesian updates guide the recursive learning process over time.",
        "Complex tensor fields represent high-dimensional quantum states.",
        "Spinor products capture rotational symmetries in the field topology.",
        "Fractional calculus provides memory effects across temporal scales.",
        "Adaptive parameters learn optimal evolution dynamics automatically.",
        "Multimodal embeddings project different data types into shared space.",
        "Contrastive learning aligns representations across multiple modalities.",
        "Language modeling predicts the next token in the sequence accurately.",
        "Entropy gating controls information flow in the cognitive architecture.",
        "Field collapse mechanisms implement quantum-inspired measurement processes.",
    ]

    train_file = data_dir / 'train.txt'
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_data))

    # Validation data
    val_data = [
        "Validation data for cognitive field training and evaluation.",
        "Testing geometric attention mechanisms on unseen examples.",
    ]

    val_file = data_dir / 'val.txt'
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_data))

    print(f"│  ✓ Created: {train_file}")
    print(f"│  ✓ Created: {val_file}")
    print(f"│  ✓ Ready for training!")
    print("└──────────────────────────────────────────────────────────────┘")

    return str(train_file)


def calculate_optimal_batch_size(d_model, seq_len, gpu_vram_gb=24):
    """
    Calculate optimal batch size based on model dimensions and available GPU VRAM.

    Args:
        d_model: Model dimension
        seq_len: Sequence length
        gpu_vram_gb: Available GPU VRAM in GB (default 24 for RTX 4090)

    Returns:
        Recommended batch size
    """
    # Rough memory estimate per sample (in GB)
    # Formula: (d_model * seq_len * 4 bytes for fp32) * safety_factor
    # Safety factor accounts for optimizer states, gradients, activations
    safety_factor = 4  # BiquatOptimizer has no state; just gradients + activations

    bytes_per_sample = d_model * seq_len * 4 * safety_factor
    gb_per_sample = bytes_per_sample / (1024 ** 3)

    # Reserve 20% VRAM for model weights and overhead
    usable_vram = gpu_vram_gb * 0.8

    # Calculate batch size
    recommended_batch = int(usable_vram / gb_per_sample)

    # Clamp to reasonable range
    recommended_batch = max(4, min(recommended_batch, 512))

    # Round to nearest power of 2 for efficiency
    import math
    recommended_batch = 2 ** int(math.log2(recommended_batch))

    print(f"\n[Batch Size Calculator]")
    print(f"  d_model={d_model}, seq_len={seq_len}, GPU VRAM={gpu_vram_gb}GB")
    print(f"  Estimated memory per sample: {gb_per_sample*1024:.1f}MB")
    print(f"  Recommended batch size: {recommended_batch}")

    return recommended_batch


def configure_trainer2_params(config):
    """
    Configure trainer2-specific parameters BEFORE model/dataset creation.
    This allows these params to influence parameter count and model architecture.

    Returns updated config dict with trainer2 params set.
    """
    print("\n" + "=" * 70)
    print("  TRAINER2 CONFIGURATION")
    print("  (Configure BEFORE model creation to influence parameter count)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # SECTION A: SCALING (safe, affects memory/compute)
    # -------------------------------------------------------------------------
    print("\n[A] SCALING (safe - affects memory/compute)")
    print("-" * 50)

    config['tbptt_window_steps'] = int(input(f"  tbptt_window_steps [64]: ").strip() or config.get('tbptt_window_steps', 64))
    config['trainer2_sdm_capacity'] = int(input(f"  sdm_capacity [2048]: ").strip() or config.get('trainer2_sdm_capacity', 2048))
    config['lr'] = float(input(f"  learning_rate [1e-4]: ").strip() or config.get('lr', 1e-4))
    config['retrieval_beta'] = float(input(f"  retrieval_beta [5.0]: ").strip() or config.get('retrieval_beta', 5.0))

    # -------------------------------------------------------------------------
    # SECTION B: PHYSICS (safe, changes dynamics)
    # -------------------------------------------------------------------------
    print("\n[B] PHYSICS (safe - changes dynamics)")
    print("-" * 50)
    print("  frame_mode: rotor | derived | learned_lowrank")
    config['frame_mode'] = input(f"  frame_mode [rotor]: ").strip() or config.get('frame_mode', 'rotor')
    print("  R_source: constitutive | curvature")
    config['R_source'] = input(f"  R_source [constitutive]: ").strip() or config.get('R_source', 'constitutive')
    print("  rotor_mode: stateful | derived | off")
    config['rotor_mode'] = input(f"  rotor_mode [stateful]: ").strip() or config.get('rotor_mode', 'stateful')
    config['beta_nudge'] = float(input(f"  beta_nudge [1e-3]: ").strip() or config.get('beta_nudge', 1e-3))

    # -------------------------------------------------------------------------
    # SECTION C: CORE ARCHITECTURE (main param count drivers)
    # -------------------------------------------------------------------------
    print("\n[C] CORE ARCHITECTURE (main param count drivers)")
    print("-" * 50)
    print("  RTX 4090 max: d=1024, L=12, V=50k, T=4096, B=32")
    print()

    # d_model: representation capacity
    print("  d_model: embedding dimension")
    print("    256=fast/small | 512=balanced | 1024=high capacity")
    print("    Params: ~O(d^2), Memory: ~O(d), Quality: +expressiveness")
    config['d_model'] = int(input(f"  d_model [512]: ").strip() or config.get('d_model', 512))

    # n_layers: depth / abstraction levels
    print("\n  n_layers: Mamba/field evolution depth")
    print("    4=shallow | 8=medium | 12+=deep abstraction")
    print("    Params: ~O(L*d^2), Memory: ~O(L), Quality: +hierarchical features")
    config['n_layers'] = int(input(f"  n_layers [4]: ").strip() or config.get('n_layers', 4))
    config['n_mamba_layers'] = config['n_layers']

    # n_attention_layers: geometric attention depth
    print("\n  n_attention_layers: geometric attention layers")
    print("    2=light | 4=moderate | 8=heavy cross-position mixing")
    print("    Params: ~O(A*d^2), Memory: ~O(A*seq^2), Quality: +long-range deps")
    config['n_attention_layers'] = int(input(f"  n_attention_layers [2]: ").strip() or config.get('n_attention_layers', 2))

    # n_heads: parallel attention patterns
    print("\n  n_heads: attention heads (parallel patterns)")
    print("    4=focused | 8=balanced | 16=diverse perspectives")
    print("    Params: ~O(1), Memory: ~O(H), Quality: +multi-view attention")
    config['n_heads'] = int(input(f"  n_heads [8]: ").strip() or config.get('n_heads', 8))

    # vocab_size: token vocabulary
    print("\n  vocab_size: tokenizer vocabulary size")
    print("    16k=small/fast | 32k=GPT-2 | 50k=large coverage")
    print("    Params: ~O(V*d), Memory: ~O(V*d), Quality: +rare word handling")
    config['vocab_size'] = int(input(f"  vocab_size [32000]: ").strip() or config.get('vocab_size', 32000))

    # max_seq_len: context window
    print("\n  max_seq_len: maximum sequence length (context window)")
    print("    1024=short | 2048=medium | 4096=long context")
    print("    Params: ~O(1), Memory: ~O(seq^2 in attn), Quality: +long docs")
    config['max_seq_len'] = int(input(f"  max_seq_len [2048]: ").strip() or config.get('max_seq_len', 2048))

    # batch_size: throughput vs memory
    print("\n  batch_size: samples per update")
    print("    16=low mem | 32=balanced | 64+=high throughput")
    print("    Params: O(1), Memory: ~O(B*seq*d), Quality: +gradient stability")
    config['batch_size'] = int(input(f"  batch_size [32]: ").strip() or config.get('batch_size', 32))

    # -------------------------------------------------------------------------
    # SECTION D: TRAINER2 GEOMETRY
    # -------------------------------------------------------------------------
    print("\n[D] TRAINER2 GEOMETRY")
    print("-" * 50)

    config['coord_dim_n'] = int(input(f"  coord_dim_n [8]: ").strip() or config.get('coord_dim_n', 8))

    if config.get('frame_mode') == 'learned_lowrank':
        config['lowrank_r'] = int(input(f"  lowrank_r [4]: ").strip() or config.get('lowrank_r', 4))

    if config.get('rotor_mode') != 'off':
        config['rotor_k'] = int(input(f"  rotor_k [6]: ").strip() or config.get('rotor_k', 6))

    print("=" * 70)
    return config


def start_training(config):
    """Initialize and start training with given config."""

    print("\n" + "=" * 70)
    print("  INITIALIZING TRAINING")
    print("=" * 70)

    # Set comprehensive defaults for ALL required keys
    config.setdefault('field_dim', 16)
    config.setdefault('spatial_size', [8, 8])
    config.setdefault('d_model', 512)
    config.setdefault('n_heads', 8)
    config.setdefault('n_layers', 4)
    config.setdefault('n_mamba_layers', 4)  # For backwards compatibility
    config.setdefault('n_attention_layers', 2)
    config.setdefault('batch_size', 16)
    config.setdefault('max_epochs', 10)
    config.setdefault('lr', 0.0003)
    config.setdefault('adaptive_field', True)
    config.setdefault('use_causal_field', True)  # True = Causal Field default
    config.setdefault('trainer_backend', 'trainer2')
    config.setdefault('trainer2_confirm', False)
    config.setdefault('force_chunked', True)
    config.setdefault('trainer2_sdm_capacity', 2048)
    config.setdefault('trainer2_sdm_static_shapes', False)
    config.setdefault('log_interval', 10)
    config.setdefault('output_dir', './checkpoints/full')
    config.setdefault('val_split', 0.1)
    config.setdefault('test_split', 0.1)
    config.setdefault('vocab_size', 32000)
    config.setdefault('max_seq_len', 512)
    config.setdefault('weight_decay', 0.01)
    config.setdefault('warmup_steps', 500)
    config.setdefault('grad_accum_steps', 1)
    config.setdefault('save_interval', 1000)
    config.setdefault('data_path', './data/train.txt')  # Default data path
    config.setdefault('data_type', 'text')  # Default data type
    # Force CUDA device - fail if not available
    if torch.cuda.is_available():
        config.setdefault('device', 'cuda')
        print(f"CUDA ENABLED: {torch.cuda.get_device_name(0)}")
    else:
        print("=" * 70)
        print("FATAL: GPU NOT AVAILABLE")
        print("=" * 70)
        print(f"torch.cuda.is_available() = False")
        print(f"torch.version.cuda = {torch.version.cuda}")
        print("")
        print("Your PyTorch installation is CPU-only.")
        print("Check: python -c \"import torch; print(torch.cuda.is_available())\"")
        print("")
        print("To fix, reinstall PyTorch with CUDA:")
        print("  pip3 uninstall torch torchvision torchaudio")
        print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 70)
        raise RuntimeError("GPU required but CUDA not available. Reinstall PyTorch with CUDA support.")

    config.setdefault('device', 'cuda')
    # num_workers: Maximum parallelism on GPU, 0 on CPU (Windows compatibility)
    import os
    default_workers = 0 if not torch.cuda.is_available() else min(os.cpu_count(), 20)
    config.setdefault('num_workers', default_workers)
    config.setdefault('seed', 42)
    config.setdefault('val_data_path', None)
    config.setdefault('evolve_field', True)

    # Seed
    torch.manual_seed(config['seed'])

    # =========================================================================
    # TRAINER2 PARAMS - Configure BEFORE model/dataset creation
    # This allows params like coord_dim_n, rotor_k to influence parameter count
    # =========================================================================
    if config.get('trainer_backend') == 'trainer2':
        config = configure_trainer2_params(config)

    print(f"\n▶ Mode: {config['mode']}")
    print(f"▶ Device: {config['device']}")
    print(f"▶ Trainer backend: {config.get('trainer_backend')}")
    print(f"▶ Data: {config.get('data_path', 'N/A')}")

    # Handle resume - load checkpoint and continue training
    if 'resume' in config:
        checkpoint_path = config['resume']
        print(f"▶ Resuming from: {checkpoint_path}")

        # Load checkpoint to get saved config
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_config = checkpoint.get('config', {})

        # Provide defaults for all required config keys
        defaults = {
            'field_dim': 16,
            'spatial_size': [8, 8],
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 4,
            'n_mamba_layers': 4,
            'n_attention_layers': 2,
            'batch_size': 16,
            'max_epochs': 10,
            'lr': 0.0003,
            'adaptive_field': True,
            'use_causal_field': False,
            'log_interval': 10,
            'output_dir': './checkpoints/full',
            'val_split': 0.1,
            'test_split': 0.1
        }

        # Merge: defaults < saved_config < current config
        config = {**defaults, **saved_config, **config}
        config['resume_from'] = checkpoint_path  # Signal trainer to load weights

        print(f"✓ Checkpoint loaded: epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
        print(f"✓ Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        # Continue with model creation below (trainer will load weights)

    # Print configuration summary
    print("\n▶ Configuration:")
    print(f"  Field: {config.get('spatial_size')} spatial, {config.get('field_dim')}D tensor")
    print(f"  Model: d={config.get('d_model')}, layers={config.get('n_layers')}, heads={config.get('n_heads')}")
    print(f"  Training: batch={config.get('batch_size')}, epochs={config.get('max_epochs')}, lr={config.get('lr')}")
    print(f"  Architecture: {'CausalField' if config.get('use_causal_field') else 'Transformer'}")

    # Initialize models
    print("\n1. Creating models...")

    field_config = FieldConfig(
        spatial_size=tuple(config['spatial_size']),
        tensor_dim=config['field_dim'],
        adaptive_learning=config.get('adaptive_field', True),
        device=config['device']
    )

    field = CognitiveTensorField(field_config)

    # Choose model architecture
    if config.get('use_causal_field', False):
        # Causal Field with parallel evolution (O(N log N))
        n_attention_layers = config.get('n_attention_layers', 2)
        model = GeometricTransformerWithMamba(
            d_model=config['d_model'],
            n_mamba_layers=config.get('n_mamba_layers', config['n_layers']),
            n_attention_layers=n_attention_layers,
            n_heads=config['n_heads'],
            field_dim=config['field_dim'],
            max_seq_len=config['max_seq_len'],
            use_positional_encoding=True,
            use_temporal_encoding=True
        )
        print(f"   Field: {config['spatial_size']} spatial, {config['field_dim']}D tensor")
        print(f"   CausalField: {config['n_layers']} layers + {n_attention_layers} attention layers, d={config['d_model']}")
        print(f"   Complexity: O(N log N) via FFT convolution (fully parallel)")
    else:
        # Standard Transformer (O(N^2))
        model = GeometricTransformer(
            field_dim=config['field_dim'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            use_positional_encoding=True,
            use_temporal_encoding=True
        )
        print(f"   Field: {config['spatial_size']} spatial, {config['field_dim']}D tensor")
        print(f"   Standard Transformer: {config['n_layers']} layers, d={config['d_model']}")
        print(f"   Complexity: O(N^2)")

    # IMPORTANT: move model to target device BEFORE attaching trainables and BEFORE optimizer construction
    model = model.to(config['device'])

    # Ensure LM head exists before optimizer so its parameters are trained
    if not hasattr(model, 'lm_head') or model.lm_head is None:
        head_param = next(model.parameters())
        model.lm_head = nn.Linear(
            config['d_model'],
            config.get('vocab_size', 32000),
            device=head_param.device,
            dtype=head_param.dtype
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable: {trainable_params:,}")
    if frozen_params > 0:
        print(f"   ✓ Frozen: {frozen_params:,}")

    # Initialize data
    print("\n2. Loading data...")

    # Check if MNIST dataset
    if config.get('data_type') == 'mnist':
        print("   ✓ Loading MNIST dataset...")
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            root=config['data_path'],
            train=True,
            download=True,
            transform=transform
        )

        val_dataset = datasets.MNIST(
            root=config['data_path'],
            train=False,
            download=True,
            transform=transform
        )

        print(f"   ✓ Train samples: {len(train_dataset)}")
        print(f"   ✓ Val samples: {len(val_dataset)}")

        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        # MNIST doesn't need tokenizer
        tokenizer = None

    else:
        # Text-based datasets
        tokenizer = CognitiveTokenizer(vocab_size=config['vocab_size'])

        # Handle multiple data paths (no RAM merge)
        if 'data_paths' in config:
            data_path = config['data_paths']
            total_mb = sum(Path(p).stat().st_size for p in data_path) / (1024 * 1024)
            print(f"   ✓ Using {len(data_path)} files ({total_mb:.1f}MB total) without RAM merge")
        else:
            data_path = config['data_path']

            # If data file doesn't exist, generate sample data
            if not Path(data_path).exists():
                print(f"   ⚠ Data file not found: {data_path}")
                print(f"   ✓ Generating sample data instead...")
                data_path = generate_sample_data()
                config['data_path'] = data_path

            file_ext = Path(data_path).suffix.lower()

            # If not a plain text file, convert it first
            if file_ext not in ['.txt', '.text']:
                print(f"   Converting {file_ext} file to text...")
                reader = UniversalFileReader()

                try:
                    content = reader.read(data_path)
                    # Save converted content to temp file
                    temp_path = Path('./data/temp_converted.txt')
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    data_path = str(temp_path)
                    print(f"   ✓ Converted {len(content)} characters")
                except Exception as e:
                    print(f"   ✗ Error converting file: {e}")
                    print(f"   Using original path")

        # Auto-calculate optimal batch size if not specified or if scaling to large model
        if 'batch_size' not in config or config.get('d_model', 256) >= 1024:
            calculated_batch = calculate_optimal_batch_size(
                d_model=config.get('d_model', 256),
                seq_len=config.get('max_seq_len', 512),
                gpu_vram_gb=24  # RTX 4090
            )
            if 'batch_size' not in config:
                config['batch_size'] = calculated_batch
                print(f"  Using auto-calculated batch size: {calculated_batch}")
            else:
                if config['batch_size'] > calculated_batch:
                    print(f"  ⚠ Warning: batch_size={config['batch_size']} may exceed VRAM")
                    print(f"  ⚠ Recommended: {calculated_batch}")

        # Check file size to decide dataset type
        if isinstance(data_path, (list, tuple)):
            file_size_mb = sum(Path(p).stat().st_size for p in data_path) / (1024 * 1024)
        else:
            file_size_mb = Path(data_path).stat().st_size / (1024 * 1024)
        use_chunked = config.get('force_chunked', True) or file_size_mb > 50
        if isinstance(data_path, (list, tuple)) and not use_chunked:
            print("   ⚠ Multiple files require ChunkedTextDataset; forcing chunked mode.")
            use_chunked = True

        # Initialize split_info (will be populated for non-chunked datasets)
        split_info = None

        if use_chunked:
            print(f"   Using ChunkedTextDataset ({file_size_mb:.1f}MB file, memory-efficient)")
            # ChunkedTextDataset for large files (memory-efficient, supports BPTT)
            train_dataset = ChunkedTextDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_length=config['max_seq_len'],
                chunk_size=10000,
                bptt_window=config.get('bptt_window', 0),
                shuffle_buffer=1000,
                seed=config.get('seed', 42)
            )
            val_dataset = None  # ChunkedTextDataset doesn't support splitting
        else:
            print(f"   Using TextDataset ({file_size_mb:.1f}MB file)")
            # TextDataset for smaller files (supports splitting)
            full_dataset = TextDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_length=config['max_seq_len']
            )

            # Auto-split: train/val/test = 80%/10%/10%
            val_split = config.get('val_split', 0.1)
            test_split = config.get('test_split', 0.1)

            total_size = len(full_dataset)
            val_size = int(total_size * val_split)
            test_size = int(total_size * test_split)
            train_size = total_size - val_size - test_size

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config.get('seed', 42))
            )

            # Capture split info for checkpoint saving (enables val set recreation)
            split_info = {
                'split_seed': config.get('seed', 42),
                'val_split_ratio': val_split,
                'test_split_ratio': test_split,
                'dataset_length': total_size,
                'train_indices': list(train_dataset.indices),
                'val_indices': list(val_dataset.indices),
                'test_indices': list(test_dataset.indices),
            }

            print(f"   ✓ Split: {train_size} train, {val_size} val, {test_size} test")

        # Override with explicit val_data_path if provided
        if config.get('val_data_path'):
            val_dataset = TextDataset(
                data_path=config['val_data_path'],
                tokenizer=tokenizer,
                max_length=config['max_seq_len']
            )

        # DataLoader with optimized settings for GPU-bound training
        _base_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=False if use_chunked else True,  # IterableDataset requires shuffle=False
            num_workers=config.get('num_workers', 0),
            pin_memory=True,
            prefetch_factor=config.get('prefetch_factor', 2),  # Lower default, adaptive wrapper handles throughput
            persistent_workers=False if use_chunked else (config.get('num_workers', 0) > 0)
        )

        # Adaptive wrapper: monitors GPU memory and throttles when under pressure
        class AdaptiveDataLoader:
            """Wraps DataLoader with dynamic memory-aware throttling."""
            def __init__(self, loader, mem_threshold_pct=0.85, cooldown_sec=0.5):
                self._loader = loader
                self._mem_threshold = mem_threshold_pct
                self._cooldown = cooldown_sec
                self._throttle_count = 0
                # Forward common attributes
                self.batch_size = loader.batch_size
                self.num_workers = loader.num_workers
                self.dataset = loader.dataset

            def __iter__(self):
                for batch in self._loader:
                    # Check GPU memory pressure before yielding
                    if torch.cuda.is_available():
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        usage_pct = reserved / total if total > 0 else 0

                        if usage_pct > self._mem_threshold:
                            # Memory pressure: cleanup and wait
                            self._throttle_count += 1
                            if self._throttle_count % 10 == 1:  # Log every 10th throttle
                                print(f"[AdaptiveDataLoader] Memory {usage_pct*100:.1f}% > {self._mem_threshold*100:.0f}% threshold, throttling...")
                            torch.cuda.empty_cache()
                            import time
                            time.sleep(self._cooldown)

                    yield batch

            def __len__(self):
                return len(self._loader)

            @property
            def throttle_stats(self):
                return {'throttle_count': self._throttle_count}

        train_loader = AdaptiveDataLoader(
            _base_loader,
            mem_threshold_pct=config.get('mem_threshold_pct', 0.85),
            cooldown_sec=config.get('throttle_cooldown_sec', 0.5)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=True,
            prefetch_factor=config.get('prefetch_factor', 2),
            persistent_workers=config.get('num_workers', 0) > 0
        ) if val_dataset else None

        if not use_chunked:
            print(f"   ✓ Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"   ✓ Validation samples: {len(val_dataset)}")
        else:
            print(f"   ✓ Using chunked streaming (size unknown until iteration)")

    if config.get('trainer_backend') != 'trainer2':
        print("\n3. Setting up training...")

    # Preflight: attach input embedding BEFORE optimizer construction (no trainables created in trainer/forward).
    if tokenizer is not None:
        if not hasattr(model, 'input_embedding') or model.input_embedding is None:
            from Liorhybrid.training.embeddings import MultimodalEmbedding
            # Use tokenizer's vocab_size, not config default (fixes index out of bounds)
            actual_vocab_size = getattr(tokenizer, 'vocab_size', config.get('vocab_size', 32000))
            print(f"[main] Creating embedding with vocab_size={actual_vocab_size}")
            model.input_embedding = MultimodalEmbedding(
                vocab_size=actual_vocab_size,
                d_model=config['d_model'],
                max_seq_len=config.get('max_seq_len', 512)
            ).to(config['device'])

    # Preflight checklist gate
    run_preflight_checklist_or_die(model, field, tokenizer, config)

    if config.get('trainer_backend') != 'trainer2':
        raise RuntimeError(
            "This pipeline is locked to trainer2 (no autograd/backprop/optimizers). "
            "Set config['trainer_backend'] = 'trainer2'."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "trainer2 requires CUDA; torch.cuda.is_available() is False."
        )

    # Trainer2 wiring (CUDA-only, manual updates). No optimizers/backprop.
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    from Liorhybrid.training import trainer2 as trainer2_mod
    t2_cfg = trainer2_mod.TrainConfig()
    applied_keys = []
    skipped_keys = []
    for key, value in config.items():
        if hasattr(t2_cfg, key):
            setattr(t2_cfg, key, value)
            applied_keys.append(key)
        else:
            skipped_keys.append(key)
    t2_cfg.run_dir = str(output_dir)
    if config.get('run_name'):
        t2_cfg.run_name = config['run_name']

    print(f"[trainer2] applied config keys: {sorted(applied_keys)}")
    print(f"[trainer2] skipped config keys: {sorted(skipped_keys)}")
    mapped_keys = []
    if 'log_interval' in config:
        t2_cfg.log_every_windows = config['log_interval']
        mapped_keys.append('log_interval->log_every_windows')
    if 'save_interval' in config:
        t2_cfg.save_every_windows = config['save_interval']
        mapped_keys.append('save_interval->save_every_windows')
    if 'output_dir' in config:
        t2_cfg.run_dir = str(output_dir)
        mapped_keys.append('output_dir->run_dir')
    if 'bptt_window' in config and config['bptt_window'] > 0:
        t2_cfg.tbptt_window_steps = config['bptt_window']
        mapped_keys.append('bptt_window->tbptt_window_steps')
    if 'lr' in config:
        t2_cfg.eta_update = config['lr']
        mapped_keys.append('lr->eta_update')
    if mapped_keys:
        print(f"[trainer2] mapped config keys: {sorted(mapped_keys)}")

    memory = config.get('trainer2_memory')
    if memory is None:
        memory = trainer2_mod.SimpleSDMMemory(
            coord_dim_n=t2_cfg.coord_dim_n,
            capacity=config.get('trainer2_sdm_capacity', 2048),
            static_shapes=bool(config.get('trainer2_sdm_static_shapes', False)),
        )
        print("WARNING: trainer2_memory not provided; using SimpleSDMMemory (SDM ring buffer).")

    hooks = config.get('trainer2_hooks')
    if hooks is None:
        hooks = trainer2_mod.build_sdm_hooks(model, field, memory)

    rotor_state = config.get('trainer2_rotor_state')
    if t2_cfg.rotor_mode != "off":
        if rotor_state is None:
            rotor_state = trainer2_mod.RotorState(
                theta=torch.zeros(t2_cfg.rotor_k, device=torch.device(config['device']), dtype=torch.float32),
                theta2=torch.zeros(t2_cfg.rotor_k, device=torch.device(config['device']), dtype=torch.float32),
            )
            print("WARNING: trainer2_rotor_state not provided; using zero-initialized RotorState.")
        if getattr(rotor_state, "theta", None) is None:
            raise RuntimeError(
                "trainer2 rotor_mode != 'off' requires rotor_state.theta."
            )

    print(f"\n3. Trainer2 wired (manual updates, no autograd)")
    print(f"   ✓ Output directory: {output_dir}")

    # Config params were already collected via configure_trainer2_params() before model creation
    # Just confirm before starting
    if config.get("trainer2_confirm", False):
        confirm = input("\n  Start trainer2? [Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("\n  Training cancelled.")
            return

    print("\n" + "=" * 70)
    print("  TRAINER2 STARTED")
    print("=" * 70)
    trainer2_mod.trainer2_entrypoint(
        cfg=t2_cfg,
        model=model,
        field=field,
        memory=memory,
        train_loader=train_loader,
        hooks=hooks,
        rotor_state=rotor_state,
        val_loader=val_loader,
        tokenizer=tokenizer,
    )


def parse_args():
    """Parse command line arguments (for non-interactive mode)."""
    parser = argparse.ArgumentParser(description='Bayesian Cognitive Field Training')
    parser.add_argument('--mode', type=str, choices=['geometric', 'full'])
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--config', type=str, help='YAML config file')
    parser.add_argument('--use-causal-field', action='store_true',
                        help='Use Causal Field with parallel evolution (O(N log N))')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of CausalField layers')
    parser.add_argument('--n-attention-layers', type=int, default=2, help='Number of attention layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Max epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    return parser.parse_args()


def main():
    """Main entry point."""

    # Check if running interactively (no args) or with CLI args
    if len(sys.argv) == 1:
        # Interactive mode - loop until user exits
        while True:
            try:
                config = interactive_menu()
                if config:
                    start_training(config)
                else:
                    # User chose to exit
                    print("\nExiting...")
                    break
            except KeyboardInterrupt:
                print("\n\n✗ Interrupted by user.")
                # Return to menu instead of exiting
                continue
            except Exception as e:
                print("\n" + "=" * 70)
                print(f"ERROR: {type(e).__name__}: {e}")
                print("=" * 70)
                import traceback
                traceback.print_exc()
                print("\nReturning to main menu...")
                print("=" * 70)
                # Return to menu instead of exiting
                continue
    else:
        # CLI mode (not fully implemented yet - redirect to interactive)
        print("\n" + "=" * 70)
        print("  CLI mode not fully implemented yet.")
        print("  Run without arguments for interactive mode:")
        print("    python main.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
