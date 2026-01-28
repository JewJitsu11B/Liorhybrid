"""
Simple Field Evolution Example

Demonstrates basic usage of the Bayesian cognitive field.

This script:
1. Creates a field with default configuration
2. Runs evolution for a fixed number of steps
3. Tracks norm conservation
4. Prints diagnostic information
5. Optionally enables adaptive parameter learning
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import argparse
from Liorhybrid.core import CognitiveTensorField, FAST_TEST_CONFIG
from Liorhybrid.utils.metrics import compute_norm_conservation


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run Bayesian cognitive field evolution'
    )
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Enable adaptive parameter learning (Corollary: Adaptive Learning)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Bayesian Cognitive Field - Simple Evolution Example")
    print("=" * 60)

    # Create field
    print("\n1. Initializing field...")
    config = FAST_TEST_CONFIG

    # Enable adaptive learning if requested
    if args.adaptive:
        config.adaptive_learning = True
        print("   Mode: Adaptive parameter learning")
    else:
        print("   Mode: Fixed parameters")

    print(f"   Spatial size: {config.spatial_size}")
    print(f"   Tensor dimension: {config.tensor_dim}")
    print(f"   Device: {config.device}")

    field = CognitiveTensorField(config)

    # Print initial state
    initial_norm = field.get_norm_squared()
    print(f"\n   Initial ||T||²: {initial_norm:.6f}")

    # Run evolution
    print("\n2. Running evolution...")
    n_steps = 100

    for step in range(n_steps):
        field.evolve_step()

        if (step + 1) % 20 == 0:
            norm = field.get_norm_squared()
            if args.adaptive:
                entropy = field.compute_entropy().item()
                alpha_val = field.alpha.item()
                nu_mean = field.nu.mean().item()
                tau_mean = field.tau.mean().item()
                print(f"   Step {step+1:3d}: ||T||² = {norm:.6f}, "
                      f"H = {entropy:.2e}, α = {alpha_val:.4f}, "
                      f"ν = {nu_mean:.4f}, τ = {tau_mean:.4f}")
            else:
                print(f"   Step {step+1:3d}: ||T||² = {norm:.6f}")

    # Final diagnostics
    print("\n3. Final diagnostics:")
    final_norm = field.get_norm_squared()
    print(f"   Final ||T||²: {final_norm:.6f}")
    print(f"   Norm change: {(final_norm - initial_norm) / initial_norm * 100:.2f}%")

    if args.adaptive:
        final_entropy = field.compute_entropy().item()
        print(f"   Final entropy: {final_entropy:.4e}")
        print(f"   Final α: {field.alpha.item():.6f}")
        print(f"   Final ν (mean): {field.nu.mean().item():.6f}")
        print(f"   Final τ (mean): {field.tau.mean().item():.6f}")

    # Compute norm conservation
    if len(field.history) > 0:
        norms = compute_norm_conservation(list(field.history))
        print(f"   Norm std/mean: {torch.std(norms) / torch.mean(norms):.4f}")

    print("\n   Total evolution time: {:.4f}".format(field.t))
    print(f"   Total steps: {field.step_count}")
    print(f"   History buffer size: {len(field.history)}")

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
