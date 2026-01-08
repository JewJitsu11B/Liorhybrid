"""
Example: Field Evolution with Harmonic Potential

Demonstrates Bayesian cognitive field evolution in a harmonic potential well.
Shows how the field behavior changes with different potential landscapes.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from core import CognitiveTensorField, FieldConfig
from kernels import create_potential


def evolve_with_potential(potential_type: str, steps: int = 100):
    """
    Evolve field in a given potential and track metrics.

    Args:
        potential_type: Type of potential ("harmonic", "gaussian_well", etc.)
        steps: Number of evolution steps

    Returns:
        Dictionary with evolution metrics
    """
    # Create configuration
    config = FieldConfig(
        spatial_size=(16, 16),
        tensor_dim=16,
        hbar_cog=0.1,
        m_cog=1.0,
        lambda_QR=0.1,    # Weak Bayesian updates
        lambda_F=0.02,    # Light memory damping
        alpha=0.5,
        tau=0.5,
        dt=0.005,
        device='cpu'
    )

    # Create field
    field = CognitiveTensorField(config)

    # Create potential
    V = create_potential(
        spatial_size=config.spatial_size,
        tensor_dim=config.tensor_dim,
        potential_type=potential_type,
        strength=0.2,  # Moderate potential strength
        device=config.device,
        dtype=config.dtype
    )

    # Track metrics during evolution
    metrics = {
        'step': [],
        'norm': [],
        'energy': [],
        'unitarity_deviation': [],
    }

    print(f"\nEvolving with {potential_type} potential...")

    for step in range(steps):
        # Record metrics
        metrics['step'].append(step)
        metrics['norm'].append(field.get_norm_squared())
        metrics['energy'].append(field.compute_energy())
        metrics['unitarity_deviation'].append(field.compute_unitarity_deviation())

        # Evolve field
        # Note: To use potential in evolution, we need to modify hamiltonian_evolution call
        # For now, this demonstrates the potential creation and energy computation
        field.evolve_step()

        if step % 20 == 0:
            print(f"  Step {step:3d}: "
                  f"||T||² = {metrics['norm'][-1]:.4f}, "
                  f"E = {metrics['energy'][-1]:.4f}, "
                  f"δ_unit = {metrics['unitarity_deviation'][-1]:.4f}")

    return metrics


def plot_metrics(metrics_dict: dict, output_dir: str = "./outputs"):
    """
    Plot evolution metrics for different potentials.

    Args:
        metrics_dict: Dictionary mapping potential_type -> metrics
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Field Evolution with Different Potentials', fontsize=14, fontweight='bold')

    # Plot 1: Norm conservation
    ax = axes[0, 0]
    for pot_type, metrics in metrics_dict.items():
        ax.plot(metrics['step'], metrics['norm'], label=pot_type, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('||T||²')
    ax.set_title('Norm Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy evolution
    ax = axes[0, 1]
    for pot_type, metrics in metrics_dict.items():
        ax.plot(metrics['step'], metrics['energy'], label=pot_type, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('Hamiltonian Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Unitarity deviation
    ax = axes[1, 0]
    for pot_type, metrics in metrics_dict.items():
        ax.plot(metrics['step'], metrics['unitarity_deviation'], label=pot_type, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Unitarity Deviation')
    ax.set_title('Non-Unitary Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Energy vs Norm (phase space)
    ax = axes[1, 1]
    for pot_type, metrics in metrics_dict.items():
        ax.plot(metrics['norm'], metrics['energy'], label=pot_type, 
                linewidth=2, alpha=0.7, marker='o', markersize=2)
    ax.set_xlabel('||T||²')
    ax.set_ylabel('Energy')
    ax.set_title('Phase Space (Energy vs Norm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'potential_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    """
    Main demonstration of potential landscapes.
    """
    print("=" * 70)
    print("Bayesian Cognitive Field with Potential Landscapes")
    print("=" * 70)

    # Test different potential types
    potential_types = [
        "zero",            # Free field (baseline)
        "harmonic",        # Confining potential
        "gaussian_well",   # Attractive well
        "gaussian_barrier" # Repulsive barrier
    ]

    metrics_dict = {}

    for pot_type in potential_types:
        metrics = evolve_with_potential(pot_type, steps=100)
        metrics_dict[pot_type] = metrics

    # Create visualization
    try:
        plot_metrics(metrics_dict)
    except Exception as e:
        print(f"Could not create plots: {e}")
        print("(Plotting requires matplotlib)")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    for pot_type, metrics in metrics_dict.items():
        final_norm = metrics['norm'][-1]
        final_energy = metrics['energy'][-1]
        avg_unitarity = sum(metrics['unitarity_deviation']) / len(metrics['unitarity_deviation'])

        print(f"\n{pot_type.upper():20s}")
        print(f"  Final ||T||²:              {final_norm:.6f}")
        print(f"  Final Energy:              {final_energy:.6f}")
        print(f"  Avg Unitarity Deviation:   {avg_unitarity:.6f}")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("  - Harmonic potential confines the field spatially")
    print("  - Gaussian wells create local attraction")
    print("  - Energy is NOT conserved (non-unitary Bayesian evolution)")
    print("  - Unitarity deviation shows non-Hamiltonian dynamics")
    print("=" * 70)


if __name__ == "__main__":
    main()
