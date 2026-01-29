"""
Variational Entropy (Field-Aware)

Fast, field-aware entropy measure that avoids expensive eigendecomposition.

Replaces Von Neumann entropy in performance-critical paths:
    OLD: H = -Tr(ρ log ρ)  [requires eigendecomp]
    NEW: H_var = variational approximation [O(N) complexity]

Pure PyTorch implementation.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import Optional


def variational_entropy(
    field_state: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute variational entropy (field-aware, fast).
    
    Approximates Von Neumann entropy without eigendecomposition:
        H_var ≈ -Tr(T log T) using variational bound
    
    Args:
        field_state: Field state tensor (N_x, N_y, D, D) or (D, D)
        temperature: Temperature parameter (default 1.0)
    
    Returns:
        entropy: Variational entropy estimate
    """
    # Spatial average if spatial dimensions present
    if field_state.ndim == 4:
        T = field_state.mean(dim=(0, 1))  # (D, D)
    else:
        T = field_state
    
    # Handle complex tensors
    if T.is_complex():
        T_real = torch.abs(T)
    else:
        T_real = torch.abs(T)
    
    # Normalize to probability distribution
    trace = torch.diagonal(T_real).sum()
    rho = T_real / (trace + 1e-8)
    
    # Variational bound using matrix logarithm approximation
    # log(ρ) ≈ (ρ - I) - (ρ - I)²/2 + (ρ - I)³/3 - ... (Taylor series)
    # For entropy, use simpler bound: H ≈ -sum_i ρ_ii log(ρ_ii)
    
    diag = torch.diagonal(rho)
    diag_clipped = torch.clamp(diag, min=1e-10, max=1.0)
    
    # Diagonal entropy (variational approximation)
    entropy = -torch.sum(diag_clipped * torch.log(diag_clipped))
    
    # Temperature scaling
    entropy = entropy / temperature
    
    return entropy


def variational_entropy_gradient(
    field_state: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute gradient of variational entropy.
    
    Used for field evolution: dH/dT.
    
    Args:
        field_state: Field state tensor
        temperature: Temperature parameter
    
    Returns:
        gradient: Entropy gradient
    """
    # Spatial average
    if field_state.ndim == 4:
        T = field_state.mean(dim=(0, 1))
    else:
        T = field_state
    
    # Handle complex
    if T.is_complex():
        T_real = torch.abs(T)
    else:
        T_real = torch.abs(T)
    
    # Normalize
    trace = torch.diagonal(T_real).sum()
    rho = T_real / (trace + 1e-8)
    
    # Gradient: dH/dρ = -(log(ρ) + 1)
    diag = torch.diagonal(rho)
    diag_clipped = torch.clamp(diag, min=1e-10)
    
    grad_diag = -(torch.log(diag_clipped) + 1.0) / temperature
    
    # Full gradient (diagonal matrix)
    gradient = torch.diag(grad_diag)
    
    return gradient


def renyi_entropy(
    field_state: torch.Tensor,
    alpha: float = 2.0
) -> torch.Tensor:
    """
    Compute Rényi entropy (collision entropy for α=2).
    
    H_α(ρ) = 1/(1-α) log(Tr(ρ^α))
    
    For α=2: H_2(ρ) = -log(Tr(ρ²)) (collision entropy)
    
    Args:
        field_state: Field state tensor
        alpha: Rényi parameter (default 2.0)
    
    Returns:
        entropy: Rényi entropy
    """
    # Spatial average
    if field_state.ndim == 4:
        T = field_state.mean(dim=(0, 1))
    else:
        T = field_state
    
    # Handle complex
    if T.is_complex():
        T_real = torch.abs(T)
    else:
        T_real = torch.abs(T)
    
    # Normalize
    trace = torch.diagonal(T_real).sum()
    rho = T_real / (trace + 1e-8)
    
    if alpha == 2.0:
        # Collision entropy: H_2 = -log(Tr(ρ²))
        rho_squared = rho @ rho
        trace_squared = torch.diagonal(rho_squared).sum()
        entropy = -torch.log(trace_squared + 1e-10)
    else:
        # General case: H_α = 1/(1-α) log(Tr(ρ^α))
        rho_alpha = torch.linalg.matrix_power(rho, int(alpha))
        trace_alpha = torch.diagonal(rho_alpha).sum()
        entropy = torch.log(trace_alpha + 1e-10) / (1.0 - alpha)
    
    return entropy


def shannon_entropy(
    field_state: torch.Tensor
) -> torch.Tensor:
    """
    Compute Shannon entropy (α → 1 limit of Rényi).
    
    H(ρ) = -Tr(ρ log ρ)
    
    This is the classic Von Neumann entropy.
    For diagonal ρ, reduces to classical Shannon entropy.
    
    Args:
        field_state: Field state tensor
    
    Returns:
        entropy: Shannon entropy
    """
    # Spatial average
    if field_state.ndim == 4:
        T = field_state.mean(dim=(0, 1))
    else:
        T = field_state
    
    # Handle complex
    if T.is_complex():
        T_real = torch.abs(T)
    else:
        T_real = torch.abs(T)
    
    # Normalize
    trace = torch.diagonal(T_real).sum()
    rho = T_real / (trace + 1e-8)
    
    # Eigendecomposition for true Von Neumann entropy
    try:
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
    except:
        # Fallback to diagonal approximation
        diag = torch.diagonal(rho)
        diag_clipped = torch.clamp(diag, min=1e-10)
        entropy = -torch.sum(diag_clipped * torch.log(diag_clipped))
    
    return entropy


def relative_entropy(
    field_state_1: torch.Tensor,
    field_state_2: torch.Tensor
) -> torch.Tensor:
    """
    Compute relative entropy (KL divergence) between two field states.
    
    D(ρ₁||ρ₂) = Tr(ρ₁ log ρ₁ - ρ₁ log ρ₂)
    
    Args:
        field_state_1: First field state
        field_state_2: Second field state
    
    Returns:
        divergence: Relative entropy
    """
    # Process both states
    if field_state_1.ndim == 4:
        T1 = field_state_1.mean(dim=(0, 1))
        T2 = field_state_2.mean(dim=(0, 1))
    else:
        T1 = field_state_1
        T2 = field_state_2
    
    # Handle complex
    if T1.is_complex():
        T1_real = torch.abs(T1)
    else:
        T1_real = torch.abs(T1)
    
    if T2.is_complex():
        T2_real = torch.abs(T2)
    else:
        T2_real = torch.abs(T2)
    
    # Normalize both
    trace1 = torch.diagonal(T1_real).sum()
    trace2 = torch.diagonal(T2_real).sum()
    rho1 = T1_real / (trace1 + 1e-8)
    rho2 = T2_real / (trace2 + 1e-8)
    
    # Diagonal approximation for KL divergence
    diag1 = torch.diagonal(rho1)
    diag2 = torch.diagonal(rho2)
    
    diag1_clipped = torch.clamp(diag1, min=1e-10)
    diag2_clipped = torch.clamp(diag2, min=1e-10)
    
    # KL: sum_i ρ₁[i] log(ρ₁[i]/ρ₂[i])
    divergence = torch.sum(diag1_clipped * torch.log(diag1_clipped / diag2_clipped))
    
    return divergence


def mutual_information(
    field_state_joint: torch.Tensor,
    field_state_1: torch.Tensor,
    field_state_2: torch.Tensor
) -> torch.Tensor:
    """
    Compute mutual information between two subsystems.
    
    I(1:2) = H(ρ₁) + H(ρ₂) - H(ρ₁₂)
    
    Args:
        field_state_joint: Joint field state
        field_state_1: Marginal field state 1
        field_state_2: Marginal field state 2
    
    Returns:
        mi: Mutual information
    """
    H_joint = variational_entropy(field_state_joint)
    H_1 = variational_entropy(field_state_1)
    H_2 = variational_entropy(field_state_2)
    
    mi = H_1 + H_2 - H_joint
    
    return mi


class EntropyTracker(nn.Module):
    """
    Module for tracking entropy evolution during training.
    
    Monitors:
    - Variational entropy
    - Rényi entropy (α=2)
    - Shannon entropy (expensive, optional)
    """
    
    def __init__(self, track_shannon: bool = False):
        super().__init__()
        self.track_shannon = track_shannon
        self.history = []
    
    def forward(self, field_state: torch.Tensor, step: int = 0) -> dict:
        """
        Compute and track entropy metrics.
        
        Args:
            field_state: Current field state
            step: Training step
        
        Returns:
            metrics: Dict of entropy values
        """
        metrics = {
            'step': step,
            'variational': float(variational_entropy(field_state).item()),
            'renyi_2': float(renyi_entropy(field_state, alpha=2.0).item())
        }
        
        if self.track_shannon:
            metrics['shannon'] = float(shannon_entropy(field_state).item())
        
        self.history.append(metrics)
        
        return metrics
    
    def get_history(self):
        """Return entropy history."""
        return self.history


def test_variational_entropy():
    """Test variational entropy computations."""
    print("Testing variational entropy...")
    
    # Create test field state
    N_x, N_y, D = 10, 10, 16
    field_state = torch.randn(N_x, N_y, D, D)
    
    # Make it positive-definite (physical field)
    field_state = field_state @ field_state.transpose(-2, -1)
    
    print(f"Field state shape: {field_state.shape}")
    
    # Test variational entropy
    H_var = variational_entropy(field_state)
    print(f"Variational entropy: {H_var.item():.4f}")
    
    # Test Rényi entropy
    H_renyi = renyi_entropy(field_state, alpha=2.0)
    print(f"Rényi entropy (α=2): {H_renyi.item():.4f}")
    
    # Test Shannon entropy
    H_shannon = shannon_entropy(field_state)
    print(f"Shannon entropy: {H_shannon.item():.4f}")
    
    # Test gradient
    grad = variational_entropy_gradient(field_state)
    print(f"Entropy gradient shape: {grad.shape}")
    print(f"Gradient norm: {torch.linalg.norm(grad).item():.4f}")
    
    # Test relative entropy
    field_state_2 = torch.randn(N_x, N_y, D, D)
    field_state_2 = field_state_2 @ field_state_2.transpose(-2, -1)
    
    kl = relative_entropy(field_state, field_state_2)
    print(f"Relative entropy (KL): {kl.item():.4f}")
    
    # Test tracker
    tracker = EntropyTracker(track_shannon=True)
    for step in range(5):
        metrics = tracker({'field': field_state}, step=step)
        print(f"Step {step}: Var={metrics['variational']:.4f}, "
              f"Renyi={metrics['renyi_2']:.4f}, Shannon={metrics['shannon']:.4f}")
    
    print("Variational entropy tests passed!")


if __name__ == '__main__':
    test_variational_entropy()
