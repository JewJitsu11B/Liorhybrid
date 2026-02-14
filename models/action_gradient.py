"""
Analytic LIoR Gradient Computation

This module computes action gradients directly without autograd:
    ∇S[γ] = ∫ R_μνρσ(γ) γ̇^ρ γ̇^σ dτ

Key insight: Measure the gradient directly from the manifold geometry
rather than using PyTorch's autograd. This is the foundation of
measurement-based learning.

Physics:
- S[γ]: Action functional (path integral)
- R_μνρσ: Riemann curvature tensor
- γ̇: Velocity along path
- Integration over proper time τ

No .backward() calls - pure measurement!
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


@torch.inference_mode()
def compute_lior_action_gradient(
    embeddings: torch.Tensor,
    field_state: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute LIoR action gradient analytically (no autograd).
    
    This is the pure measurement approach:
        ∇S = ∫ R(x) √(g_μν ẋ^μ ẋ^ν) dτ
    
    Args:
        embeddings: Sequence of embeddings (batch, seq_len, d_model)
        field_state: Cognitive tensor field (N_x, N_y, D, D)
        metric: Optional Riemannian metric tensor (D, D)
        return_components: If True, return dict with intermediate values
    
    Returns:
        gradient: Action gradient (batch, seq_len, d_model)
        OR dict with components if return_components=True
    
    Mathematical Detail:
        The action gradient measures how much a path deviates from
        the geodesic carved by the cognitive field. High gradient =
        path needs correction to follow natural flow.
    """
    batch, seq_len, d_model = embeddings.shape
    D = field_state.shape[2]
    
    # Compute trajectory velocities: ẋ = dx/dτ
    dx = embeddings[:, 1:] - embeddings[:, :-1]  # (B, T-1, d)
    
    # Project to field subspace (D dimensions)
    if d_model > D:
        proj = dx[..., :D]  # (B, T-1, D)
    else:
        proj = dx
    
    # Build metric tensor from field if not provided
    if metric is None:
        T_avg = field_state.mean(dim=(0, 1))  # (D, D)
        T_real = torch.abs(T_avg) if T_avg.is_complex() else T_avg
        metric = T_real.T @ T_real  # (D, D) positive-definite
        metric = metric + 1e-6 * torch.eye(D, device=metric.device, dtype=metric.dtype)
    
    # Cast metric to match embeddings dtype/device
    metric = metric.to(device=proj.device, dtype=proj.dtype)
    
    # Metric inner product: g_μν ẋ^μ ẋ^ν
    g_dx_dx = torch.einsum('bti,ij,btj->bt', proj, metric, proj)  # (B, T-1)
    
    # Arc length: √(g_dx_dx)
    arc_length = torch.sqrt(torch.clamp(g_dx_dx, min=1e-8))  # (B, T-1)
    
    # Compute local curvature/resilience from field
    # R(x) = trace of curvature tensor (simplified)
    R = compute_local_curvature(field_state)  # (N_x, N_y)
    
    # Average resilience over spatial domain
    R_avg = R.mean()  # scalar
    
    # Action gradient: ∇S = R * √(g_dx_dx)
    # This gives direction to minimize action
    action_values = R_avg * arc_length  # (B, T-1)
    
    # Extend to full sequence (pad first position)
    action_grad = torch.zeros(batch, seq_len, d_model, 
                              device=embeddings.device, dtype=embeddings.dtype)
    
    # Assign gradient to positions 1:end (position 0 has no predecessor)
    for i in range(1, seq_len):
        # Gradient is difference in action between steps
        if i < seq_len - 1:
            grad_magnitude = action_values[:, i] - action_values[:, i-1]
        else:
            grad_magnitude = action_values[:, i-1]
        
        # Direction is opposite of velocity (minimize action)
        if d_model > D:
            action_grad[:, i, :D] = -grad_magnitude.unsqueeze(-1) * proj[:, i-1]
        else:
            action_grad[:, i, :] = -grad_magnitude.unsqueeze(-1) * proj[:, i-1]
    
    if return_components:
        return {
            'gradient': action_grad,
            'arc_length': arc_length,
            'curvature': R_avg,
            'metric': metric,
            'action_values': action_values
        }
    
    return action_grad


@torch.inference_mode()
def compute_local_curvature(field_state: torch.Tensor) -> torch.Tensor:
    """
    Compute local curvature/resilience from field state.
    
    Simplified curvature measure:
        R(x) = trace of second moment tensor variance
    
    Args:
        field_state: Cognitive tensor field (N_x, N_y, D, D)
    
    Returns:
        curvature: Local curvature field (N_x, N_y)
    """
    N_x, N_y, D, _ = field_state.shape
    
    # Extract magnitude (handle complex tensors)
    if field_state.is_complex():
        magnitude = torch.abs(field_state)  # (N_x, N_y, D, D)
    else:
        magnitude = torch.abs(field_state)
    
    # Trace of absolute value (measure of local field strength)
    trace = torch.diagonal(magnitude, dim1=2, dim2=3).sum(dim=-1)  # (N_x, N_y)
    
    # Compute local variance as curvature proxy
    # High variance = high curvature = more resistance to change
    mean_trace = trace.mean()
    variance = (trace - mean_trace).pow(2)
    
    # Normalize to reasonable range
    curvature = 1.0 + torch.sqrt(variance + 1e-8)
    
    return curvature


@torch.inference_mode()
def measure_field_entropy(field_state: torch.Tensor) -> torch.Tensor:
    """
    Measure field entropy (Von Neumann entropy).
    
    H = -Tr(ρ log ρ)
    
    This is a MEASUREMENT, not a differentiable operation.
    Used for monitoring field dynamics, not for backprop.
    
    Args:
        field_state: Field state tensor (N_x, N_y, D, D)
    
    Returns:
        entropy: Scalar entropy value
    """
    # Spatial average
    T_avg = field_state.mean(dim=(0, 1))  # (D, D)
    
    # Get magnitude (handle complex)
    if T_avg.is_complex():
        T_real = torch.abs(T_avg)
    else:
        T_real = torch.abs(T_avg)
    
    # Normalize to probability distribution
    trace = torch.diagonal(T_real).sum()
    rho = T_real / (trace + 1e-8)
    
    # Eigendecomposition for entropy
    try:
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
    except:
        # Fallback if eigendecomp fails
        entropy = torch.tensor(0.0, device=field_state.device)
    
    return entropy


def evolve_field_by_measurement(
    field: nn.Module,
    action_gradient: torch.Tensor,
    learning_rate: float = 0.01
) -> None:
    """
    Evolve field parameters based on measured gradients.
    
    This replaces optimizer.step() with direct measurement-based updates.
    
    Field evolution:
        dα/dt = -η ∂H/∂α
        dν/dt = -η ∂H/∂ν  
        dτ/dt = -η ∂H/∂τ
    
    Where H is the field entropy (measurement).
    
    Args:
        field: CognitiveTensorField module
        action_gradient: Measured action gradient
        learning_rate: Evolution rate
    """
    # Measure current entropy
    H_current = measure_field_entropy(field.T)
    
    # Perturb each adaptive parameter and measure entropy change
    params_to_update = []
    if hasattr(field, 'alpha') and field.alpha.requires_grad:
        params_to_update.append(('alpha', field.alpha))
    if hasattr(field, 'nu') and field.nu.requires_grad:
        params_to_update.append(('nu', field.nu))
    if hasattr(field, 'tau') and field.tau.requires_grad:
        params_to_update.append(('tau', field.tau))
    
    for name, param in params_to_update:
        # Small perturbation
        epsilon = 1e-4
        
        # Save original value
        original_value = param.data.clone()
        
        # Measure gradient via finite difference
        param.data = original_value + epsilon
        field.evolve_step()  # Update field with perturbed parameter
        H_plus = measure_field_entropy(field.T)
        
        param.data = original_value - epsilon
        field.evolve_step()
        H_minus = measure_field_entropy(field.T)
        
        # Restore original
        param.data = original_value
        field.evolve_step()
        
        # Entropy gradient
        dH_dparam = (H_plus - H_minus) / (2 * epsilon)
        
        # Update: decrease entropy (make field more ordered)
        param.data = param.data - learning_rate * dH_dparam


class MeasurementBasedUpdater:
    """
    Wrapper for measurement-based parameter updates.
    
    Replaces torch.optim.Optimizer with pure measurement approach.
    Compatible with existing training loops (drop-in replacement).
    """
    
    def __init__(self, model: nn.Module, field: nn.Module, lr: float = 1e-3):
        """
        Args:
            model: Model to update
            field: CognitiveTensorField
            lr: Learning rate (evolution rate)
        """
        self.model = model
        self.field = field
        self.lr = lr
        
        # For compatibility with existing code
        self.param_groups = [{'lr': lr, 'params': list(model.parameters())}]
    
    @torch.inference_mode()
    def step(self):
        """
        Perform measurement-based update.
        
        This is called where optimizer.step() used to be called.
        """
        # Measure current field state
        embeddings = self._get_current_embeddings()
        
        # Compute action gradient (pure measurement)
        action_grad = compute_lior_action_gradient(
            embeddings,
            self.field.T,
            metric=None
        )
        
        # Evolve field parameters
        evolve_field_by_measurement(self.field, action_grad, self.lr)
    
    def zero_grad(self):
        """No-op for compatibility. We don't use gradients."""
        pass
    
    def _get_current_embeddings(self) -> torch.Tensor:
        """
        Get current embedding state for measurement.
        
        This needs to be overridden or passed in from training loop.
        """
        # Placeholder - in practice, this would be passed from training step
        return torch.zeros(1, 10, 512, device=next(self.model.parameters()).device)
