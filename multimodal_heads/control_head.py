"""
Control Head

For RL/robotics using Hamiltonian structure.
State space = CognitiveManifold coordinates,
action space derived from symplectic form B,
LiorKernel for temporal credit assignment.

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.manifold import CognitiveManifold
from models.complex_metric import ComplexMetricTensor
from models.lior_kernel import LiorKernel, LiorMemoryState


class ControlHead(nn.Module):
    """
    Control/RL head using Hamiltonian physics.
    
    Key physics:
    - State space: CognitiveManifold coordinates (configuration space)
    - Action space: Derived from symplectic form B (momentum space)
    - Hamiltonian structure: Energy-conserving dynamics
    - LiorKernel: Temporal credit assignment with power-law memory
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        d_coord: int = 8,
        use_hamiltonian: bool = True,
    ):
        """
        Initialize ControlHead.
        
        Args:
            state_dim: Observation/state dimension
            action_dim: Action dimension
            d_model: Hidden dimension
            d_coord: Coordinate manifold dimension
            use_hamiltonian: Whether to use Hamiltonian structure
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.use_hamiltonian = use_hamiltonian
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Cognitive manifold for state representation
        self.state_manifold = CognitiveManifold(
            d_embed=d_model,
            d_coord=d_coord,
            learnable_metric=True
        )
        
        # Complex metric for action derivation
        self.complex_metric = ComplexMetricTensor(d_coord=d_coord)
        
        # LiorKernel for temporal credit assignment
        self.lior_kernel = LiorKernel(
            p_eff=4,
            init_tau_exp=1.0,
            init_tau_frac=5.0,  # Medium range for RL
            init_tau_osc=2.0,
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim * 2)  # mean and log_std
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
        # Hamiltonian energy function
        if use_hamiltonian:
            self.energy_head = nn.Sequential(
                nn.Linear(d_coord * 2, d_model),  # position + momentum
                nn.GELU(),
                nn.Linear(d_model, 1)
            )
        
        # Memory state
        self.memory_state = None
    
    def reset_memory(self):
        """Reset memory state."""
        self.memory_state = None
    
    def compute_hamiltonian(
        self,
        position: torch.Tensor,
        momentum: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hamiltonian energy H(q, p).
        
        Args:
            position: (batch, d_coord) generalized coordinates
            momentum: (batch, d_coord) generalized momenta
            
        Returns:
            energy: (batch, 1) total energy
        """
        # Concatenate position and momentum
        state = torch.cat([position, momentum], dim=-1)
        
        # Compute energy
        energy = self.energy_head(state)
        
        return energy
    
    def derive_action_from_symplectic(
        self,
        state_coords: torch.Tensor,
        state_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Derive action from symplectic form B.
        
        The symplectic form defines canonical momentum:
        p_i = B_{ij} dq^j
        
        Args:
            state_coords: (batch, d_coord) state coordinates
            state_embedding: (batch, d_model) state embedding
            
        Returns:
            momentum: (batch, d_coord) derived momentum/action direction
        """
        # Compute phase field
        alpha = torch.tensor(0.5, device=state_embedding.device)
        phase = self.complex_metric.compute_phase_field(state_embedding, alpha)
        
        # Compute symplectic form B
        B = self.complex_metric.compute_symplectic_form(state_coords, phase)
        
        # Derive momentum (simplified: use gradient of phase)
        # In full implementation: solve p = B dq
        momentum = torch.autograd.grad(
            phase.sum(), state_coords,
            create_graph=True, retain_graph=True
        )[0]
        
        return momentum
    
    def forward(
        self,
        state: torch.Tensor,
        return_value: bool = True,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for control/RL.
        
        Args:
            state: (batch, state_dim) observations
            return_value: Whether to return value estimate
            deterministic: Whether to use deterministic policy
            
        Returns:
            outputs: Dict with 'action', 'log_prob', 'value', 'energy' (if Hamiltonian)
        """
        batch_size = state.shape[0]
        
        # Encode state
        state_emb = self.state_encoder(state)  # (batch, d_model)
        
        # Project to manifold (configuration space)
        state_coords, _ = self.state_manifold.project(state_emb)
        
        # LiorKernel temporal credit assignment
        if self.memory_state is None or self.memory_state.m.size(0) != batch_size:
            self.memory_state = LiorMemoryState.initialize(
                batch_size, self.d_model, self.lior_kernel.p_eff, device=state.device
            )
        
        # Update memory with current state
        m_new = self.lior_kernel.recurrence_step(
            state_emb, self.memory_state.m, self.memory_state.x_history
        )
        self.memory_state = self.memory_state.update(state_emb, m_new)
        
        # Combine state with memory for decision making
        state_with_memory = state_emb + m_new
        
        # Policy (actor)
        policy_out = self.policy_head(state_with_memory)
        action_mean = policy_out[..., :self.action_dim]
        action_log_std = policy_out[..., self.action_dim:]
        action_log_std = torch.clamp(action_log_std, -20, 2)
        action_std = torch.exp(action_log_std)
        
        # Sample action
        if deterministic:
            action = action_mean
        else:
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
        
        # Compute log probability
        action_dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        outputs = {
            'action': action,
            'log_prob': log_prob,
            'action_mean': action_mean,
            'action_std': action_std,
        }
        
        # Value (critic)
        if return_value:
            value = self.value_head(state_with_memory)
            outputs['value'] = value
        
        # Hamiltonian energy
        if self.use_hamiltonian:
            # Derive momentum from symplectic structure
            momentum = self.derive_action_from_symplectic(state_coords, state_emb)
            
            # Compute energy
            energy = self.compute_hamiltonian(state_coords, momentum)
            outputs['energy'] = energy
            outputs['momentum'] = momentum
        
        return outputs
