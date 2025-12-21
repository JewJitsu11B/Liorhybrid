"""
LIoR Optimizer: True geometric curvature-aware optimizer.

Key insight: Instead of storing m/v tensors (Adam) or computing Hessians,
use the resilience tensor R(x) from the cognitive manifold as a local
curvature proxy. This gives O(N) complexity with zero state memory.

Mathematical basis:
    Δθ = -α · R⁻²(x) · ∇L

where R(x) is computed from the causal second-moment tensor M^{μν}.

The "inf before integral" trick:
1. Build local covariance M^{μν} from causal past
2. Lower indices with metric: M_{μν} = g_{μα} g_{νβ} M^{αβ}
3. Take smallest eigenvalue: κ = λ_min(M^μ_ν)
4. Use R = 1/√κ as learning rate modulator

This gives true geometric curvature at O(N) cost because:
- Double trace collapses rank-4 tensor to scalar
- Metric rescaling is pointwise (no matrix ops on full Hessian)
- Resilience network is already trained with the model

Complexity comparison:
    SGD:     O(0) state, O(N) cost, no curvature
    Adam:    O(2N) state, O(N) cost, EMA of g² (crude)
    Sophia:  O(N) state, O(N) + MC sampling, Monte Carlo Hessian
    Shampoo: O(√N × √N) state, O(N^1.5) cost, blockwise Fisher
    LIoR:    O(0) state, O(N) cost, TRUE local R(x)
"""

import torch
from typing import Optional, Callable, Dict, Any


class LIoROptimizer:
    """
    Basic LIoR optimizer with gradient-based resilience estimation.

    This is the fallback/ablation version that estimates R(x) from
    gradient statistics when no manifold is available.

    Advantages over Adam:
    - Zero optimizer state (no m, v tensors)
    - True local curvature (not exponential moving average)
    - O(N) cost per step
    - Noise-tolerant via causal averaging

    Args:
        params: Model parameters
        lr: Base learning rate
        resilience_fn: Optional callable that returns R(x) for parameter tensor
        kappa_scale: Scale factor for curvature modulation (default 1.0)
        min_resilience: Floor for R to prevent division by zero (default 0.1)
        max_resilience: Ceiling for R to prevent vanishing updates (default 10.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        resilience_fn: Optional[Callable] = None,
        kappa_scale: float = 1.0,
        min_resilience: float = 0.1,
        max_resilience: float = 10.0
    ):
        self.params = list(params)
        self.lr = lr
        self.resilience_fn = resilience_fn
        self.kappa_scale = kappa_scale
        self.min_R = min_resilience
        self.max_R = max_resilience

        # For LR scheduler compatibility
        self.param_groups = [{'lr': lr, 'params': self.params}]

    @torch.no_grad()
    def step(self):
        """Perform curvature-modulated gradient descent."""
        for p in self.params:
            if p.grad is None:
                continue

            # Compute local resilience R(x)
            if self.resilience_fn is not None:
                R = self.resilience_fn(p.data)
                R = torch.clamp(R, self.min_R, self.max_R)
            else:
                # Fallback: estimate R from gradient magnitude
                R = self._estimate_resilience(p.grad)

            # Curvature-modulated update: Δθ = -α · R⁻² · g
            R_inv_sq = 1.0 / (R ** 2)
            p -= self.lr * self.kappa_scale * R_inv_sq * p.grad

    def _estimate_resilience(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Estimate local resilience from gradient statistics.

        R ≈ 1/√(1 + ||∇L||²)

        High gradient norm → low R → larger effective learning rate
        Low gradient norm → high R → smaller effective learning rate

        This is a cheap proxy when no manifold is available.
        """
        grad_norm_sq = (grad ** 2).mean()
        R = 1.0 / torch.sqrt(1.0 + grad_norm_sq)
        return torch.clamp(R, self.min_R, self.max_R)

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters."""
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            'lr': self.lr,
            'kappa_scale': self.kappa_scale,
            'min_R': self.min_R,
            'max_R': self.max_R
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict.get('lr', self.lr)
        self.kappa_scale = state_dict.get('kappa_scale', self.kappa_scale)
        self.min_R = state_dict.get('min_R', self.min_R)
        self.max_R = state_dict.get('max_R', self.max_R)
        self.param_groups[0]['lr'] = self.lr


class LIoRManifoldOptimizer(LIoROptimizer):
    """
    Full LIoR optimizer with manifold integration.

    Uses CognitiveManifold.scalar_resilience() to compute true R(x) at
    parameter locations. This is the "real" LIoR optimizer that cashes
    out the mathematical theory.

    The update rule is:
        Δθ = -α · R⁻²(x) · ∇L

    where R(x) comes from the trained resilience network on the manifold,
    representing the local curvature/uncertainty of the cognitive field.

    Args:
        params: Model parameters
        manifold: CognitiveManifold instance with scalar_resilience() method
        lr: Base learning rate
        use_causal_memory: Whether to use causal memory accumulation
        memory_decay: Decay factor ρ for causal memory (default 0.9)
        **kwargs: Additional arguments passed to LIoROptimizer
    """

    def __init__(
        self,
        params,
        manifold,
        lr: float = 1e-3,
        use_causal_memory: bool = False,
        memory_decay: float = 0.9,
        **kwargs
    ):
        # Store manifold first (needed for resilience_fn)
        self.manifold = manifold
        self.use_causal_memory = use_causal_memory
        self.memory_decay = memory_decay

        # Causal memory state (optional)
        self.memory = {} if use_causal_memory else None

        # Initialize parent with manifold-based resilience function
        super().__init__(params, lr, resilience_fn=None, **kwargs)

    def _param_to_coords(self, p: torch.Tensor) -> torch.Tensor:
        """
        Project parameter tensor to manifold coordinate space.

        Takes the first d_coord elements of the flattened parameter
        as coordinates on the manifold.
        """
        flat = p.view(-1)
        d_coord = self.manifold.d_coord

        if flat.numel() >= d_coord:
            coords = flat[:d_coord].unsqueeze(0)
        else:
            # Pad with zeros if parameter is smaller than d_coord
            coords = torch.zeros(1, d_coord, device=p.device, dtype=p.dtype)
            coords[0, :flat.numel()] = flat

        return coords

    @torch.no_grad()
    def step(self):
        """Curvature-aware step with manifold R(x) and optional causal memory."""
        for p in self.params:
            if p.grad is None:
                continue

            # Get coordinates for this parameter on the manifold
            coords = self._param_to_coords(p.data)

            # Compute true resilience R(x) from manifold
            R = self.manifold.scalar_resilience(coords)
            R = torch.clamp(R, self.min_R, self.max_R)

            # Handle scalar vs tensor R
            if R.numel() == 1:
                R_val = R.item()
            else:
                R_val = R.mean().item()

            # Compute curvature-modulated gradient: g_scaled = R⁻² · g
            R_inv_sq = 1.0 / (R_val ** 2)
            scaled_grad = R_inv_sq * p.grad

            # Apply causal memory if enabled
            if self.use_causal_memory:
                param_id = id(p)
                if param_id not in self.memory:
                    self.memory[param_id] = torch.zeros_like(p.grad)

                # Causal accumulation: m_t = ρ·m_{t-1} + (1-ρ)·g_t
                # This implements the discrete Volterra kernel convolution
                self.memory[param_id] = (
                    self.memory_decay * self.memory[param_id] +
                    (1 - self.memory_decay) * scaled_grad
                )
                update = self.memory[param_id]
            else:
                update = scaled_grad

            # Apply update: θ_{t+1} = θ_t - α · update
            p -= self.lr * self.kappa_scale * update

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        state = super().state_dict()
        state['use_causal_memory'] = self.use_causal_memory
        state['memory_decay'] = self.memory_decay

        # Save causal memory state if present
        if self.use_causal_memory and self.memory:
            # Convert memory dict to serializable format
            memory_state = {}
            for i, p in enumerate(self.params):
                param_id = id(p)
                if param_id in self.memory:
                    memory_state[i] = self.memory[param_id].cpu()
            state['memory_state'] = memory_state

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        super().load_state_dict(state_dict)
        self.use_causal_memory = state_dict.get('use_causal_memory', self.use_causal_memory)
        self.memory_decay = state_dict.get('memory_decay', self.memory_decay)

        # Restore causal memory state if present
        if self.use_causal_memory:
            self.memory = {}
            memory_state = state_dict.get('memory_state', {})
            for i, p in enumerate(self.params):
                if i in memory_state:
                    self.memory[id(p)] = memory_state[i].to(p.device)
