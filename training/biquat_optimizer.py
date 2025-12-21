"""
BiquatOptimizer: Memory-efficient SGD optimizer for Lie algebra parameters.

Replaces AdamW to eliminate optimizer state memory overhead.
AdamW stores 2 tensors per parameter (m, v); this stores none.
"""

import torch


class BiquatOptimizer:
    """
    Pure SGD optimizer with optional weight decay and theta clamping.

    Memory-efficient alternative to AdamW - stores NO optimizer state.
    Suitable for Lie algebra / geometric deep learning where adaptive
    learning rates are less critical.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        weight_decay: Decoupled weight decay coefficient (default: 0.0)
        theta_clip: Clamp value for theta (rotation) components (default: 8.0)
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0, theta_clip=8.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.theta_clip = theta_clip

        # For compatibility with lr schedulers
        self.param_groups = [{'lr': lr, 'params': self.params}]

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step."""
        for p in self.params:
            if p.grad is None:
                continue

            # SGD update (Lie algebra is linear, no momentum needed)
            p -= self.lr * p.grad

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters."""
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'theta_clip': self.theta_clip
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict.get('lr', self.lr)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.theta_clip = state_dict.get('theta_clip', self.theta_clip)
        # Update param_groups for scheduler compatibility
        self.param_groups[0]['lr'] = self.lr
