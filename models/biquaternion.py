"""
Biquaternion Algebra for White-Box Cognitive Spacetime - PURE REAL

Complex biquaternions (H_C) via PURE REAL arithmetic (no torch.complex).
Avoids ComplexHalf bugs entirely. All ops fp16/bf16 compatible.

State layout: [B, N, 16] real tensor
    0..3   : Q_M_re  - present quaternion real part
    4..7   : Q_M_im  - present quaternion imaginary part
    8..11  : Q_H_re  - memory quaternion real part
    12..15 : Q_H_im  - memory quaternion imaginary part

Total: 16 real DOF = 2 biquaternions (4 real quaternions)

This layer exposes (Q_M, Q_H) as structured 2x-quaternion fields. Higher-level
modules may treat these as coordinates in cognitive spacetime and compute
metrics/curvatures offline; this layer itself remains O(N) and doesn't perform
neighbor search or distance calculations.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# Pack/Unpack Helpers (Pure Real)
# =============================================================================

def pack_biquat(Q_M_re: torch.Tensor, Q_M_im: torch.Tensor,
                Q_H_re: torch.Tensor, Q_H_im: torch.Tensor) -> torch.Tensor:
    """Pack 4 real quaternions into [B, N, 16]."""
    return torch.cat([Q_M_re, Q_M_im, Q_H_re, Q_H_im], dim=-1)


def unpack_biquat(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
    """Unpack [B, N, 16] into 4 real quaternions (Q_M_re, Q_M_im, Q_H_re, Q_H_im)."""
    return state[..., 0:4], state[..., 4:8], state[..., 8:12], state[..., 12:16]


# =============================================================================
# Real Quaternion Product (Hamilton's Formula)
# =============================================================================

def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Real quaternion product p*q using Hamilton's formula.
    p, q: [..., 4] real tensors (fp16/bf16 ok)
    """
    p0, p1, p2, p3 = p.unbind(-1)
    q0, q1, q2, q3 = q.unbind(-1)

    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0

    return torch.stack([r0, r1, r2, r3], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: (a, b, c, d) -> (a, -b, -c, -d)"""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


###placeholder###
# Future: metric/regularization functions
# def quat_norm_sq(q: torch.Tensor) -> torch.Tensor:
#     """Quaternion norm squared for future metric computation."""
#     return (q ** 2).sum(dim=-1)
#
# def biquat_minkowski_norm(q_re, q_im):
#     """Pseudo-Riemannian norm: |q_re|^2 - |q_im|^2 for cognitive spacetime metric."""
#     return (q_re ** 2).sum(-1) - (q_im ** 2).sum(-1)
###placeholder###


# =============================================================================
# BiQuatTransform - Learnable Biquaternion Linear Operator (Pure Real)
# =============================================================================

class BiQuatTransform(nn.Module):
    """
    Learnable linear operator in biquaternion space using PURE REAL arithmetic.

    Represents W = W_re + i*W_im as two real quaternions.
    Applies: (W_re + i*W_im) * (q_re + i*q_im) using quaternion product.

    This is an SL(2,C) transformation (Lorentz rotation + boost).
    """

    def __init__(self, learnable: bool = True, normalize: bool = False, name: str = ""):
        super().__init__()
        self.normalize = normalize
        self.name = name  # For diagnostic logging

        # W = W_re + i*W_im stored as two real 4-vectors
        # Init to identity: W = 1 + 0i + 0j + 0k (real part only)
        if learnable:
            self.W_re = nn.Parameter(torch.tensor([1., 0., 0., 0.], dtype=torch.float32))
            self.W_im = nn.Parameter(torch.tensor([0., 0., 0., 0.], dtype=torch.float32))
        else:
            self.register_buffer('W_re', torch.tensor([1., 0., 0., 0.], dtype=torch.float32))
            self.register_buffer('W_im', torch.tensor([0., 0., 0., 0.], dtype=torch.float32))

    def forward(self, q_re: torch.Tensor, q_im: torch.Tensor, diagnose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply biquaternion multiplication: W * q = (W_re + i*W_im) * (q_re + i*q_im)

        Args:
            q_re, q_im: [B, N, 4] real tensors (fp16/bf16)
            diagnose: If True, print diagnostic info
        Returns:
            out_re, out_im: [B, N, 4] real tensors
        """
        if diagnose:
            print(f"    [BiQuatTransform:{self.name}] input q_re: nan={q_re.isnan().any().item()}, "
                  f"inf={q_re.isinf().any().item()}, range=[{q_re.min().item():.4g}, {q_re.max().item():.4g}]")
            print(f"    [BiQuatTransform:{self.name}] input q_im: nan={q_im.isnan().any().item()}, "
                  f"inf={q_im.isinf().any().item()}, range=[{q_im.min().item():.4g}, {q_im.max().item():.4g}]")

        W_re, W_im = self.W_re, self.W_im

        if self.normalize:
            # Unit biquaternion constraint
            norm_sq = (W_re ** 2).sum() + (W_im ** 2).sum()
            # FP16-SAFE: Use 0.01 instead of 1e-8
            norm = torch.sqrt(norm_sq + 0.01)
            W_re = W_re / norm
            W_im = W_im / norm

        # Broadcast to input shape and match dtype (for AMP)
        W_re = W_re.to(dtype=q_re.dtype, device=q_re.device).view(1, 1, 4)
        W_im = W_im.to(dtype=q_re.dtype, device=q_re.device).view(1, 1, 4)

        if diagnose:
            print(f"    [BiQuatTransform:{self.name}] W_re: nan={W_re.isnan().any().item()}, "
                  f"inf={W_re.isinf().any().item()}, range=[{W_re.min().item():.4g}, {W_re.max().item():.4g}]")
            print(f"    [BiQuatTransform:{self.name}] W_im: nan={W_im.isnan().any().item()}, "
                  f"inf={W_im.isinf().any().item()}, range=[{W_im.min().item():.4g}, {W_im.max().item():.4g}]")

        # Complex multiplication via real quaternion products:
        # (W_re + i*W_im) * (q_re + i*q_im) = (W_re*q_re - W_im*q_im) + i*(W_re*q_im + W_im*q_re)
        out_re = quat_mul(W_re, q_re) - quat_mul(W_im, q_im)
        out_im = quat_mul(W_re, q_im) + quat_mul(W_im, q_re)

        # FP16-SAFE: Clamp outputs to prevent gradient overflow during backward pass
        out_re = torch.clamp(out_re, min=-1e4, max=1e4)
        out_im = torch.clamp(out_im, min=-1e4, max=1e4)

        if diagnose:
            print(f"    [BiQuatTransform:{self.name}] output out_re: nan={out_re.isnan().any().item()}, "
                  f"inf={out_re.isinf().any().item()}, range=[{out_re.min().item():.4g}, {out_re.max().item():.4g}]")
            print(f"    [BiQuatTransform:{self.name}] output out_im: nan={out_im.isnan().any().item()}, "
                  f"inf={out_im.isinf().any().item()}, range=[{out_im.min().item():.4g}, {out_im.max().item():.4g}]")

        return out_re, out_im


# =============================================================================
# Causal Accumulator (Pure Real, Bounded Hyperparameters)
# =============================================================================

class CausalAccumulator(nn.Module):
    """
    Causal accumulation law using pure real arithmetic:

        Q_H_new = decay * Q_H + impulse_scale * impulse(Q_M)
        T = alpha * Q_M + (1-alpha) * transport(Q_H_new)

    Non-commutativity from temporal ordering (recursive update), not algebra.

    All scalar hyperparameters are bounded:
        - alpha: sigmoid -> (0, 1)
        - decay: sigmoid -> (0, 1) to prevent blowup
        - impulse_scale: softplus -> [0, inf) to keep non-negative
    """

    def __init__(
        self,
        alpha: float = 0.5,
        decay: float = 0.9,
        impulse_scale: float = 0.1,
        learnable_transforms: bool = True,
        normalize_transforms: bool = False
    ):
        super().__init__()

        # Raw params - will be transformed to safe ranges in forward
        # Init raw values so transformed values match desired init
        self.alpha_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(alpha), dtype=torch.float32))
        self.decay_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(decay), dtype=torch.float32))
        self.impulse_scale_raw = nn.Parameter(torch.tensor(self._inv_softplus(impulse_scale), dtype=torch.float32))

        # Learnable quaternionic transforms (with names for diagnostics)
        self.impulse_map = BiQuatTransform(learnable=learnable_transforms, normalize=normalize_transforms, name="impulse")
        self.transport_map = BiQuatTransform(learnable=learnable_transforms, normalize=normalize_transforms, name="transport")

    @staticmethod
    def _inv_sigmoid(y: float) -> float:
        """Inverse sigmoid for initialization."""
        y = max(1e-6, min(1 - 1e-6, y))
        return float(torch.log(torch.tensor(y / (1 - y))))

    @staticmethod
    def _inv_softplus(y: float) -> float:
        """Inverse softplus for initialization."""
        y = max(1e-6, y)
        return float(torch.log(torch.exp(torch.tensor(y)) - 1))

    def forward(
        self,
        Q_M_re: torch.Tensor, Q_M_im: torch.Tensor,
        Q_H_re: torch.Tensor, Q_H_im: torch.Tensor,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply causal accumulation.

        Args:
            Q_M_re, Q_M_im: Present state [B, N, 4]
            Q_H_re, Q_H_im: Memory state [B, N, 4]
            diagnose: If True, print diagnostic info
        Returns:
            T_re, T_im: Output field [B, N, 4]
            Q_H_new_re, Q_H_new_im: Updated memory [B, N, 4]
        """
        if diagnose:
            print(f"  [CausalAccumulator] START")
            print(f"  [CausalAccumulator] Q_M_re: nan={Q_M_re.isnan().any().item()}, "
                  f"inf={Q_M_re.isinf().any().item()}, range=[{Q_M_re.min().item():.4g}, {Q_M_re.max().item():.4g}]")
            print(f"  [CausalAccumulator] Q_H_re: nan={Q_H_re.isnan().any().item()}, "
                  f"inf={Q_H_re.isinf().any().item()}, range=[{Q_H_re.min().item():.4g}, {Q_H_re.max().item():.4g}]")

        # Bounded hyperparams (cast to input dtype for AMP)
        alpha = torch.sigmoid(self.alpha_raw).to(Q_M_re.dtype)
        decay = torch.sigmoid(self.decay_raw).to(Q_H_re.dtype)
        impulse_scale = F.softplus(self.impulse_scale_raw).to(Q_H_re.dtype)

        if diagnose:
            print(f"  [CausalAccumulator] alpha={alpha.item():.4g}, decay={decay.item():.4g}, "
                  f"impulse_scale={impulse_scale.item():.4g}")

        # Impulse from present
        impulse_re, impulse_im = self.impulse_map(Q_M_re, Q_M_im, diagnose=diagnose)

        # Recursive memory update: Q_H_new = decay * Q_H + impulse_scale * impulse
        Q_H_new_re = decay * Q_H_re + impulse_scale * impulse_re
        Q_H_new_im = decay * Q_H_im + impulse_scale * impulse_im

        if diagnose:
            print(f"  [CausalAccumulator] Q_H_new_re: nan={Q_H_new_re.isnan().any().item()}, "
                  f"inf={Q_H_new_re.isinf().any().item()}, range=[{Q_H_new_re.min().item():.4g}, {Q_H_new_re.max().item():.4g}]")

        # Transport memory for output combination
        Q_H_trans_re, Q_H_trans_im = self.transport_map(Q_H_new_re, Q_H_new_im, diagnose=diagnose)

        # Output: T = alpha * Q_M + (1-alpha) * Q_H_transported
        T_re = alpha * Q_M_re + (1 - alpha) * Q_H_trans_re
        T_im = alpha * Q_M_im + (1 - alpha) * Q_H_trans_im

        if diagnose:
            print(f"  [CausalAccumulator] T_re: nan={T_re.isnan().any().item()}, "
                  f"inf={T_re.isinf().any().item()}, range=[{T_re.min().item():.4g}, {T_re.max().item():.4g}]")
            print(f"  [CausalAccumulator] END")

        return T_re, T_im, Q_H_new_re, Q_H_new_im


# =============================================================================
# BiQuatCausalLayer - Full Layer (Pure Real)
# =============================================================================

class BiQuatCausalLayer(nn.Module):
    """
    Causal field layer using biquaternion algebra with PURE REAL arithmetic.
    O(N) cost, no torch.complex, fp16/bf16 compatible.

    Args:
        d_model: Model dimension
        d_field: Field dimension (must be 16)
        alpha: Initial causal mixing
        detach_memory: If True (default), detach memory every step.
        bptt_window: If > 0, detach memory every N steps (windowed BPTT).
                     If -1, never detach (full BPTT, high memory!).
                     If 0, use detach_memory parameter.
    """

    def __init__(
        self,
        d_model: int,
        d_field: int = 16,
        alpha: float = 0.5,
        detach_memory: bool = True,
        bptt_window: int = 0
    ):
        super().__init__()
        assert d_field == 16, "Biquaternion layer requires d_field=16"

        self.d_model = d_model
        self.d_field = d_field
        self.detach_memory = detach_memory
        self.bptt_window = bptt_window
        self.step_counter = 0  # Track steps for windowed BPTT

        self.input_proj = nn.Linear(d_model, d_field)
        self.accumulator = CausalAccumulator(alpha=alpha)
        self.output_proj = nn.Linear(d_field, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: dict = None,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with causal field evolution.

        Args:
            x: [B, N, d_model]
            memory: Optional dict with 'Q_H_re', 'Q_H_im' keys
            diagnose: If True, print diagnostic info
        Returns:
            output: [B, N, d_model]
            new_memory: Updated memory dict
        """
        B, N, D = x.shape
        residual = x

        if diagnose:
            print(f"[BiQuatCausalLayer] START shape={list(x.shape)}")
            print(f"[BiQuatCausalLayer] input x: nan={x.isnan().any().item()}, "
                  f"inf={x.isinf().any().item()}, range=[{x.min().item():.4g}, {x.max().item():.4g}]")

        # Project to field dimension
        x_proj = self.input_proj(x)  # [B, N, 16]

        if diagnose:
            print(f"[BiQuatCausalLayer] x_proj: nan={x_proj.isnan().any().item()}, "
                  f"inf={x_proj.isinf().any().item()}, range=[{x_proj.min().item():.4g}, {x_proj.max().item():.4g}]")

        # Unpack present state only - memory is carried in dict, not projection
        Q_M_re, Q_M_im, _, _ = unpack_biquat(x_proj)
        # ^ We ignore Q_H from projection: memory comes from recursive state only

        # Get memory (or init to zeros on first step)
        if memory is not None and 'Q_H_re' in memory:
            Q_H_re = memory['Q_H_re']
            Q_H_im = memory['Q_H_im']
            if diagnose:
                print(f"[BiQuatCausalLayer] Using existing memory")
        else:
            Q_H_re = torch.zeros_like(Q_M_re)
            Q_H_im = torch.zeros_like(Q_M_im)
            if diagnose:
                print(f"[BiQuatCausalLayer] Initializing memory to zeros")

        # Causal accumulation
        T_re, T_im, Q_H_new_re, Q_H_new_im = self.accumulator(
            Q_M_re, Q_M_im, Q_H_re, Q_H_im, diagnose=diagnose
        )

        # Pack output
        output_packed = pack_biquat(T_re, T_im, Q_H_new_re, Q_H_new_im)

        if diagnose:
            print(f"[BiQuatCausalLayer] output_packed: nan={output_packed.isnan().any().item()}, "
                  f"inf={output_packed.isinf().any().item()}, range=[{output_packed.min().item():.4g}, {output_packed.max().item():.4g}]")

        # Project back to model dim
        output = self.output_proj(output_packed)

        if diagnose:
            print(f"[BiQuatCausalLayer] after output_proj: nan={output.isnan().any().item()}, "
                  f"inf={output.isinf().any().item()}, range=[{output.min().item():.4g}, {output.max().item():.4g}]")

        output = self.norm(residual + output)

        if diagnose:
            print(f"[BiQuatCausalLayer] final output: nan={output.isnan().any().item()}, "
                  f"inf={output.isinf().any().item()}, range=[{output.min().item():.4g}, {output.max().item():.4g}]")
            print(f"[BiQuatCausalLayer] END")

        # Increment step counter
        self.step_counter += 1

        # Determine whether to detach based on BPTT window
        should_detach = False

        if self.bptt_window > 0:
            # Windowed BPTT: detach every N steps
            should_detach = (self.step_counter % self.bptt_window == 0)
        elif self.bptt_window == -1:
            # Full BPTT: never detach
            should_detach = False
        else:
            # bptt_window == 0: use detach_memory parameter
            should_detach = self.detach_memory

        # Store memory (detach if needed)
        if should_detach:
            Q_H_store_re = Q_H_new_re.detach()
            Q_H_store_im = Q_H_new_im.detach()
        else:
            Q_H_store_re = Q_H_new_re
            Q_H_store_im = Q_H_new_im

        new_memory = {
            'Q_H_re': Q_H_store_re,
            'Q_H_im': Q_H_store_im
        }

        return output, new_memory
