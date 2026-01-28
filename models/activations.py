"""
Activation Functions for Cognitive Field Models

SwiGLU as default FFN activation for maximum knowledge per parameter.
Debug alerts (not silent fallbacks) when using legacy activations.

SwiGLU: (W_up * x) ⊙ SiLU(W_gate * x) @ W_down

References:
- Shazeer (2020): GLU Variants Improve Transformer
- PaLM, LLaMA, Mistral all use SwiGLU
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation: gated linear unit with SiLU (Swish) gate.

    Output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

    Default expansion_factor = 8/3 gives parameter parity with 4x GELU FFN:
    - GELU: 2 * d_model * 4 * d_model = 8 * d_model²
    - SwiGLU: 3 * d_model * (8/3) * d_model = 8 * d_model²

    Args:
        d_model: Input/output dimension
        expansion_factor: Multiplier for intermediate dim (default 8/3)
        bias: Use bias in linear layers (default False for efficiency)
        dropout: Dropout rate after activation
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: float = None,
        bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        if expansion_factor is None:
            expansion_factor = 8 / 3  # Param parity with 4x GELU

        d_ff = int(d_model * expansion_factor)
        # Round to multiple of 64 for hardware efficiency (tensor cores)
        d_ff = ((d_ff + 63) // 64) * 64

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Store for diagnostics
        self._d_ff = d_ff
        self._expansion = d_ff / d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            Output tensor with same shape as input
        """
        # gate branch: SiLU activation
        gate = F.silu(self.w_gate(x))
        # up branch: linear projection (no activation)
        up = self.w_up(x)
        # element-wise product then down projection
        return self.w_down(self.dropout(gate * up))

    def extra_repr(self) -> str:
        return f'd_ff={self._d_ff}, expansion={self._expansion:.3f}'


class FFN(nn.Module):
    """
    Configurable FFN wrapper - SwiGLU default, debug alerts on fallback.

    NO silent fallbacks. If activation='gelu' or 'relu', logs a warning.
    This ensures developers are aware when not using optimal activation.

    Args:
        d_model: Input/output dimension
        expansion_factor: Multiplier for intermediate dim
                         (default: 8/3 for SwiGLU, 4 for GELU/ReLU)
        activation: 'swiglu' (default), 'gelu', or 'relu'
        dropout: Dropout rate
        bias: Use bias in linear layers (default False)

    Example:
        # Recommended (default SwiGLU)
        ffn = FFN(d_model=256)

        # Legacy activation (triggers warning)
        ffn = FFN(d_model=256, activation='gelu')
        # -> UserWarning: FFN using legacy activation 'gelu'...
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: float = None,
        activation: str = 'swiglu',
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        self._activation_type = activation

        if activation != 'swiglu':
            # Debug alert - NOT a silent fallback
            warnings.warn(
                f"FFN using legacy activation '{activation}' instead of SwiGLU. "
                "This may reduce model quality. Set activation='swiglu' for optimal performance.",
                UserWarning,
                stacklevel=2
            )

        if activation == 'swiglu':
            self.ffn = SwiGLU(
                d_model=d_model,
                expansion_factor=expansion_factor,
                bias=bias,
                dropout=dropout
            )
        elif activation == 'gelu':
            # Legacy GELU FFN
            d_ff = int(d_model * (expansion_factor if expansion_factor is not None else 4))
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(d_ff, d_model, bias=bias)
            )
        elif activation == 'relu':
            # Legacy ReLU FFN
            d_ff = int(d_model * (expansion_factor if expansion_factor is not None else 4))
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(d_ff, d_model, bias=bias)
            )
        else:
            raise ValueError(
                f"Unknown activation: '{activation}'. "
                "Use 'swiglu' (recommended), 'gelu', or 'relu'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

    def extra_repr(self) -> str:
        return f'activation={self._activation_type}'
