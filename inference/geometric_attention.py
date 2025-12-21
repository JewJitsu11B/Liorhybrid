"""
Geometric Attention Mechanism

Implements attention using geometric products (wedge, tensor, spinor)
instead of standard dot-product similarity.

This is the "conscious" cognitive interface to the evolved T_ij field.

Architecture:
    Input: Q (query from prompt), K, V (from field state), T_field
    Compute: geometric_score(Q, K, T_field) using wedge/tensor/spinor
    Apply: softmax normalization
    Output: attention_weights @ V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .geometric_products import geometric_score, geometric_score_from_phase, geometric_score_from_exponential
from .field_extraction import FieldToKeyValue
try:
    from bayesian_cognitive_field.training.execution_tracker import track_first_call
except ModuleNotFoundError:
    from training.execution_tracker import track_first_call


# =============================================================================
# OPTION 3: TRIALITY-PROJECTED BIQUATERNION STRUCTURE
# =============================================================================
# 24 generator DOF → 16D complex biquaternion via exponential map
#
# Structure:
#   A ∈ ℝ⁸  - real exponential generator (amplitude/spinor 1)
#   B ∈ ℝ⁸  - phase/imaginary generator (amplitude/spinor 2)
#   Θ ∈ ℝ⁸  - shared rotation-boost generator (bivector space)
#
# Θ components:
#   Θ₁–Θ₃: Spatial rotations (SO(3))
#   Θ₄–Θ₆: Lorentz boosts (hyperbolic)
#   Θ₇:    Complex phase (U(1))
#   Θ₈:    Dilation / causal scaling
#
# Output: Ψ = A·exp(Θ) + B·exp(iΘ) ∈ ℝ¹⁶
#
# This is a triality embedding: 3 symmetric 8D structures interacting
# via exponential/gauge symmetry, matching (V, S⁺, S⁻) of Spin(8).
# =============================================================================


class ExponentialPhaseExtractor(nn.Module):
    """
    Option 3: 24-parameter generator projected to 16D complex biquaternion
    via (A, B, Θ) → A·exp(Θ) + B·exp(iΘ)

    This structure provides:
    - 8 extra DOF for smoother optimization landscape
    - Geodesic updates via Lie exponential map
    - Disentangled rotation/boost/phase/dilation
    - Full bf16 compatibility (no complex dtype, no atan2, no sqrt)
    - Never-NaN exponentials (Θ clamped to [-8, 8])

    Expected convergence speedup: 3-10× fewer steps vs scalar phase.
    """

    def __init__(self, d_k: int):
        """
        Initialize Option 3 exponential extractor.

        Args:
            d_k: Dimension per attention head
        """
        super().__init__()
        self.d_k = d_k

        # Project to 24D generator space: A(8) + B(8) + Θ(8)
        self.proj = nn.Linear(d_k, 24)

        # Initialize with small weights for stability
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 16D biquaternion via exponential map.

        Args:
            x: Input tensor (batch, n_heads, seq_len, d_k)

        Returns:
            psi_real: 16D real biquaternion (batch, n_heads, seq_len, 16)
        """
        # Project to 24D generator space
        gen = self.proj(x)  # (batch, n_heads, seq_len, 24)

        # Split into 3×8 generators
        A_gen = gen[..., 0:8]    # Real exponential generator
        B_gen = gen[..., 8:16]   # Phase/imaginary generator
        Theta_gen = gen[..., 16:24]  # Shared rotation-boost generator

        # ---------------------------------------------------------
        # 1. Compute exp(Θ) - real exponential, bounded
        # ---------------------------------------------------------
        # Clamp Θ to prevent exp() overflow (exp(8) ≈ 2981, safe for bf16)
        Theta_clamped = Theta_gen.clamp(-8.0, 8.0)
        exp_Theta = torch.exp(Theta_clamped)  # (batch, n_heads, seq_len, 8)

        # ---------------------------------------------------------
        # 2. Compute exp(iΘ) without complex dtype
        # ---------------------------------------------------------
        # sin/cos are stable because Θ is clamped
        # These stay in [-1, 1] naturally
        cos_Theta = torch.cos(Theta_clamped)
        sin_Theta = torch.sin(Theta_clamped)

        # ---------------------------------------------------------
        # 3. Compute Ψ = A·exp(Θ) + B·exp(iΘ)
        # ---------------------------------------------------------
        # A branch: A_k * exp(Θ_k) for each of 8 components
        A_part = A_gen * exp_Theta  # (batch, n_heads, seq_len, 8)

        # B branch: B_k * exp(iΘ_k) = B_k * (cos(Θ_k) + i*sin(Θ_k))
        # Store as 16 reals: [B*cos, B*sin]
        B_cos = B_gen * cos_Theta  # (batch, n_heads, seq_len, 8)
        B_sin = B_gen * sin_Theta  # (batch, n_heads, seq_len, 8)

        # ---------------------------------------------------------
        # 4. Combine into 16D real biquaternion
        # ---------------------------------------------------------
        # Layout: [A·exp(Θ), B·cos(Θ), B·sin(Θ)] but we need 16D
        # Interpretation: first 8 = real part, second 8 = imaginary part
        # A·exp(Θ) contributes to real part
        # B·cos(Θ) contributes to real part
        # B·sin(Θ) contributes to imaginary part
        psi_real = A_part + B_cos  # Real part (8D)
        psi_imag = B_sin           # Imaginary part (8D)

        psi = torch.cat([psi_real, psi_imag], dim=-1)  # (batch, n_heads, seq_len, 16)

        return psi

    def get_generators(self, x: torch.Tensor) -> tuple:
        """
        Extract raw generators for regularization.

        Args:
            x: Input tensor (batch, n_heads, seq_len, d_k)

        Returns:
            (A_gen, B_gen, Theta_gen): Each (batch, n_heads, seq_len, 8)
        """
        gen = self.proj(x)
        return gen[..., 0:8], gen[..., 8:16], gen[..., 16:24]


class PhaseExtractor(nn.Module):
    """
    Extract amplitude and phase from vector representations.

    Compresses high-dimensional vectors (d_k) into scalar (A, theta) pairs
    using the Psi = A * exp(i*theta) representation.

    This enables parallel multi-head computation by avoiding dimension
    mismatch issues between field tensors and per-head dimensions.
    """

    def __init__(self, d_k: int):
        """
        Initialize phase extractor.

        Args:
            d_k: Dimension per attention head
        """
        super().__init__()
        # Project d_k-dimensional vector to 2D complex plane (real, imaginary)
        self.to_complex = nn.Linear(d_k, 2)

        # Initialize with very small weights to prevent gradient explosion
        # atan2 and sqrt have steep gradients near zero
        nn.init.normal_(self.to_complex.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.to_complex.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract amplitude and phase from input vectors.

        Args:
            x: Input tensor (batch, n_heads, seq_len, d_k)

        Returns:
            A: Amplitudes (batch, n_heads, seq_len)
            theta: Phases in radians (batch, n_heads, seq_len)
        """
        # Ensure input dtype matches layer weights (FP16 safety)
        target_dtype = self.to_complex.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Project to complex plane components
        components = self.to_complex(x)  # (batch, n_heads, seq_len, 2)
        real = components[..., 0]
        imag = components[..., 1]

        # Clamp values to prevent extreme gradients in backward pass
        real = torch.clamp(real, min=-10, max=10)
        imag = torch.clamp(imag, min=-10, max=10)

        # Calculate amplitude (magnitude in complex plane)
        # FP16-SAFE: Use 1.0 eps (not 0.1) - sqrt gradient = 1/(2*sqrt(x)) explodes near 0
        A = torch.sqrt(real**2 + imag**2 + 1.0)

        # Calculate phase angle using stable formulation
        # FP16-SAFE: Use 0.1 offset (not 1e-4) - atan2 gradient unstable near (0,0)
        theta = torch.atan2(imag + 0.1, real + 0.1)  # Range: [-pi, pi]

        # Clamp outputs to reasonable ranges
        A = torch.clamp(A, min=0.1, max=10.0)
        theta = torch.clamp(theta, min=-3.14, max=3.14)

        return A, theta


class GeometricAttention(nn.Module):
    """
    Geometric attention layer replacing standard scaled dot-product attention.

    Standard transformer attention:
        scores = (Q @ K^T) / sqrt(d_k)
        weights = softmax(scores)
        output = weights @ V

    Geometric attention:
        scores = geometric_score(Q, K, T_field)  # wedge + tensor + spinor
        weights = softmax(scores / temperature)
        output = weights @ V

    The geometric score captures:
    - Wedge: Antisymmetric "repulsion" or "new information span"
    - Tensor: Full correlation structure (no projection loss)
    - Spinor: Rotational features from field topology
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,  # Fixed: Wedge/Tensor/Spinor/Hodge
        dropout: float = 0.1,
        geometric_weights: tuple = (1.0, 1.0, 1.0, 1.0),  # (wedge, tensor, spinor, hodge)
        temperature_scale: float = 1.0,
        use_exponential_form: bool = True  # Option 3: triality exponential
    ):
        """
        Initialize geometric attention.

        Args:
            d_model: Model dimension (4×32D after holomorphic contraction)
            n_heads: Number of attention heads (fixed at 4 independent operators)
            dropout: Dropout rate
            geometric_weights: (w_wedge, w_tensor, w_spinor, w_hodge) combination
            temperature_scale: Scaling for softmax temperature
            use_exponential_form: If True, use Option 3 ExponentialPhaseExtractor
                                  If False, use legacy PhaseExtractor (sqrt/atan2)

        Note: The 4 heads correspond to 4 independent 32D spaces, NOT a unified 128D space.
        Each 32D = 4 geometric ops × 8 octonions.

        The 4 operators complete the information generation cycle:
        - Wedge (∧): rank-1 → rank-2 antisymmetric
        - Tensor (⊗): rank-1 → rank-2 full correlation
        - Spinor: operations within rank-2 causal space
        - Hodge (⋆): rank-2 → rank-1 conservation/contraction
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 32D = 4 ops × 8 octonions
        self.use_exponential_form = use_exponential_form

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % 4 == 0, "n_heads must be a multiple of 4 (Wedge/Tensor/Spinor/Hodge blocks)"


        # Query projection (from input tokens)
        self.W_q = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable geometric weights (initialized from input)
        self.geometric_weights = nn.Parameter(
            torch.tensor(geometric_weights, dtype=torch.float32)
        )

        # Temperature scaling
        self.temperature = nn.Parameter(
            torch.tensor(temperature_scale, dtype=torch.float32)
        )

        self.dropout = nn.Dropout(dropout)

        # Phase extractor: Option 3 (exponential) or legacy (sqrt/atan2)
        if use_exponential_form:
            self.phase_extractor = ExponentialPhaseExtractor(d_k=self.d_k)
        else:
            self.phase_extractor = PhaseExtractor(d_k=self.d_k)

    def get_generators_for_regularization(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> Optional[tuple]:
        """
        Extract raw A, B, Θ generators for band regularization.

        Only available when use_exponential_form=True.

        Args:
            Q: Query tensor (batch, n_heads, seq_len, d_k)
            K: Key tensor (batch, n_heads, seq_len, d_k)

        Returns:
            (A_Q, B_Q, Theta_Q, A_K, B_K, Theta_K) or None if legacy mode
        """
        if not self.use_exponential_form:
            return None

        A_Q, B_Q, Theta_Q = self.phase_extractor.get_generators(Q)
        A_K, B_K, Theta_K = self.phase_extractor.get_generators(K)
        return A_Q, B_Q, Theta_Q, A_K, B_K, Theta_K

    @track_first_call
    def forward(
        self,
        Q_input: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        T_field: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of geometric attention.

        Args:
            Q_input: Query input (batch, seq_len_q, d_model)
            K: Keys from field (batch, seq_len_k, d_model)
            V: Values from field (batch, seq_len_k, d_model)
            T_field: Cognitive tensor field (N_x, N_y, D, D) complex
            mask: Optional attention mask (batch, seq_len_q, seq_len_k)

        Returns:
            output: Attention output (batch, seq_len_q, d_model)
            attention_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q_input.shape[0]
        seq_len_q = Q_input.shape[1]
        seq_len_k = K.shape[1]

        # Ensure dtype consistency for FP16 training
        target_dtype = self.W_q.weight.dtype
        Q_input = Q_input.to(target_dtype)
        K = K.to(target_dtype)
        V = V.to(target_dtype)

        # Project queries
        Q = self.W_q(Q_input)  # (batch, seq_len_q, d_model)

        # Split into heads
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, n_heads, seq_len, d_k)

        # Phase-based parallel computation across all 4 heads
        if self.use_exponential_form:
            # Option 3: 16D biquaternion via triality exponential
            psi_Q = self.phase_extractor(Q)  # (batch, 4, seq_q, 16)
            psi_K = self.phase_extractor(K)  # (batch, 4, seq_k, 16)

            # Compute geometric attention scores using exponential form
            attention_scores = geometric_score_from_exponential(
                psi_Q,
                psi_K,
                T_field,
                weights=tuple(self.geometric_weights)
            )
        else:
            # Legacy: scalar (A, theta) pairs via sqrt/atan2
            A_Q, theta_Q = self.phase_extractor(Q)  # (batch, 4, seq_q)
            A_K, theta_K = self.phase_extractor(K)  # (batch, 4, seq_k)

            # Compute geometric attention scores in parallel
            attention_scores = geometric_score_from_phase(
                A_Q,
                theta_Q,
                A_K,
                theta_K,
                T_field,
                weights=tuple(self.geometric_weights)
            )
        # Output: (batch, 4, seq_len_q, seq_len_k)

        # Scale by temperature
        attention_scores = attention_scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask_expanded = mask[:, None, None, :]          # (B,1,1,seq_k)
            elif mask.dim() == 3:
                mask_expanded = mask[:, None, :, :]             # (B,1,seq_q,seq_k)
            elif mask.dim() == 4:
                mask_expanded = mask                            # (B,H,seq_q,seq_k) or (B,1,seq_q,seq_k)
            else:
                raise ValueError(f"Unsupported attention mask shape: {mask.shape}")

            mask_bool = mask_expanded.to(torch.bool)
            neg = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(~mask_bool, neg)

        # Softmax normalization (THIS IS THE KEY STEP)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (batch, n_heads, seq_len_q, seq_len_k) @ (batch, n_heads, seq_len_k, d_k)
        # -> (batch, n_heads, seq_len_q, d_k)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.d_model)

        # Final projection
        output = self.W_o(output)

        return output, attention_weights


class GeometricTransformer(nn.Module):
    """
    Complete geometric transformer that integrates field evolution with attention.

    This is the full cognitive architecture:
    1. T_ij field provides memory (evolved via PDE)
    2. Field state → K, V via projection
    3. User prompt → Q via embedding
    4. Geometric attention(Q, K, V, T_field) produces output
    5. Output → tokens via decoder

    Usage:
        # Initialize
        field = CognitiveTensorField(config)
        transformer = GeometricTransformer(field_dim=4, d_model=512)

        # Evolve field (background "unconscious" process)
        for _ in range(100):
            field.evolve_step()

        # Query field (foreground "conscious" process)
        prompt_tokens = embed_prompt("Why is decision-making hard?")
        output = transformer(prompt_tokens, field.T)
    """

    def __init__(
        self,
        field_dim: int,      # D (dimension of T_ij tensor)
        d_model: int = 512,  # Transformer hidden dimension
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        geometric_weights: tuple = (1.0, 1.0, 1.0, 1.0),
        use_positional_encoding: bool = True,
        use_temporal_encoding: bool = False
    ):
        """
        Initialize geometric transformer.

        Args:
            field_dim: Dimension D of T_ij field (e.g., 4)
            d_model: Transformer model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            geometric_weights: (w_wedge, w_tensor, w_spinor, w_hodge)
            use_positional_encoding: Add spatial position info
            use_temporal_encoding: Add temporal evolution info
        """
        super().__init__()

        self.field_dim = field_dim
        self.d_model = d_model

        # Field state → K, V projection
        self.field_to_kv = FieldToKeyValue(
            field_dim=field_dim,
            d_model=d_model,
            use_positional_encoding=use_positional_encoding,
            use_temporal_encoding=use_temporal_encoding
        )

        # Geometric attention layers
        self.layers = nn.ModuleList([
            GeometricTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                geometric_weights=geometric_weights
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        Q_input: torch.Tensor,
        T_field: torch.Tensor,
        time: float = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass: query the field with input tokens.

        Args:
            Q_input: Query input (batch, seq_len, d_model)
            T_field: Cognitive tensor field (N_x, N_y, D, D) complex
            time: Optional time for temporal encoding
            mask: Optional attention mask

        Returns:
            output: Transformer output (batch, seq_len, d_model)
            attention_weights_list: List of attention weights from each layer
        """
        # Extract K, V from field state
        K, V = self.field_to_kv(T_field, time=time)

        # Add batch dimension if needed
        if K.dim() == 2:
            K = K.unsqueeze(0)  # (1, N_tokens, d_model)
            V = V.unsqueeze(0)

        # Repeat K, V for batch if needed
        batch_size = Q_input.shape[0]
        if K.shape[0] == 1 and batch_size > 1:
            K = K.repeat(batch_size, 1, 1)
            V = V.repeat(batch_size, 1, 1)

        # Apply transformer layers
        output = Q_input
        attention_weights_list = []

        for layer in self.layers:
            output, attn_weights = layer(output, K, V, T_field, mask)
            attention_weights_list.append(attn_weights)

        output = self.norm(output)

        return output, attention_weights_list


class GeometricTransformerLayer(nn.Module):
    """
    Single transformer layer with geometric attention.

    Architecture:
    1. Geometric multi-head attention
    2. Add & norm
    3. Feedforward network
    4. Add & norm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,  # Deprecated, use expansion_factor
        dropout: float = 0.0,
        geometric_weights: tuple = (1.0, 1.0, 1.0),
        ffn_activation: str = 'swiglu',
        expansion_factor: float = None
    ):
        super().__init__()

        self.attention = GeometricAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            geometric_weights=geometric_weights
        )

        # FFN (SwiGLU default)
        from bayesian_cognitive_field.models.activations import FFN
        self.ff = FFN(
            d_model=d_model,
            expansion_factor=expansion_factor,
            activation=ffn_activation,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q_input: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        T_field: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer layer.

        Args:
            Q_input: Input (batch, seq_len, d_model)
            K, V: Keys and values from field
            T_field: Cognitive tensor field
            mask: Optional attention mask

        Returns:
            output: Layer output (batch, seq_len, d_model)
            attention_weights: Attention weights
        """
        # Geometric attention
        attn_output, attn_weights = self.attention(Q_input, K, V, T_field, mask)

        # Add & norm
        Q_input = self.norm1(Q_input + self.dropout(attn_output))

        # Feedforward
        ff_output = self.ff(Q_input)

        # Add & norm
        output = self.norm2(Q_input + self.dropout(ff_output))

        return output, attn_weights
