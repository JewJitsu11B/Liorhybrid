"""
Full Geometric Stack: CausalField → SBERT Pooling → DPR → Geometric Attention

This integrates all geometric components into a unified O(N log N) architecture:

Architecture Flow:
    Input (text/image/video)
         ↓
    Multimodal Embedding
         ↓
    CausalField Encoder (O(N log N)) ← FFT-based parallel convolution
         ↓
    SBERT-style Pooling (mean/max/attention-weighted)
         ↓
    DPR K/V Generation ← Statistically optimal embeddings
         ↓
    Geometric Attention ← Field-contracted wedge/tensor/spinor products
         ↓
    Output Projection

Key innovations:
1. O(N log N) base processing via CausalField (FFT convolution)
2. Geometric operators (Trinor/Wedge/Spinor) replace matrix operations
3. Field contractions maintain memory efficiency
4. Complex metric G = A + iB with phase orthogonality

Complexity:
- CausalField encoder: O(N log N) parallel
- SBERT pooling: O(N)
- DPR: O(1) per position
- Geometric attention: O(N²) but with field contractions → manageable
- Total: Dominated by O(N log N) CausalField

This is the target architecture for the Bayesian Cognitive Field system.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .dpr_encoder import DPRKeyValueGenerator
from .geometric_attention import GeometricAttention
from .geometric_products import geometric_score
from ..models.causal_field import CausalFieldBlock, BiQuatCausalBlock


class SBERTPooling(nn.Module):
    """
    SBERT-style pooling layer.

    Pools sequence embeddings to fixed-size representation using:
    1. Mean pooling (default)
    2. Max pooling
    3. Attention-weighted pooling

    This is NOT the full O(N²) BERT attention layer - just the pooling
    mechanism from Sentence-BERT for creating sequence-level embeddings.
    """

    def __init__(
        self,
        d_model: int = 512,
        pooling_mode: str = 'mean'  # 'mean', 'max', 'attention'
    ):
        super().__init__()

        self.d_model = d_model
        self.pooling_mode = pooling_mode

        if pooling_mode == 'attention':
            # Learnable attention weights for pooling
            self.attention = nn.Linear(d_model, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence to single vector.

        Args:
            embeddings: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) optional mask

        Returns:
            pooled: (batch, d_model)
        """
        if attention_mask is None:
            attention_mask = torch.ones(
                embeddings.shape[0],
                embeddings.shape[1],
                device=embeddings.device
            )

        if self.pooling_mode == 'mean':
            # Mean pooling with mask
            mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
            sum_embeddings = (embeddings * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            pooled = sum_embeddings / (sum_mask + 1e-9)

        elif self.pooling_mode == 'max':
            # Max pooling with mask
            mask_expanded = attention_mask.unsqueeze(-1)
            embeddings_masked = embeddings.masked_fill(mask_expanded == 0, -1e9)
            pooled = embeddings_masked.max(dim=1)[0]

        elif self.pooling_mode == 'attention':
            # Attention-weighted pooling
            attn_weights = self.attention(embeddings)  # (batch, seq, 1)
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = (embeddings * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return pooled


class GeometricStack(nn.Module):
    """
    Full geometric processing stack.

    Integrates:
    1. CausalField encoder (O(N log N) parallel)
    2. SBERT pooling (O(N) aggregation)
    3. DPR K/V generation (statistical optimization)
    4. Geometric attention (field-contracted products)

    This replaces the standard transformer with a fully geometric,
    causally-structured, O(N log N)-dominated architecture.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 4,
        n_attention_layers: int = 2,
        n_heads: int = 8,
        field_dim: int = 4,
        use_dpr: bool = True,
        dpr_use_pretrained: bool = True,
        dpr_trainable_seeds: bool = False,
        pooling_mode: str = 'mean',
        timing_debug: bool = False,
        ffn_activation: str = 'swiglu',
        ffn_expansion_factor: float = None,
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_attention_layers = n_attention_layers
        self.use_dpr = use_dpr
        self.dpr_use_pretrained = dpr_use_pretrained
        self.dpr_trainable_seeds = dpr_trainable_seeds
        self.timing_debug = timing_debug

        # Layer 1: BiQuaternion causal encoder (O(N) pure real, no torch.complex)
        # Replaces cubic O(d^3) octonionic ops with structured quaternion matmuls
        # NOTE: use_attention=False here - attention is handled by GeometricAttention later
        #       Having O(N^2) attention in every causal layer was causing 12s+ batch times
        self.causal_field_layers = nn.ModuleList([
            BiQuatCausalBlock(
                d_model=d_model,
                d_field=16,          # 16D = 4 real quaternions (Q_M_re, Q_M_im, Q_H_re, Q_H_im)
                n_heads=n_heads,
                expand_factor=ffn_expansion_factor,  # None -> SwiGLU default 8/3
                use_attention=False, # O(N^2) attention disabled - use GeometricAttention instead
                alpha=0.5,           # Causal mixing: 0=all memory, 1=all present
                detach_memory=True,  # O(1) memory per node, no BPTT
                ffn_activation=ffn_activation,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Layer 1.5: Holomorphic contraction (λ = 1/2)
        # Projects 256D embeddings → 4 independent 32D spaces
        # Each 32D space = 4 geometric ops × 8 octonions
        # The 4 spaces correspond to the 4 causal operators (Wedge/Tensor/Spinor/Hodge)
        self.holomorphic_contract = nn.Linear(d_model, d_model // 2, bias=False)

        # Initialize with orthogonal matrix for stable gradients
        # Orthogonal init preserves gradient norms through contraction
        nn.init.orthogonal_(self.holomorphic_contract.weight)

        # Layer 2: SBERT pooling (O(N))
        self.pooling = SBERTPooling(
            d_model=d_model // 2,  # Operates on 4×32D
            pooling_mode=pooling_mode
        )

        # Layer 3: DPR K/V generation (optional)
        if use_dpr:
            self.dpr = DPRKeyValueGenerator(
                d_model=d_model // 2,  # Operates on 4×32D
                freeze_encoders=True,
                use_pretrained=dpr_use_pretrained
            )
        else:
            self.dpr = None

        # Layer 4: Geometric attention layers
        # Fixed 4 heads corresponding to 4 independent 32D spaces
        # Each head: 32D = 4 geometric ops × 8 octonions
        # Operators: Wedge (∧), Tensor (⊗), Spinor, Hodge (⋆)
        # These complete the information generation cycle: rank-1 → rank-2 → rank-1
        self.geometric_attention = nn.ModuleList([
            GeometricAttention(
                d_model=d_model // 2,  # Input: 4×32D
                n_heads=4,              # 4 independent operators
                dropout=0.1,
                geometric_weights=(1.0, 1.0, 1.0, 1.0)  # wedge, tensor, spinor, hodge
            )
            for _ in range(n_attention_layers)
        ])

        # Output projection (back to full d_model for compatibility)
        self.output_projection = nn.Linear(d_model // 2, d_model)

        # DPR-seeded K/V initialization (must happen before optimizer construction).
        # Default: frozen seed buffers + trainable deltas (ΔK/ΔV) for stability.
        self.register_buffer("K_seed", torch.empty(0), persistent=True)
        self.register_buffer("V_seed", torch.empty(0), persistent=True)
        self.K_delta = None
        self.V_delta = None

        # Optional mode: trainable seeds directly (enabled via dpr_trainable_seeds=True)
        self.K_learned = None
        self.V_learned = None

    def set_timing_debug(self, enabled: bool):
        """Enable or disable timing debug for the entire stack."""
        self.timing_debug = enabled

    @torch.compiler.disable
    def initialize_kv_from_field(self, field_state: torch.Tensor, *, force: bool = False) -> None:
        """
        Initialize DPR-seeded K/V outside the training forward path.

        Call this before optimizer construction so K/V params are included.
        """
        if self.dpr is None:
            return

        if not force:
            if self.dpr_trainable_seeds and self.K_learned is not None and self.V_learned is not None:
                return
            if (not self.dpr_trainable_seeds) and self.K_seed.numel() > 0 and self.K_delta is not None:
                return

        with torch.no_grad():
            K_seed, V_seed = self.dpr.generate_kv(field_state=field_state, batch_size=1)

        ref = next(self.parameters())
        K_seed = K_seed.to(device=ref.device, dtype=ref.dtype).contiguous()
        V_seed = V_seed.to(device=ref.device, dtype=ref.dtype).contiguous()

        K_seed_2d = K_seed.squeeze(0).contiguous()
        V_seed_2d = V_seed.squeeze(0).contiguous()

        if self.dpr_trainable_seeds:
            if self.K_learned is None:
                self.K_learned = nn.Parameter(K_seed_2d.clone())
                self.V_learned = nn.Parameter(V_seed_2d.clone())
            else:
                if self.K_learned.shape != K_seed_2d.shape or self.V_learned.shape != V_seed_2d.shape:
                    raise RuntimeError(
                        f"DPR seed shape changed: K {tuple(self.K_learned.shape)} -> {tuple(K_seed_2d.shape)}, "
                        f"V {tuple(self.V_learned.shape)} -> {tuple(V_seed_2d.shape)}. Restart training."
                    )
                self.K_learned.data.copy_(K_seed_2d)
                self.V_learned.data.copy_(V_seed_2d)
            # Clear seed+delta mode state
            self.K_seed.resize_(0)
            self.V_seed.resize_(0)
            self.K_delta = None
            self.V_delta = None
            return

        # Seed+delta mode: update buffers in-place (do not rebind registered buffers)
        if self.K_seed.numel() == 0:
            self.K_seed.resize_as_(K_seed_2d).copy_(K_seed_2d)
            self.V_seed.resize_as_(V_seed_2d).copy_(V_seed_2d)
        else:
            if self.K_seed.shape != K_seed_2d.shape or self.V_seed.shape != V_seed_2d.shape:
                raise RuntimeError(
                    f"DPR seed shape changed: K {tuple(self.K_seed.shape)} -> {tuple(K_seed_2d.shape)}, "
                    f"V {tuple(self.V_seed.shape)} -> {tuple(V_seed_2d.shape)}. Restart training."
                )
            self.K_seed.copy_(K_seed_2d)
            self.V_seed.copy_(V_seed_2d)

        if self.K_delta is None or self.V_delta is None:
            self.K_delta = nn.Parameter(torch.zeros_like(self.K_seed))
            self.V_delta = nn.Parameter(torch.zeros_like(self.V_seed))
        else:
            if self.K_delta.shape != self.K_seed.shape or self.V_delta.shape != self.V_seed.shape:
                raise RuntimeError(
                    f"DPR delta shape mismatch: ΔK {tuple(self.K_delta.shape)} vs K_seed {tuple(self.K_seed.shape)}, "
                    f"ΔV {tuple(self.V_delta.shape)} vs V_seed {tuple(self.V_seed.shape)}. Restart training."
                )
            # Fresh epoch init semantics: keep anchor, zero deltas
            self.K_delta.data.zero_()
            self.V_delta.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        field_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, None]:
        """
        Full forward pass through geometric stack.

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            field_state: Cognitive tensor field (N_x, N_y, D, D)
            attention_mask: Optional mask (batch, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
            None: (for backwards compatibility)
        """
        import time

        if self.timing_debug:
            t0 = time.perf_counter()

        batch_size, seq_len, _ = x.shape

        # Defensive check: enforce max sequence length
        if seq_len > 512:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum 512. "
                f"This prevents O(L^2) attention from consuming excessive memory. "
                f"Truncate input sequences in your dataloader."
            )

        # Step 1: CausalField encoding (O(N log N) parallel via FFT)
        if self.timing_debug:
            t_causal_start = time.perf_counter()

        memory = None
        encoder_output = x
        for i, layer in enumerate(self.causal_field_layers):
            # Only diagnose first layer to avoid spam
            layer_diagnose = diagnose and (i == 0)
            encoder_output, memory = layer(encoder_output, memory, diagnose=layer_diagnose)

        if self.timing_debug:
            timing_causal = time.perf_counter() - t_causal_start

        # Step 1.5: Holomorphic contraction
        if self.timing_debug:
            t_contract_start = time.perf_counter()
        encoder_output = self.holomorphic_contract(encoder_output)  # (batch, seq_len, 4x32)
        if self.timing_debug:
            timing_contract = time.perf_counter() - t_contract_start

        # Step 2: SBERT pooling (O(N))
        # Create sequence-level representation
        if self.timing_debug:
            t_pool_start = time.perf_counter()
        pooled = self.pooling(encoder_output, attention_mask)  # (batch, 4x32)
        if self.timing_debug:
            timing_pool = time.perf_counter() - t_pool_start

        # Step 3: DPR K/V generation (if enabled)
        # ASYMMETRIC RETRIEVAL: K/V are learnable parameters initialized from field
        if self.timing_debug:
            t_dpr_start = time.perf_counter()
        if self.dpr is not None:
            if self.dpr_trainable_seeds:
                if self.K_learned is None or self.V_learned is None:
                    raise RuntimeError(
                        "DPR K/V seeds not initialized. Call `model.geometric_stack.initialize_kv_from_field(field.T)` "
                        "before constructing the optimizer/starting training."
                    )
                K_base = self.K_learned
                V_base = self.V_learned
            else:
                if self.K_seed.numel() == 0 or self.K_delta is None or self.V_delta is None:
                    raise RuntimeError(
                        "DPR K/V seeds not initialized. Call `model.geometric_stack.initialize_kv_from_field(field.T)` "
                        "before constructing the optimizer/starting training."
                    )
                if self.K_delta.shape != self.K_seed.shape or self.V_delta.shape != self.V_seed.shape:
                    raise RuntimeError(
                        "DPR delta/seed shape mismatch. Restart training to reinitialize optimizer state."
                    )
                K_base = self.K_seed + self.K_delta
                V_base = self.V_seed + self.V_delta

            # Use learned K/V (these get gradient updates during training)
            K = K_base.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_tokens, d_model)
            V = V_base.unsqueeze(0).expand(batch_size, -1, -1)

            # Use encoder output as queries
            Q = encoder_output
        else:
            # Standard Q=K=V from encoder output
            Q = K = V = encoder_output
        if self.timing_debug:
            timing_dpr = time.perf_counter() - t_dpr_start

        # Step 4: Geometric attention (field-contracted)
        # Apply geometric products with field contractions
        if self.timing_debug:
            t_attn_start = time.perf_counter()

        if diagnose:
            print(f"\n[GeometricStack] Before geometric_attention:")
            print(f"  Q: nan={Q.isnan().any().item()}, inf={Q.isinf().any().item()}, "
                  f"range=[{Q.min().item():.4g}, {Q.max().item():.4g}]")
            print(f"  K: nan={K.isnan().any().item()}, inf={K.isinf().any().item()}, "
                  f"range=[{K.min().item():.4g}, {K.max().item():.4g}]")
            print(f"  V: nan={V.isnan().any().item()}, inf={V.isinf().any().item()}, "
                  f"range=[{V.min().item():.4g}, {V.max().item():.4g}]")

        attn_output = Q

        for i, attn_layer in enumerate(self.geometric_attention):
            attn_output, _ = attn_layer(
                Q_input=attn_output,
                K=K,
                V=V,
                T_field=field_state,
                mask=attention_mask
            )
        if self.timing_debug:
            timing_attn = time.perf_counter() - t_attn_start

        # Step 5: Output projection
        if self.timing_debug:
            t_out_start = time.perf_counter()
        output = self.output_projection(attn_output)
        if self.timing_debug:
            timing_out = time.perf_counter() - t_out_start

        if self.timing_debug:
            total_time = time.perf_counter() - t0
            print(f"\n[GeometricStack] batch={batch_size}, seq_len={seq_len}, total={total_time:.3f}s")
            print(f"  causal_field={timing_causal:.4f}s ({timing_causal/total_time*100:.1f}%)")
            print(f"  contract={timing_contract:.4f}s, pool={timing_pool:.4f}s")
            print(f"  dpr={timing_dpr:.4f}s, attn={timing_attn:.4f}s, out_proj={timing_out:.4f}s")

        return output, None

    def reset_kv_from_field(self, field_state: torch.Tensor):
        """
        Reset K/V learned parameters from current field state.

        Call this at the START of each epoch to reinitialize K/V based on
        the evolved field state. The parameters will then be fine-tuned
        during that epoch.

        Args:
            field_state: Current cognitive tensor field
        """
        if self.dpr is None:
            return

        # Must be initialized before optimizer construction. Do not create new Parameters here.
        if self.dpr_trainable_seeds:
            if self.K_learned is None or self.V_learned is None:
                raise RuntimeError(
                    "DPR K/V seeds not initialized. Call `model.geometric_stack.initialize_kv_from_field(field.T)` "
                    "before constructing the optimizer/starting training."
                )
        else:
            if self.K_seed.numel() == 0 or self.K_delta is None or self.V_delta is None:
                raise RuntimeError(
                    "DPR K/V seeds not initialized. Call `model.geometric_stack.initialize_kv_from_field(field.T)` "
                    "before constructing the optimizer/starting training."
                )

        self.initialize_kv_from_field(field_state, force=True)


class GeometricTransformerWithMamba(nn.Module):
    """
    Complete geometric transformer with CausalField base.

    Uses CausalField for O(N log N) parallel processing via FFT convolution.

    Usage:
        model = GeometricTransformerWithMamba(
            d_model=512,
            n_layers=4,
            n_attention_layers=2,
            field_dim=4
        )

        output = model(
            input_embeddings,
            field_state,
            time=field.t
        )
    """

    def __init__(
        self,
        d_model: int = 512,
        n_mamba_layers: int = 4,  # Keep param name for backwards compatibility
        n_attention_layers: int = 2,
        n_heads: int = 8,
        field_dim: int = 4,
        use_dpr: bool = True,
        dpr_use_pretrained: bool = True,
        dpr_trainable_seeds: bool = False,
        use_positional_encoding: bool = True,
        use_temporal_encoding: bool = True,
        timing_debug: bool = False,
        **kwargs  # Absorb any legacy params like use_causal_field
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_mamba_layers  # Store as n_layers
        self.n_attention_layers = n_attention_layers
        self.use_positional_encoding = use_positional_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.timing_debug = timing_debug

        # Positional encoding
        if use_positional_encoding:
            pe = self._create_positional_encoding(2048, d_model)
            self.register_buffer("pos_encoding", pe, persistent=False)
        else:
            self.pos_encoding = None

        # Temporal encoding
        if use_temporal_encoding:
            self.temporal_projection = nn.Linear(1, d_model)
        else:
            self.temporal_projection = None

        # Core geometric stack (uses CausalField)
        self.geometric_stack = GeometricStack(
            d_model=d_model,
            n_layers=n_mamba_layers,
            n_attention_layers=n_attention_layers,
            n_heads=n_heads,
            field_dim=field_dim,
            use_dpr=use_dpr,
            dpr_use_pretrained=dpr_use_pretrained,
            dpr_trainable_seeds=dpr_trainable_seeds,
            timing_debug=timing_debug
        )

        # Output head (for language modeling, etc.)
        self.lm_head = None  # Can be added later

    def set_timing_debug(self, enabled: bool):
        """Enable or disable timing debug for the entire model."""
        self.timing_debug = enabled
        self.geometric_stack.set_timing_debug(enabled)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor,
        field_state: torch.Tensor,
        time: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through geometric transformer with CausalField.

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            field_state: Cognitive tensor field (N_x, N_y, D, D)
            time: Current field time (for temporal encoding)
            attention_mask: Optional mask (batch, seq_len)
            diagnose: If True, print diagnostic info about NaN/Inf

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: None (CausalField uses FFT convolution, not attention)
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding (device-stable for torch.compile)
        if self.use_positional_encoding and self.pos_encoding is not None:
            pos_enc = self.pos_encoding[:seq_len].to(dtype=x.dtype)
            x = x + pos_enc.unsqueeze(0)

        # Add temporal encoding
        if self.use_temporal_encoding and time is not None:
            time_tensor = torch.tensor([[time]], dtype=x.dtype, device=x.device)
            time_enc = self.temporal_projection(time_tensor)  # (1, d_model)
            x = x + time_enc.unsqueeze(1)

        # Process through geometric stack
        output, _ = self.geometric_stack(
            x=x,
            field_state=field_state,
            attention_mask=attention_mask,
            diagnose=diagnose
        )

        # CausalField doesn't produce traditional attention weights
        attention_weights = None

        return output, attention_weights


def demonstrate_geometric_operators():
    """
    Demonstration of how geometric operators replace matrix operations.

    Standard Mamba:
        h_t = A @ h_{t-1} + B @ x_t
        y_t = C @ h_t

    Geometric Mamba:
        h_t = Trinor(h_{t-1}) ⊗ Wedge(x_t)
        y_t = Spinor(h_t)

    This shows the exact correspondence and how the geometric algebra
    enforces causal structure through the mathematics itself.
    """
    print("=" * 80)
    print("Geometric Operators vs Matrix Operations")
    print("=" * 80)

    print("\nStandard Mamba SSM:")
    print("  h_t = A @ h_{t-1} + B @ x_t")
    print("  where:")
    print("    A: (d_state, d_state) transition matrix")
    print("    B: (d_state, d_input) input projection")
    print("    C: (d_output, d_state) output projection")
    print("  Properties:")
    print("    - Linear dynamics")
    print("    - Associative: (AB)C = A(BC)")
    print("    - No inherent causal structure")

    print("\n" + "-" * 80)

    print("\nGeometric Mamba with CI8:")
    print("  h_t = Trinor(h_{t-1}) ⊗ Wedge(x_t)")
    print("  where:")
    print("    Trinor: Geometric evolution operator")
    print("      - Learned phase rotation θ")
    print("      - Learned rotation axis ω")
    print("      - Learned scaling σ")
    print("    Wedge: Antisymmetric projection")
    print("      - Creates orthogonal coupling")
    print("      - Enforces causal divergence")
    print("    ⊗: Octonion multiplication")
    print("      - Non-associative: (ab)c ≠ a(bc)")
    print("      - Path-dependent dynamics")
    print("    Spinor: Rotational invariants")
    print("      - Extracts stable features")
    print("      - Phase-independent output")

    print("\n" + "-" * 80)

    print("\nKey Advantages:")
    print("  1. O(N) complexity maintained")
    print("  2. Causal structure enforced by algebra")
    print("  3. Non-associative → path-dependent learning")
    print("  4. Geometric operators are interpretable")
    print("  5. Field state integration via CI8")

    print("=" * 80)


if __name__ == "__main__":
    # Demonstrate the geometric operator correspondence
    demonstrate_geometric_operators()
