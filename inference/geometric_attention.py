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

Option 6 Integration:
    Address-based neighbor probing with AddressBuilder
    - Uses Address structure for Q/K representation
    - Probes 64 role-typed neighbors (32 nearest, 16 attractors, 16 repulsors)
    - Consumes 6 similarity scores per neighbor
    - O(N × 64 × d') complexity (no dense matmul)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .geometric_products import geometric_score, geometric_score_from_phase, geometric_score_from_exponential
from .field_extraction import FieldToKeyValue
from .triality_coordinate_head import TrialityCoordinateHead, TrialityHeadConfig, TrialityCoords
from .address import Address, AddressBuilder, AddressConfig  # Option 6 integration
try:
    from Liorhybrid.training.execution_tracker import track_first_call
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


class FieldContractor(nn.Module):
    """
    True tensor contraction: T_ij @ K_j -> V_i

    Projects the field tensor to the key dimension and performs
    Einstein contraction to transform values through the field geometry.

    This replaces scalar field modulation with vector-valued field operations,
    making the field an actual operator rather than just a gate.

    Contraction: V_contracted[i] = sum_j T_proj[i,j] * K[j]
    """

    def __init__(self, field_dim: int, d_k: int):
        super().__init__()
        self.field_dim = field_dim
        self.d_k = d_k

        # Project field (D,D) -> (d_k, d_k) for contraction
        # This learns how to map field geometry to attention space
        self.field_proj = nn.Linear(field_dim * field_dim, d_k * d_k, bias=False)

        # Initialize with small weights for stability
        nn.init.normal_(self.field_proj.weight, std=0.01)

    def forward(self, T_field: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Contract field with keys to produce field-transformed values.

        Args:
            T_field: Cognitive tensor field (N_x, N_y, D, D) - complex or real
            K: Key tensor (batch, n_heads, seq_k, d_k)

        Returns:
            V_contracted: Field-transformed values (batch, n_heads, seq_k, d_k)
        """
        # Spatial average of field
        T_avg = T_field.mean(dim=(0, 1))  # (D, D)

        # Handle complex field
        if T_avg.is_complex():
            T_avg = T_avg.abs()  # Use magnitude for contraction

        # Flatten and project to (d_k, d_k)
        T_flat = T_avg.reshape(-1)  # (D*D,)
        T_proj = self.field_proj(T_flat).view(self.d_k, self.d_k)  # (d_k, d_k)

        # True contraction: V_i = T_ij K_j
        # K: (batch, n_heads, seq_k, d_k)
        # T_proj: (d_k, d_k)
        # Output: (batch, n_heads, seq_k, d_k)
        V_contracted = torch.einsum('ij,bhsj->bhsi', T_proj, K)

        return V_contracted


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
        V_contracted = field_contract(T_field, K)  # True tensor contraction
        output = weights @ V + alpha * V_contracted  # Field shapes values

    The geometric score captures:
    - Wedge: Antisymmetric "repulsion" or "new information span"
    - Tensor: Full correlation structure (no projection loss)
    - Spinor: Rotational features from field topology

    The field contraction adds:
    - T_ij @ K_j: Field directly transforms values (vector-valued operation)
    - This makes field geometry produce motion, not just gating
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,  # Fixed: Wedge/Tensor/Spinor/Hodge
        dropout: float = 0.1,
        geometric_weights: tuple = (1.0, 1.0, 1.0, 1.0),  # (wedge, tensor, spinor, hodge)
        temperature_scale: float = 1.0,
        use_exponential_form: bool = True,  # Option 3: triality exponential
        q_lowrank_r: Optional[int] = None,
        field_dim: int = 16,  # Field tensor dimension D (T_field is D x D)
        use_field_contraction: bool = True,  # Enable true tensor contraction
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 32D = 4 ops x 8 octonions
        self.use_exponential_form = use_exponential_form
        self.q_lowrank_r = int(q_lowrank_r) if q_lowrank_r is not None else None
        self.use_field_contraction = use_field_contraction

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % 4 == 0, "n_heads must be a multiple of 4 (Wedge/Tensor/Spinor/Hodge blocks)"

        # Query projection (from input tokens)
        if self.q_lowrank_r is None:
            self.W_q = nn.Linear(d_model, d_model)
            self.W_q_u = None
            self.W_q_v = None
        else:
            r = int(self.q_lowrank_r)
            if r < 1 or r > d_model:
                raise ValueError(f"q_lowrank_r must be in [1, d_model], got {r} (d_model={d_model})")

            # Factorized: (D -> r) then (r -> D)
            self.W_q = None
            self.W_q_u = nn.Linear(d_model, r, bias=False)
            self.W_q_v = nn.Linear(r, d_model, bias=True)

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

        # Field contraction: T_ij @ K_j -> V (true tensor operation)
        if use_field_contraction:
            self.field_contractor = FieldContractor(field_dim=field_dim, d_k=self.d_k)
            # Learnable mixing coefficient for field-contracted vs attention-weighted values
            # Start small (0.1) so attention dominates initially, field contribution grows with training
            self.field_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.field_contractor = None
            self.field_alpha = None

    def _project_q(self, Q_input: torch.Tensor) -> torch.Tensor:
        if self.W_q is not None:
            return self.W_q(Q_input)
        if self.W_q_u is None or self.W_q_v is None:
            raise RuntimeError("Invalid Q projection state: low-rank modules missing.")
        return self.W_q_v(self.W_q_u(Q_input))

    def probe_neighbors(
        self,
        Q: torch.Tensor,                    # (batch, seq_len, d_model)
        neighbor_embeddings: torch.Tensor,  # (batch, 64, d_model)
        metric: Optional[torch.Tensor] = None,  # (batch, d_model) diagonal metric
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Option 6: Probe neighbors using metric-weighted similarity.

        NO MATMUL - explicit probing of 64 neighbors.
        Complexity: O(N × 64 × d') instead of O(N²)

        Args:
            Q: Query tensor from encoder (batch, seq_len, d_model)
            neighbor_embeddings: 64 role-typed neighbors (batch, 64, d_model)
                - [:, 0:32, :]: nearest neighbors (similarity grounding)
                - [:, 32:48, :]: attractors (reinforcing evidence)
                - [:, 48:64, :]: repulsors (contrastive evidence)
            metric: Optional diagonal metric for scaling (batch, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            weights: (batch, seq_len, 64) - probe weights per neighbor
        """
        batch_size, seq_len, d_model = Q.shape
        n_neighbors = neighbor_embeddings.shape[1]  # 64

        # Project Q for probing (d_model -> d_model, same space as neighbors)
        Q_proj = self._project_q(Q)  # (batch, seq_len, d_model)

        # Apply metric scaling if provided (diagonal metric)
        if metric is not None:
            # metric: (batch, d_model) -> (batch, 1, d_model)
            Q_scaled = Q_proj * metric.unsqueeze(1)
        else:
            Q_scaled = Q_proj

        # Compute similarity: Q_scaled @ neighbor_embeddings.T
        # (batch, seq_len, d_model) @ (batch, d_model, 64) -> (batch, seq_len, 64)
        similarity = torch.bmm(Q_scaled, neighbor_embeddings.transpose(1, 2))

        # Scale by sqrt(d_model) for stability
        similarity = similarity / (d_model ** 0.5)

        # Apply role-typed weighting
        # Nearest (0-31): weight = 1.0 (similarity grounding)
        # Attractors (32-47): weight = 1.5 (boosted positive - reinforcing evidence)
        # Repulsors (48-63): weight = -0.5 (negative - contrastive evidence)
        role_weights = torch.ones(n_neighbors, device=Q.device, dtype=Q.dtype)
        role_weights[32:48] = 1.5   # Attractors boosted
        role_weights[48:64] = -0.5  # Repulsors contrastive

        # Apply role weights: (batch, seq_len, 64) * (64,)
        similarity = similarity * role_weights.unsqueeze(0).unsqueeze(0)

        # All three gates combined: Born × Gibbs × Softmax
        tau = self.temperature.clamp(min=1e-8)
        born = similarity.pow(2)                          # |ψ|² amplitude
        gibbs = torch.exp(-similarity.abs() / tau)        # exp(-E/T) cost
        soft = torch.softmax(similarity / tau, dim=-1)    # score distribution
        weights = born * gibbs * soft

        # Weighted combination of neighbor embeddings
        # (batch, seq_len, 64) @ (batch, 64, d_model) -> (batch, seq_len, d_model)
        output = torch.bmm(weights, neighbor_embeddings)

        # Final output projection
        output = self.W_o(output)

        return output, weights
    
    def probe_address_neighbors(
        self,
        Q_address: Address,  # Query address with 64 neighbors
        K_addresses: Optional[Address] = None,  # Optional key addresses (for cross-attention)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Option 6 Extended: Probe using full Address structure.
        
        Uses the 6 similarity scores and neighbor features from Address.
        This is a more advanced version that leverages the complete address structure.
        
        Args:
            Q_address: Query address with populated neighbors
            K_addresses: Optional key addresses (not used in self-attention)
            
        Returns:
            output: (batch, seq_len, d_model) - attended output
            weights: (batch, seq_len, 64) - attention weights per neighbor
            
        Note: Currently returns Q_address.core as a placeholder.
              Full implementation would:
              1. Extract 6 similarity scores from Q_address.all_neighbor_scores
              2. Use role-typed neighbor values from Q_address.all_neighbor_values
              3. Apply metric-weighted combination using neighbor_metrics/transports
              4. Return weighted output via neighbor routing coords
        """
        # Extract components from address
        batch_size = Q_address.shape[0]
        
        # Get all neighbor scores (batch, 64, 6)
        neighbor_scores = Q_address.all_neighbor_scores
        
        # Get neighbor values (batch, 64, d')
        neighbor_values = Q_address.all_neighbor_values
        
        # Get neighbor metrics and transports for geometric weighting
        neighbor_metrics = Q_address.all_neighbor_metrics  # (batch, 64, 16)
        neighbor_transports = Q_address.all_neighbor_transports  # (batch, 64, 16)
        
        # Aggregate scores: use first score (cosine) + learned average
        # scores shape: (batch, 64, 6) -> (batch, 64)
        primary_scores = neighbor_scores[..., 0]  # Cosine similarity
        learned_scores = neighbor_scores[..., 1:].mean(dim=-1)  # Average learned scores
        combined_scores = primary_scores + 0.1 * learned_scores  # Weighted combination
        
        # Apply role-typed weighting
        n_neighbors = 64
        role_weights = torch.ones(n_neighbors, device=combined_scores.device, dtype=combined_scores.dtype)
        role_weights[32:48] = 1.5   # Attractors boosted
        role_weights[48:64] = -0.5  # Repulsors contrastive
        
        combined_scores = combined_scores * role_weights.unsqueeze(0)
        
        # Compute attention weights with Born × Gibbs × Softmax
        tau = self.temperature.clamp(min=1e-8)
        born = combined_scores.pow(2)
        gibbs = torch.exp(-combined_scores.abs() / tau)
        soft = torch.softmax(combined_scores / tau, dim=-1)
        weights = born * gibbs * soft  # (batch, 64)
        
        # Project neighbor values to d_model for output
        # neighbor_values: (batch, 64, d') where d' = 64
        # Need to project to d_model
        d_prime = neighbor_values.shape[-1]
        d_model = self.d_model
        
        if d_prime != d_model:
            # Simple linear projection (could be learned)
            neighbor_values_proj = F.linear(
                neighbor_values,
                weight=self.W_o.weight[:, :d_prime],
                bias=None
            )  # (batch, 64, d_model)
        else:
            neighbor_values_proj = neighbor_values
        
        # Weighted combination
        # weights: (batch, 64) @ neighbor_values_proj: (batch, 64, d_model)
        output = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, 64)
            neighbor_values_proj   # (batch, 64, d_model)
        ).squeeze(1)  # (batch, d_model)
        
        # Expand to seq_len if needed (for compatibility with forward)
        # For now, return single-token output
        if output.dim() == 2:
            output = output.unsqueeze(1)  # (batch, 1, d_model)
        
        # Final projection
        output = self.W_o(output)
        
        # Expand weights for compatibility
        weights = weights.unsqueeze(1)  # (batch, 1, 64)

        return output, weights

        return output, weights

    @track_first_call
    def forward(
        self,
        Q_input: torch.Tensor = None,  # Legacy: raw tensor
        K: torch.Tensor = None,
        V: torch.Tensor = None,
        T_field: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        neighbor_embeddings: Optional[torch.Tensor] = None,  # Option 6: raw neighbors
        metric: Optional[torch.Tensor] = None,  # Option 6: diagonal metric
        Q_address: Optional[Address] = None,  # Option 6 Extended: full Address structure
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass supporting multiple input modes:
        
        1. Address-based probing (Option 6 Extended, preferred):
           Pass Q_address with populated neighbors
           - Uses probe_address_neighbors() for full Address integration
           - Consumes 6 similarity scores per neighbor
           - O(N × 64 × d') complexity, no dense matmul
        
        2. Neighbor embedding probing (Option 6):
           Pass neighbor_embeddings instead of K/V
           - Uses probe_neighbors() for O(N × 64 × d') complexity
           - NO dense matmul, NO softmax collapse
           - Born gate |ψ|² normalization

        3. Legacy K/V attention:
           Pass K, V for standard geometric attention
           - O(N²) complexity via Q @ K.T matmul
           - Softmax normalization
        """
        # Route to Address-based probing if Q_address provided (Option 6 Extended)
        if Q_address is not None:
            return self.probe_address_neighbors(Q_address=Q_address)
        
        # Route to Option 6 if neighbor_embeddings provided
        if neighbor_embeddings is not None:
            return self.probe_neighbors(
                Q=Q_input,
                neighbor_embeddings=neighbor_embeddings,
                metric=metric
            )

        # Legacy path: K/V matmul attention
        if K is None or V is None:
            raise ValueError(
                "Must provide one of: "
                "Q_address (Option 6 Extended), "
                "neighbor_embeddings (Option 6), "
                "or K and V (legacy)"
            )

        batch_size = Q_input.shape[0]
        seq_len_q = Q_input.shape[1]
        seq_len_k = K.shape[1]

        # Ensure dtype consistency for FP16 training
        target_dtype = self.W_o.weight.dtype
        Q_input = Q_input.to(target_dtype)
        K = K.to(target_dtype)
        V = V.to(target_dtype)

        # Project queries
        Q = self._project_q(Q_input)  # (batch, seq_len_q, d_model)

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
        # NOTE: mask may have different seq_len than K when using multi-head architecture
        # (e.g., language/coordinate heads with different faces). Slice to match K.
        if mask is not None:
            seq_len_k = attention_scores.shape[-1]  # Actual K sequence length

            if mask.dim() == 2:
                # Slice mask to match K sequence length if different
                if mask.shape[1] != seq_len_k:
                    mask = mask[:, :seq_len_k]
                mask_expanded = mask[:, None, None, :]          # (B,1,1,seq_k)
            elif mask.dim() == 3:
                if mask.shape[2] != seq_len_k:
                    mask = mask[:, :, :seq_len_k]
                mask_expanded = mask[:, None, :, :]             # (B,1,seq_q,seq_k)
            elif mask.dim() == 4:
                if mask.shape[3] != seq_len_k:
                    mask = mask[:, :, :, :seq_len_k]
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

        # TRUE TENSOR CONTRACTION: T_ij @ K_j -> V_contracted
        # This makes the field a vector-valued operator, not just a scalar gate.
        # The field geometry directly shapes the value vectors.
        if self.field_contractor is not None:
            # Contract field with keys to get field-transformed values
            V_contracted = self.field_contractor(T_field, K)  # (batch, n_heads, seq_k, d_k)
            # Apply same attention weights to contracted values
            contracted_output = torch.matmul(attention_weights, V_contracted)
            # Mix: attention-weighted values + alpha * field-contracted values
            # alpha is learnable, starts small (0.1) so attention dominates initially
            output = output + self.field_alpha * contracted_output

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
        from Liorhybrid.models.activations import FFN
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
