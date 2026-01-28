"""
Geometric Product Operations

Implements wedge, tensor, and spinor products for geometric attention.

These replace the standard cosine similarity (dot product) in transformers
with operations that capture richer geometric structure.

References:
- Wedge product: Antisymmetric exterior product (Grassmann algebra)
- Tensor product: Full correlation structure (no information loss)
- Spinor product: Rotational features via Clifford algebra
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import math
import torch
import torch.nn.functional as F


def wedge_product(Q: torch.Tensor, K: torch.Tensor, T_field: torch.Tensor) -> torch.Tensor:
    """
    Compute field-contracted wedge product for attention scores.

    MEMORY FIX: Instead of computing full outer products (causing OOM),
    contract through the cognitive field T_ij to get scalar scores directly.

    Mathematical approach:
        score(i,j) = Σ_μν T_μν (Q_i^μ K_j^ν - K_j^μ Q_i^ν)

    This is the antisymmetric bilinear form contracted with the field tensor.
    Memory: O(batch × seq²) instead of O(batch × seq² × d²)

    Args:
        Q: Query tensor (..., seq_len_q, d_model)
        K: Key tensor (..., seq_len_k, d_model)
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex

    Returns:
        Wedge product scores (..., seq_len_q, seq_len_k)

    Interpretation:
        High score = Q and K span new space (orthogonal concepts)
        Low score = Q and K are parallel (redundant information)
        Field T_ij guides which directions matter for orthogonality
    """
    # Average field over spatial dimensions to get metric
    T_avg = T_field.mean(dim=(0, 1))  # (D, D)

    # Convert complex field to real (use magnitude)
    if T_avg.is_complex():
        T_avg = torch.abs(T_avg)

    # Ensure T_avg matches d_model dimension
    d_model = Q.shape[-1]
    D = T_avg.shape[0]

    if d_model != D:
        if d_model < D:
            # Truncate field to match d_model
            T_avg = T_avg[:d_model, :d_model]
        else:
            # Pad field to match d_model
            T_avg = F.pad(T_avg, (0, d_model - D, 0, d_model - D))

    # Cast T_avg to match Q dtype for FP16 training
    T_avg = T_avg.to(Q.dtype)

    # Compute bilinear forms through field
    # Q @ T @ K^T: shape (..., seq_q, seq_k)
    Q_T = torch.matmul(Q, T_avg)  # (..., seq_q, d_model)
    scores_QK = torch.matmul(Q_T, K.transpose(-2, -1))  # (..., seq_q, seq_k)

    # K @ T @ Q^T: shape (..., seq_k, seq_q)
    K_T = torch.matmul(K, T_avg)  # (..., seq_k, d_model)
    scores_KQ = torch.matmul(K_T, Q.transpose(-2, -1))  # (..., seq_k, seq_q)

    # Antisymmetrize: Q∧K = Q⊗K - K⊗Q (contracted form)
    wedge_scores = scores_QK - scores_KQ.transpose(-2, -1)

    return wedge_scores


def tensor_product(Q: torch.Tensor, K: torch.Tensor, T_field: torch.Tensor) -> torch.Tensor:
    """
    Compute field-contracted tensor product for attention scores.

    MEMORY FIX: Instead of computing full outer products,
    contract through field to get scalar scores directly.

    Mathematical approach:
        score(i,j) = ||Q_i|| × ||K_j|| × Tr(T)

    This captures total signal strength modulated by field magnitude.
    Memory: O(batch × seq²) instead of O(batch × seq² × d²)

    Args:
        Q: Query tensor (..., seq_len_q, d_model)
        K: Key tensor (..., seq_len_k, d_model)
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex

    Returns:
        Tensor product scores (..., seq_len_q, seq_len_k)

    Interpretation:
        Captures signal magnitude interaction modulated by field
        High score = Both Q and K have high magnitude + field is strong
        Acts as gating mechanism for salient memories
    """
    # Average field over spatial dimensions
    T_avg = T_field.mean(dim=(0, 1))  # (D, D)

    # Convert complex field to real (use magnitude)
    if T_avg.is_complex():
        T_avg = torch.abs(T_avg)

    # Compute field strength as trace
    field_strength = torch.trace(T_avg)

    # Compute norms of Q and K
    Q_norm = torch.norm(Q, dim=-1, keepdim=True)  # (..., seq_q, 1)
    K_norm = torch.norm(K, dim=-1, keepdim=True)  # (..., seq_k, 1)

    # Tensor product score: ||Q|| × ||K|| × field_strength
    # Broadcasting: (..., seq_q, 1) × (..., 1, seq_k) → (..., seq_q, seq_k)
    tensor_scores = Q_norm * K_norm.transpose(-2, -1) * field_strength

    return tensor_scores


def spinor_product(
    Q: torch.Tensor,
    K: torch.Tensor,
    T_field: torch.Tensor,
    spatial_pos: tuple = None
) -> torch.Tensor:
    """
    Compute field-contracted spinor product for rotational attention.

    MEMORY FIX: Contract spinor computation through field to avoid
    creating large bivector tensors.

    Spinors capture rotational features - how the field's current state
    "rotates" the Q-K interaction. This is sensitive to field topology
    and captures order/sequence information crucial for causality.

    Mathematical approach:
        score(i,j) = Q_i^† T_field T_field^† K_j

    This is the rotational alignment score modulated by field curvature.
    Memory: O(batch × seq²) instead of O(batch × seq² × d²)

    Args:
        Q: Query tensor (..., seq_len_q, d_model)
        K: Key tensor (..., seq_len_k, d_model)
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex
        spatial_pos: Optional (x, y) position (currently unused)

    Returns:
        Spinor product scores (..., seq_len_q, seq_len_k)

    Interpretation:
        Captures rotational alignment between query and memory
        High score = Query aligns with field's rotational flow
        Sensitive to phase and sequence order
    """
    # Average field over spatial dimensions
    T_local = T_field.mean(dim=(0, 1))  # (D, D)

    # Extract real and imaginary parts
    T_real = T_local.real
    T_imag = T_local.imag

    # Match dimensions
    d_model = Q.shape[-1]
    D = T_real.shape[0]

    if d_model != D:
        if d_model < D:
            T_real = T_real[:d_model, :d_model]
            T_imag = T_imag[:d_model, :d_model]
        else:
            T_real = F.pad(T_real, (0, d_model - D, 0, d_model - D))
            T_imag = F.pad(T_imag, (0, d_model - D, 0, d_model - D))

    # Cast T matrices to match Q dtype for FP16 training
    T_real = T_real.to(Q.dtype)
    T_imag = T_imag.to(Q.dtype)

    # Compute rotational coupling: Q^† T T^† K
    # Real component
    T_sq_real = torch.matmul(T_real, T_real.T) + torch.matmul(T_imag, T_imag.T)

    # Apply to Q and K
    Q_rot = torch.matmul(Q, T_sq_real)  # (..., seq_q, d_model)
    spinor_scores = torch.matmul(Q_rot, K.transpose(-2, -1))  # (..., seq_q, seq_k)

    # Add imaginary coupling for phase sensitivity
    T_sq_imag = torch.matmul(T_real, T_imag.T) - torch.matmul(T_imag, T_real.T)
    Q_rot_imag = torch.matmul(Q, T_sq_imag)
    spinor_scores_imag = torch.matmul(Q_rot_imag, K.transpose(-2, -1))

    # Combine real and imaginary contributions
    spinor_scores_total = spinor_scores + torch.abs(spinor_scores_imag)

    return spinor_scores_total


def geometric_score(
    Q: torch.Tensor,
    K: torch.Tensor,
    T_field: torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0),
    spatial_pos: tuple = None
) -> torch.Tensor:
    """
    Compute combined geometric attention score using field contractions.

    Combines wedge, tensor, and spinor products with learnable weights:
        score = w_wedge * wedge(Q,K,T) + w_tensor * tensor(Q,K,T) + w_spinor * spinor(Q,K,T)

    All products are contracted through the cognitive field T_ij, ensuring
    memory efficiency: O(batch × seq²) total.

    Args:
        Q: Query tensor (..., seq_len_q, d_model)
        K: Key tensor (..., seq_len_k, d_model)
        T_field: Cognitive tensor field (N_x, N_y, D, D)
        weights: (w_wedge, w_tensor, w_spinor) combination weights
        spatial_pos: Optional spatial position (currently unused)

    Returns:
        Combined attention scores (..., seq_len_q, seq_len_k)

    Interpretation:
        - Wedge: Causal divergence / orthogonality
        - Tensor: Signal strength / gating
        - Spinor: Rotational alignment / sequence order
    """
    w_wedge, w_tensor, w_spinor = weights

    # Compute field-contracted products (all return scalar scores)
    score_wedge = wedge_product(Q, K, T_field)
    score_tensor = tensor_product(Q, K, T_field)
    score_spinor = spinor_product(Q, K, T_field, spatial_pos)

    # Normalize individual scores to prevent dominance
    # Use unbiased=False to handle small batches (batch size 1)
    # FP16-safe epsilon values
    score_wedge = score_wedge / (score_wedge.std(unbiased=False) + 1e-4)
    score_tensor = score_tensor / (score_tensor.std(unbiased=False) + 1e-4)
    score_spinor = score_spinor / (score_spinor.std(unbiased=False) + 1e-4)

    # Combine with weights
    score = w_wedge * score_wedge + w_tensor * score_tensor + w_spinor * score_spinor

    return score


def geometric_score_from_phase(
    A_Q: torch.Tensor,
    theta_Q: torch.Tensor,
    A_K: torch.Tensor,
    theta_K: torch.Tensor,
    T_field: torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    Compute geometric attention score from phase-compressed representations.

    This function enables PARALLEL multi-head computation by working with
    scalar phase angles rather than high-dimensional vectors.

    Theory: Psi = A * exp(i*theta)
    The 4 operators form the complete information generation cycle:
    Rank-1 (vectors) → Rank-2 (causal interactions) → Rank-1 (new information)

    - Wedge (∧): Creates antisymmetric rank-2 structure from rank-1
    - Tensor (⊗): Creates full correlation rank-2 structure from rank-1
    - Spinor: Rotational operations within rank-2 causal space
    - Hodge (⋆): Conservation factor - metric contraction rank-2 → rank-1

    Args:
        A_Q: Query amplitudes (batch, n_heads, seq_len_q)
        theta_Q: Query phases (batch, n_heads, seq_len_q)
        A_K: Key amplitudes (batch, n_heads, seq_len_k)
        theta_K: Key phases (batch, n_heads, seq_len_k)
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex
        weights: (w_wedge, w_tensor, w_spinor, w_hodge) combination weights

    Returns:
        Attention scores (batch, n_heads, seq_len_q, seq_len_k)

    Performance:
        Replaces sequential head loop with single parallel operation.
        Expected speedup: 8-10x for 8-head attention.
    """
    w_wedge, w_tensor, w_spinor, w_hodge = weights

    # Extract field properties to modulate attention
    T_avg = T_field.mean(dim=(0, 1))  # (D, D) - spatial average

    # Field strength (trace) - scalar modulation of tensor product
    # FP16-SAFE: Use larger epsilon values (fp16 min ~6e-5, precision ~0.1%)
    # Small eps like 1e-3/1e-4 are at the noise floor and cause gradient NaN
    if T_avg.is_complex():
        field_strength = torch.abs(torch.trace(T_avg)) + 0.01  # eps for zero trace
        field_coupling = torch.abs(T_avg).mean() + 0.01
        field_volume = torch.norm(T_avg) + 0.01  # eps for zero norm
    else:
        field_strength = torch.trace(T_avg).abs() + 0.01
        field_coupling = T_avg.abs().mean() + 0.01
        field_volume = torch.norm(T_avg) + 0.01

    # Compute phase difference matrix: (batch, n_heads, seq_q, seq_k)
    # Broadcasting: (batch, n_heads, seq_q, 1) - (batch, n_heads, 1, seq_k)
    delta_theta = theta_Q.unsqueeze(-1) - theta_K.unsqueeze(-2)

    # SEQUENTIAL PIPELINE: rank-1 → rank-2 → rank-2 → rank-2 → rank-1
    # Each step feeds into the next (physical information generation process)
    # All interactions computed in parallel (vectorized)

    # Broadcasting setup for amplitude operations
    A_Q_expanded = A_Q.unsqueeze(-1)      # (batch, n_heads, seq_q, 1)
    A_K_expanded = A_K.unsqueeze(-2)      # (batch, n_heads, 1, seq_k)

    # Step 1: Wedge (∧) - rank-1 → rank-2 antisymmetric
    # Creates the initial antisymmetric structure from input vectors
    rank2 = torch.sin(delta_theta) * w_wedge
    rank2 = torch.tanh(rank2 / 10.0)  # Stabilize

    # Step 2: Tensor (⊗) - rank-2 → rank-2 with correlation
    # Adds magnitude correlation structure to the antisymmetric base
    rank2 = rank2 * (A_Q_expanded * A_K_expanded * field_strength * w_tensor)
    rank2 = torch.tanh(rank2 / 10.0)  # Stabilize

    # Step 3: Spinor - rank-2 → rank-2 with rotation
    # Adds rotational structure within the rank-2 causal space
    rotational_component = torch.cos(delta_theta) * field_coupling * w_spinor
    rank2 = rank2 + torch.tanh(rotational_component / 10.0)

    # Step 4: Hodge (*) - rank-2 -> rank-1 metric contraction
    # Conservation factor: contracts rank-2 back to rank-1 using field metric
    A_avg = (A_Q_expanded + A_K_expanded) / 2
    # FP16-SAFE: Use 0.1 eps (not 1e-3) to avoid gradient instability
    contraction_factor = A_avg * field_strength / (field_volume + 0.1) * w_hodge
    score = rank2 * torch.tanh(contraction_factor / 10.0)

    # Final stabilization
    score = torch.clamp(score, min=-1e4, max=1e4)

    return score


# =============================================================================
# OPTION 3: EXPONENTIAL FORM GEOMETRIC SCORING
# =============================================================================


def geometric_score_from_exponential(
    psi_Q: torch.Tensor,
    psi_K: torch.Tensor,
    T_field: torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    Compute geometric attention score from 16D biquaternion representations.

    This is the Option 3 scoring function that works with ExponentialPhaseExtractor.
    The 16D psi vectors encode the full triality structure:
    - First 8D: Real part (A·exp(Θ) + B·cos(Θ))
    - Second 8D: Imaginary part (B·sin(Θ))

    The geometric score combines:
    - Wedge: Antisymmetric interaction (real × imag cross-terms)
    - Tensor: Full correlation (dot product modulated by field)
    - Spinor: Rotational coupling (via imaginary parts)
    - Hodge: Conservation/contraction factor

    Args:
        psi_Q: Query biquaternion (batch, n_heads, seq_len_q, 16)
        psi_K: Key biquaternion (batch, n_heads, seq_len_k, 16)
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex
        weights: (w_wedge, w_tensor, w_spinor, w_hodge) combination weights

    Returns:
        Attention scores (batch, n_heads, seq_len_q, seq_len_k)
    """
    w_wedge, w_tensor, w_spinor, w_hodge = weights

    # Split into real and imaginary parts
    psi_Q_real = psi_Q[..., :8]   # (batch, n_heads, seq_q, 8)
    psi_Q_imag = psi_Q[..., 8:]   # (batch, n_heads, seq_q, 8)
    psi_K_real = psi_K[..., :8]   # (batch, n_heads, seq_k, 8)
    psi_K_imag = psi_K[..., 8:]   # (batch, n_heads, seq_k, 8)

    # Extract field properties for modulation
    T_avg = T_field.mean(dim=(0, 1))  # (D, D)
    if T_avg.is_complex():
        field_strength = torch.abs(torch.trace(T_avg)) + 0.01
        field_coupling = torch.abs(T_avg).mean() + 0.01
    else:
        field_strength = torch.trace(T_avg).abs() + 0.01
        field_coupling = T_avg.abs().mean() + 0.01

    # =================================================================
    # WEDGE PRODUCT: Antisymmetric (real × imag cross-terms)
    # =================================================================
    # Q_real · K_imag - Q_imag · K_real (antisymmetric bilinear form)
    # Shape: (batch, heads, seq_q, 8) @ (batch, heads, 8, seq_k)
    wedge_QK = torch.matmul(psi_Q_real, psi_K_imag.transpose(-2, -1))
    wedge_KQ = torch.matmul(psi_Q_imag, psi_K_real.transpose(-2, -1))
    wedge_score = (wedge_QK - wedge_KQ) * w_wedge
    # Already bounded since inputs are from exp/sin/cos

    # =================================================================
    # TENSOR PRODUCT: Full correlation (dot product)
    # =================================================================
    # Q · K for both real and imaginary parts, modulated by field
    tensor_real = torch.matmul(psi_Q_real, psi_K_real.transpose(-2, -1))
    tensor_imag = torch.matmul(psi_Q_imag, psi_K_imag.transpose(-2, -1))
    tensor_score = (tensor_real + tensor_imag) * field_strength * w_tensor

    # =================================================================
    # SPINOR PRODUCT: Rotational coupling (imaginary coherence)
    # =================================================================
    # Measures phase alignment via imaginary part correlation
    spinor_score = torch.matmul(psi_Q_imag, psi_K_imag.transpose(-2, -1))
    spinor_score = spinor_score * field_coupling * w_spinor

    # =================================================================
    # HODGE STAR: Local conservation / measure factor in [0, 1]
    # =================================================================
    # PERF FIX (Dec 2025): Removed norm_product.mean() which was O(N²) reduction.
    # Now uses local sigmoid squashing - no batch-wide stats needed.
    # Like a depth finder: read local geometry, don't survey the whole ocean.
    Q_norm = psi_Q.norm(dim=-1, keepdim=True)           # (B, H, L_q, 1)
    K_norm = psi_K.norm(dim=-1, keepdim=True)           # (B, H, L_k, 1)
    norm_product = Q_norm * K_norm.transpose(-2, -1)    # (B, H, L_q, L_k)

    # Map norm_product → [0,1] locally via monotone squashing.
    # beta controls sharpness of Hodge response (tunable hyperparameter)
    hodge_beta = 0.5
    hodge_raw = torch.sigmoid(hodge_beta * norm_product)  # in (0,1)
    hodge_factor = hodge_raw * w_hodge                    # in (0, w_hodge)

    # =================================================================
    # COMBINE: Weighted sum of geometric products with Hodge gating
    # =================================================================
    score = wedge_score + tensor_score + spinor_score

    # Hodge as conservation gate: 0→kill, 1→preserve
    # Affine shift prevents zeroing everything: gate in [0.5, 0.5+0.5*w_hodge]
    score = score * (0.5 + 0.5 * hodge_factor)

    # PERF FIX (Dec 2025): Removed score.std() which was O(N²) reduction.
    # Use transformer-standard 1/√d scaling instead - O(1) and principled.
    d_head = psi_Q_real.size(-1)  # 8
    score = score * (1.0 / math.sqrt(d_head))

    return score
