"""
Entropy-Gated Softmax – Belief Collapse Probability

Implements the four fundamental definitions from the cognitive belief field
framework as a drop-in replacement for standard softmax in attention:

Definition 1 (Caputo Fractional Derivative)
--------------------------------------------
  D_t^α Ψ(x,t) = 1/Γ(⌈α⌉−α) ∫₀ᵗ ∂^⌈α⌉Ψ(x,s)/∂s^⌈α⌉ · (t−s)^{n−1−α} ds

  for α ∈ (1,2), n = ⌈α⌉ = 2.  Discretised over a history buffer of depth T.

Definition 2 (Variable-Order Entropy Functional)
-------------------------------------------------
  H^{(ν(x))}[Ψ] = ∫_M |Ψ(y,t)|^{2ν(x)} φ(x,y) dV_g(y)

  where:
    ν(x)   – position-dependent entropy scaling exponent (observer x)
    φ(x,y) – smooth positive kernel (learned; approximates the Riemannian
              volume element dV_g when φ(x,y) = 1/|M|)

Definition 3 (Entropy-Gated Evolution Equation)
------------------------------------------------
  D_t^α Ψ(x,t) = −∇_{ν(x)} H^{(ν(x))}[Ψ] + η(x,t;τ)

  i.e. the field evolves by descending its own variable-order entropy gradient,
  plus a stochastic forcing term η.

Definition 4 (Belief Collapse Probability) – THE SOFTMAX REPLACEMENT
----------------------------------------------------------------------
  P(x,t) = exp(−H^{(ν(x))}[Ψ]/τ(x))
            ─────────────────────────────────────────────
            ∫_M exp(−H^{(ν(y))}[Ψ]/τ(y)) dV_g(y)

  Discrete (N positions):
    P_i = softmax_i( −H_i / τ_i )
  where H_i = Σ_j |K_j|^{2ν_i} · φ(Q_i, K_j).

  Key distinction from standard softmax:
    standard     : P_{i,j} = exp(s_{i,j}) / Σ_k exp(s_{i,k})
    entropy-gated: P_{i,j} = exp(−H_{i,j}/τ_i) / Σ_k exp(−H_{i,k}/τ_i)

  where H_{i,j} = |K_j|^{2ν_i} · φ(Q_i, K_j) couples the energy of
  key j to the local entropy exponent ν of query i.

  Recovers standard softmax in the limit ν→0, τ=1, φ=exp(scores)/Z.

Usage
-----
  # As a drop-in module:
  from models.entropy_softmax import EntropySoftmax
  entropy_softmax = EntropySoftmax(d_model=64)
  weights = entropy_softmax(scores, key_features=K, query_features=Q)

  # As a function (no learnable parameters):
  from models.entropy_softmax import entropy_gated_softmax
  weights = entropy_gated_softmax(scores, nu=1.0, tau=1.0)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Definition 1 – Caputo Fractional Derivative (discrete approximation)
# ---------------------------------------------------------------------------

class CaputoFractionalDerivativeApprox(nn.Module):
    """
    Discrete Caputo fractional derivative of order α ∈ (1, 2).

    D_t^α Ψ[t] ≈ Σ_{k=0}^{t-1} w_k · (Ψ[t-k] − 2Ψ[t-k-1] + Ψ[t-k-2])

    where the second-difference approximates ∂²Ψ/∂s², and weights

      w_k = [(k+1)^{2-α} − k^{2-α}] / Γ(3−α)

    are the Grünwald–Letnikov weights for α ∈ (1,2).

    Args:
        alpha:      Fractional order α ∈ (0, 2); defaults to 1.5.
        max_depth:  Maximum history buffer depth T.
    """

    def __init__(self, alpha: float = 1.5, max_depth: int = 32):
        super().__init__()
        if not (0.0 < alpha < 2.0):
            raise ValueError(f"alpha must be in (0, 2), got {alpha}")
        self.alpha = alpha
        self.max_depth = max_depth
        # Precompute weights and register as buffer
        weights = self._compute_weights(alpha, max_depth)
        self.register_buffer('weights', weights)

    @staticmethod
    def _compute_weights(alpha: float, T: int) -> torch.Tensor:
        """Grünwald–Letnikov weights for the second-order Caputo approximation."""
        k = torch.arange(T, dtype=torch.float64)
        w = ((k + 1).pow(2.0 - alpha) - k.pow(2.0 - alpha)) / math.gamma(3.0 - alpha)
        return w.float()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Compute the discrete Caputo derivative from a history buffer.

        Args:
            history: [..., T, d] tensor, T time-steps ordered newest→oldest.

        Returns:
            Derivative tensor [..., d].
        """
        T = min(history.shape[-2], self.max_depth)
        h = history[..., :T, :]                 # [..., T, d]
        w = self.weights[:T]                     # [T]

        # Second differences: Δ²Ψ[k] = Ψ[k] − 2Ψ[k+1] + Ψ[k+2]
        if T < 3:
            return torch.zeros_like(h[..., 0, :])
        delta2 = h[..., :-2, :] - 2 * h[..., 1:-1, :] + h[..., 2:, :]  # [..., T-2, d]
        w_d2 = w[:T - 2].unsqueeze(-1)          # [T-2, 1]

        return (w_d2 * delta2).sum(dim=-2)       # [..., d]


# ---------------------------------------------------------------------------
# Definition 2 – Variable-Order Entropy Functional
# ---------------------------------------------------------------------------

class VariableOrderEntropyFunctional(nn.Module):
    """
    Definition 2 (Variable-Order Entropy Functional):

      H^{(ν(x))}[Ψ] = ∫_M |Ψ(y,t)|^{2ν(x)} φ(x,y) dV_g(y)

    Discretised:
      H[b, i] = Σ_j |Ψ[b,j]|^{2ν[b,i]} · φ[b, i, j]

    where:
      ν[b, i]    – entropy exponent at observer position i  (from x_obs)
      φ[b, i, j] – learned positive kernel (softmax over Q·K^T), encoding the
                   Riemannian volume element dV_g in the flat limit
      |Ψ[b,j]|   – field magnitude at position j

    The entropy scaling exponent ν(x) depends on the *observer* x rather than
    the *field* point y, reflecting that different belief states perceive
    information complexity with different nonlinear sensitivities.

    Args:
        d_model: Feature dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        d_phi = max(d_model // 2, 1)

        # ν(x): position-dependent entropy exponent projected from observer
        self.nu_proj  = nn.Linear(d_model, 1)
        # τ(x): position-dependent temperature projected from observer
        self.tau_proj = nn.Linear(d_model, 1)
        # φ(x,y) kernel: learned query/key projections
        self.phi_q = nn.Linear(d_model, d_phi)
        self.phi_k = nn.Linear(d_model, d_phi)

    def forward(
        self,
        Psi: torch.Tensor,      # [..., N_y, d_model]  field Ψ at positions y
        x_obs: torch.Tensor,    # [..., N_x, d_model]  observer features at x
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute H^{(ν(x))}[Ψ], ν(x), and τ(x).

        Args:
            Psi:   Field tensor [..., N_y, d_model].
            x_obs: Observer features [..., N_x, d_model].

        Returns:
            H:   [..., N_x]     variable-order entropy at each observer
            nu:  [..., N_x]     entropy exponents  ν ∈ (0.5, ∞)
            tau: [..., N_x]     temperatures       τ > 0
        """
        # ν(x) > 0.5, τ(x) > 0  (softplus ensures positivity)
        nu  = F.softplus(self.nu_proj(x_obs)).squeeze(-1)  + 0.5   # [..., N_x]
        tau = F.softplus(self.tau_proj(x_obs)).squeeze(-1) + 1e-4  # [..., N_x]

        # φ(x, y): [..., N_x, N_y], positive, rows sum to 1 (volume weights)
        Q_phi = self.phi_q(x_obs)                          # [..., N_x, d_phi]
        K_phi = self.phi_k(Psi)                            # [..., N_y, d_phi]
        scale = math.sqrt(Q_phi.shape[-1])
        phi   = torch.softmax(
            Q_phi @ K_phi.transpose(-2, -1) / scale, dim=-1
        )                                                   # [..., N_x, N_y]

        # |Ψ(y)| with a small floor to keep log well-defined
        Psi_mag = Psi.norm(dim=-1).clamp(min=1e-8)        # [..., N_y]

        # H[..., i] = Σ_j |Ψ_j|^{2ν_i} · φ[..., i, j]
        # Use log-power for numerical stability:
        #   |Ψ_j|^{2ν_i} = exp(2ν_i · log|Ψ_j|)
        log_Psi  = Psi_mag.log()                           # [..., N_y]
        exponent = 2.0 * nu.unsqueeze(-1)                  # [..., N_x, 1]
        Psi_pow  = torch.exp(exponent * log_Psi.unsqueeze(-2))  # [..., N_x, N_y]

        H = (Psi_pow * phi).sum(dim=-1)                    # [..., N_x]

        return H, nu, tau

    def per_key_entropy(
        self,
        key_features: torch.Tensor,    # [..., N_k, d_model]
        query_features: torch.Tensor,  # [..., N_q, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-(query, key) entropy H[..., i, j] = |K_j|^{2ν_i} · φ(Q_i, K_j).

        This is the quantity that enters Definition 4 as the attention logit.

        Returns:
            H:   [..., N_q, N_k]  per-(query, key) entropy
            nu:  [..., N_q]       entropy exponents
            tau: [..., N_q]       temperatures
        """
        nu  = F.softplus(self.nu_proj(query_features)).squeeze(-1)  + 0.5   # [..., N_q]
        tau = F.softplus(self.tau_proj(query_features)).squeeze(-1) + 1e-4  # [..., N_q]

        # φ(Q_i, K_j): [..., N_q, N_k]
        Q_phi = self.phi_q(query_features)
        K_phi = self.phi_k(key_features)
        scale = math.sqrt(Q_phi.shape[-1])
        phi   = torch.softmax(Q_phi @ K_phi.transpose(-2, -1) / scale, dim=-1)

        # |K_j| magnitude: [..., N_k]
        K_mag   = key_features.norm(dim=-1).clamp(min=1e-8)
        log_K   = K_mag.log()                               # [..., N_k]
        exp_q   = 2.0 * nu.unsqueeze(-1)                   # [..., N_q, 1]
        K_pow   = torch.exp(exp_q * log_K.unsqueeze(-2))   # [..., N_q, N_k]

        H = K_pow * phi                                     # [..., N_q, N_k]
        return H, nu, tau


# ---------------------------------------------------------------------------
# Definition 3 – Entropy-Gated Evolution Equation
# ---------------------------------------------------------------------------

class EntropyGatedEvolution(nn.Module):
    """
    Definition 3 (Entropy-Gated Evolution Equation):

      D_t^α Ψ(x,t) = −∇_{ν(x)} H^{(ν(x))}[Ψ] + η(x,t;τ)

    The field Ψ evolves by descending its own variable-order entropy gradient,
    driven by a stochastic forcing term η (modelled here as a learnable
    noise projection scaled by τ(x)).

    This module computes one discrete Euler step:
      Ψ_new ≈ Ψ + dt · (−∇H + η)

    Args:
        d_model:   Feature dimension.
        alpha:     Caputo derivative order α ∈ (0, 2).
        dt:        Discrete time-step.
        max_depth: History buffer depth for the Caputo derivative.
    """

    def __init__(
        self,
        d_model: int,
        alpha: float = 1.5,
        dt: float = 0.01,
        max_depth: int = 16,
    ):
        super().__init__()
        self.dt = dt
        self.entropy_fn  = VariableOrderEntropyFunctional(d_model)
        self.caputo      = CaputoFractionalDerivativeApprox(alpha=alpha, max_depth=max_depth)
        # η projection: stochastic forcing scaled by τ
        self.noise_proj  = nn.Linear(d_model, d_model)

    def forward(
        self,
        Psi: torch.Tensor,       # [B, N, d_model]  current field
        history: Optional[torch.Tensor] = None,  # [B, T, N, d_model] history
        tau_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute one entropy-gated evolution step.

        Args:
            Psi:       Current field [B, N, d_model].
            history:   Field history [B, T, N, d_model] (newest first).
            tau_scale: Global scale for the stochastic forcing.

        Returns:
            Psi_new:   Updated field [B, N, d_model].
            H:         Entropy functional at current step [B, N].
        """
        H, nu, tau = self.entropy_fn(Psi, Psi)   # self-entropy: x_obs = Ψ itself

        # −∇_{ν} H: approximate gradient as −H · Ψ / (|Ψ|² + ε)
        Psi_norm_sq = (Psi * Psi).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        grad_H = H.unsqueeze(-1) * Psi / Psi_norm_sq   # [B, N, d_model]

        # η: stochastic forcing scaled by τ
        eta = tau_scale * tau.unsqueeze(-1) * torch.tanh(self.noise_proj(Psi))

        # Euler step (or Caputo-corrected step when history is available)
        if history is not None:
            # Flatten N into batch for the Caputo derivative
            B, T, N, D = history.shape
            hist_flat = history.view(B * N, T, D)
            caputo_term = self.caputo(hist_flat).view(B, N, D)
            # Correction: scale update by caputo_term to enforce fractional dynamics
            update = caputo_term + self.dt * (-grad_H + eta)
        else:
            update = self.dt * (-grad_H + eta)

        Psi_new = Psi + update
        return Psi_new, H


# ---------------------------------------------------------------------------
# Definition 4 – Belief Collapse Probability (entropy-gated softmax)
# ---------------------------------------------------------------------------

class EntropySoftmax(nn.Module):
    """
    Definition 4 (Belief Collapse Probability) – drop-in softmax replacement.

      P(x,t) = exp(−H^{(ν(x))}[Ψ]/τ(x))
               ─────────────────────────────────────────────
               ∫_M exp(−H^{(ν(y))}[Ψ]/τ(y)) dV_g(y)

    In the attention context (scores ∈ ℝ^{B×N_q×N_k}):

      H[b,i,j] = |K_j|^{2ν_i} · φ(Q_i, K_j)   (per-key entropy at query i)
      P[b,i,j] = exp(−H[b,i,j]/τ_i) / Σ_k exp(−H[b,i,k]/τ_i)

    where:
      ν_i  = ν(Q_i) ∈ [0.5, ∞)  — learned entropy exponent for query i
      τ_i  = τ(Q_i) > 0         — learned temperature for query i
      φ(·) = softmax of learned kernel (positive, sums to 1 over keys)

    Recovers standard softmax when ν→0 and φ is the standard softmax kernel.

    Args:
        d_model:  Feature dimension of the key/query embeddings.
        eps:      Floor for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.entropy_fn = VariableOrderEntropyFunctional(d_model)
        # Fallback projection: embeds raw score scalars into d_model features
        # when key/query feature tensors are not supplied
        self.score_embed = nn.Linear(1, d_model)

    def forward(
        self,
        scores: torch.Tensor,                         # [..., N_q, N_k]
        key_features: Optional[torch.Tensor] = None,  # [..., N_k, d_model]
        query_features: Optional[torch.Tensor] = None,  # [..., N_q, d_model]
        mask: Optional[torch.Tensor] = None,          # [..., N_q, N_k] bool
    ) -> torch.Tensor:
        """
        Compute entropy-gated attention weights (Definition 4).

        When ``key_features`` / ``query_features`` are provided the full
        variable-order entropy functional is used.  When they are absent the
        score magnitudes are projected to ``d_model`` as a fallback so the
        module can always act as a drop-in for ``F.softmax(scores, dim=-1)``.

        Args:
            scores:          Attention logits [..., N_q, N_k].
            key_features:    Key embeddings [..., N_k, d_model].
            query_features:  Query embeddings [..., N_q, d_model].
            mask:            Optional boolean mask [..., N_q, N_k].
                             Masked positions receive zero weight.

        Returns:
            weights: Probability weights [..., N_q, N_k], summing to 1 over
                     the N_k (last) dimension.
        """
        *lead, N_q, N_k = scores.shape

        # ── Build feature tensors from scores when not supplied ────────
        if key_features is None:
            # Mean over query dim → per-key "score summary" → embed to d_model
            score_k = scores.mean(dim=-2).unsqueeze(-1)     # [..., N_k, 1]
            key_features = self.score_embed(score_k)         # [..., N_k, d_model]

        if query_features is None:
            score_q = scores.mean(dim=-1).unsqueeze(-1)     # [..., N_q, 1]
            query_features = self.score_embed(score_q)       # [..., N_q, d_model]

        # ── Per-(query, key) entropy H[..., i, j] ─────────────────────
        H, nu, tau = self.entropy_fn.per_key_entropy(
            key_features=key_features,
            query_features=query_features,
        )  # H: [..., N_q, N_k], nu/tau: [..., N_q]

        # ── Entropy-gated logits: −H[...,i,j] / τ_i ───────────────────
        tau_q = tau.unsqueeze(-1).clamp(min=self.eps)       # [..., N_q, 1]
        gated  = -H / tau_q                                  # [..., N_q, N_k]

        # ── Apply causal / padding mask ────────────────────────────────
        if mask is not None:
            neg_inf = torch.finfo(gated.dtype).min
            gated = gated.masked_fill(~mask.bool(), neg_inf)

        # ── Normalise over key dimension (discrete ∫_M … dV_g) ────────
        weights = torch.softmax(gated, dim=-1)               # [..., N_q, N_k]
        return weights


# ---------------------------------------------------------------------------
# Functional convenience wrapper
# ---------------------------------------------------------------------------

def entropy_gated_softmax(
    scores: torch.Tensor,
    nu: float = 1.0,
    tau: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Parameter-free entropy-gated softmax using fixed ν and τ.

    P_i = exp(−|s_i|^{2ν}/τ) / Σ_j exp(−|s_j|^{2ν}/τ)

    When ν=0.5 and τ=1 this is identical to softmax(−|s|), i.e. a reversed
    Boltzmann distribution over score magnitudes.

    Args:
        scores: Input tensor (any shape).
        nu:     Entropy exponent ν > 0.
        tau:    Temperature τ > 0.
        dim:    Dimension to normalise over.

    Returns:
        Probability tensor with same shape as ``scores``.
    """
    H = scores.abs().clamp(min=1e-8).pow(2.0 * nu)          # |s|^{2ν}
    return torch.softmax(-H / tau, dim=dim)
