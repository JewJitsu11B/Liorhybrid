"""
Config Cost Estimator

One-shot parameter / memory / compute estimator for quickly exploring configs
without running training.

COMMENT KEY (variables that affect cost + big-O)

Symbols:
- B: batch_size
- T: max_seq_len
- V: vocab_size
- d: d_model
- L: n_layers (+ n_attention_layers where applicable)
- H: n_heads
- (N_x, N_y): spatial_size, N = N_x * N_y
- D: field_dim
- dtype bytes: bytes_per_param(dtype), field bytes: bytes_per_complex(field_dtype)

Parameter drivers (dominant terms):
- Token embedding:        V·d
- Positional embedding:   T·d
- Modality embedding:     3·d
- LM head:                d·V + V
- Core stack (proxy):     O(L·d²)
- Trainer2 geometry:      O(n² + k·n + r·n) where n=coord_dim_n, k=rotor_k, r=lowrank_r

Memory drivers (rough):
- Params:                 total_params · bytes_per_param
- Grads:                  trainable_params · bytes_per_param
- Adam states (if used):  ~2 · trainable_params · bytes_per_param
- Field state:            N · D² · bytes_per_complex
- Activations (proxy):    O(B·T·d·L)

Compute drivers (rough):
- Dense attention:        O(B·T²·d)  (when using quadratic attention)
- Mamba/structured ops:   treated here as O(B·T·d²·L) proxy for sizing

FUTURE: Multi-domain coarse/fine embedding system
- Each domain (coding, math, legal, ethics) has coarse + fine granularity slots
- K = [domain_1_coarse | domain_1_fine | domain_2_coarse | ...]
- Parameters: n_domains × 2 × dim_per_slot (e.g., 4 domains × 2 × 64 = 512)
- See training/embeddings.py for implementation scaffold
"""
from __future__ import annotations
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math


@dataclass(frozen=True)
class CostEstimate:
    total_params: int
    trainable_params: int
    param_bytes: int
    grad_bytes: int
    optimizer_state_bytes: int
    field_bytes: int
    activation_bytes_est: int
    total_bytes_est: int
    flops_fwd_est: Optional[float] = None
    flops_step_est: Optional[float] = None


def _bytes_per_param(dtype: str) -> int:
    dtype = (dtype or "fp32").lower()
    if dtype in ("fp16", "float16", "half"):
        return 2
    if dtype in ("bf16", "bfloat16"):
        return 2
    if dtype in ("fp32", "float32", "float"):
        return 4
    if dtype in ("fp64", "float64", "double"):
        return 8
    return 4


def _bytes_per_complex(dtype: str) -> int:
    # Field default in this repo is complex64.
    dtype = (dtype or "complex64").lower()
    if dtype in ("complex64",):
        return 8
    if dtype in ("complex128",):
        return 16
    return 8


def _field_bytes(spatial_size: Tuple[int, int], field_dim: int, field_dtype: str = "complex64") -> int:
    n_x, n_y = spatial_size
    elems = int(n_x) * int(n_y) * int(field_dim) * int(field_dim)
    return elems * _bytes_per_complex(field_dtype)


def estimate_cost(config: Dict) -> CostEstimate:
    """
    Roughly estimate params, memory, and FLOPs for a given config.

    Notes:
    - Parameter counts are exact for modules we can infer analytically here,
      and approximate where the model has complex internal operators.
    - FLOPs and activation memory are coarse estimates meant for comparing configs.
    """
    d_model = int(config.get("d_model", 256))
    n_layers = int(config.get("n_layers", 4))
    n_attention_layers = int(config.get("n_attention_layers", 2))
    n_heads = int(config.get("n_heads", 4))
    vocab_size = int(config.get("vocab_size", 32000))
    seq_len = int(config.get("max_seq_len", 512))
    batch_size = int(config.get("batch_size", 16))
    field_dim = int(config.get("field_dim", 16))
    spatial_size = tuple(config.get("spatial_size", (8, 8)))
    dtype = str(config.get("dtype", "fp32"))
    bytes_per = _bytes_per_param(dtype)

    use_causal_field = bool(config.get("use_causal_field", False))

    # --- Embedding + LM head (exact) ---
    # MultimodalEmbedding (text part dominates for LM): token_emb + pos_emb + modality_emb
    token_emb = vocab_size * d_model
    pos_emb = seq_len * d_model
    modality_emb = 3 * d_model
    # LM head: d_model -> vocab
    lm_head = d_model * vocab_size + vocab_size

    # --- Core model (approx) ---
    # Standard transformer: roughly 4*d_model^2 per layer for projections + FFN ~ O(d_model^2)
    # CausalField stack is more complex; we approximate it at the same order for sizing.
    # This is intentionally conservative for memory planning.
    core_per_layer = 12 * d_model * d_model  # coarse proxy
    core_params = core_per_layer * (n_layers + n_attention_layers)

    # --- Trainer2 geometry params ---
    coord_dim_n = int(config.get("coord_dim_n", 8))
    rotor_k = int(config.get("rotor_k", 6))
    lowrank_r = int(config.get("lowrank_r", 4))
    # Geometry adds: metric tensor O(n²), rotor state O(k), lowrank O(r*n)
    geometry_params = coord_dim_n * coord_dim_n + rotor_k * 2 + lowrank_r * coord_dim_n

    total_params = token_emb + pos_emb + modality_emb + lm_head + core_params + geometry_params

    # TODO: Multi-domain coarse/fine embedding params
    # When implemented, add: n_domains × 2 × dim_per_slot for embedding tables

    # Trainable params estimate
    trainable_params = total_params

    # --- Memory ---
    param_bytes = total_params * bytes_per
    grad_bytes = trainable_params * bytes_per

    optimizer = str(config.get("optimizer", "biquat")).lower()
    if optimizer in ("adam", "adamw"):
        optimizer_state_bytes = 2 * trainable_params * bytes_per
    else:
        # biquat / lior store no per-param state in this repo
        optimizer_state_bytes = 0

    field_bytes = _field_bytes(spatial_size, field_dim, field_dtype=str(config.get("field_dtype", "complex64")))

    # Activation memory (very rough): keep a few activations per layer.
    act_per_layer = batch_size * seq_len * d_model * bytes_per
    activation_bytes_est = int(act_per_layer * max(1, (n_layers + n_attention_layers)) * 3)

    total_bytes_est = param_bytes + grad_bytes + optimizer_state_bytes + field_bytes + activation_bytes_est

    # --- FLOPs (very rough) ---
    flops_fwd = None
    flops_step = None
    if not use_causal_field:
        # transformer-ish forward FLOPs ~ O(B*T*d_model^2*layers) + attention O(B*H*T^2*d_k)
        flops_proj = 4.0 * batch_size * seq_len * (d_model ** 2) * n_layers
        flops_attn = float(batch_size * n_heads * (seq_len ** 2) * (d_model / max(1, n_heads)) * n_layers)
        flops_fwd = flops_proj + flops_attn
        flops_step = flops_fwd * 3.0  # forward + backward + optimizer
    else:
        # CausalField stack is dominated by structured ops; treat as O(B*T*d_model^2*layers) proxy.
        flops_fwd = float(batch_size * seq_len * (d_model ** 2) * (n_layers + n_attention_layers) * 2.0)
        flops_step = flops_fwd * 3.0

    return CostEstimate(
        total_params=total_params,
        trainable_params=trainable_params,
        param_bytes=param_bytes,
        grad_bytes=grad_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        field_bytes=field_bytes,
        activation_bytes_est=activation_bytes_est,
        total_bytes_est=total_bytes_est,
        flops_fwd_est=flops_fwd,
        flops_step_est=flops_step,
    )


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"


def print_estimate(config: Dict) -> None:
    est = estimate_cost(config)
    print("\n" + "=" * 70)
    print("CONFIG COST ESTIMATE (approx)")
    print("=" * 70)
    print(f"Total params:      {est.total_params:,}")
    print(f"Trainable params:  {est.trainable_params:,}")
    print(f"Param memory:      {format_bytes(est.param_bytes)}")
    print(f"Grad memory:       {format_bytes(est.grad_bytes)}")
    print(f"Opt state memory:  {format_bytes(est.optimizer_state_bytes)}")
    print(f"Field memory:      {format_bytes(est.field_bytes)}")
    print(f"Activation est:    {format_bytes(est.activation_bytes_est)}")
    print(f"Total est:         {format_bytes(est.total_bytes_est)}")
    if est.flops_fwd_est is not None:
        print(f"FLOPs fwd est:     {est.flops_fwd_est/1e12:.3f} TFLOPs")
    if est.flops_step_est is not None:
        print(f"FLOPs/step est:    {est.flops_step_est/1e12:.3f} TFLOPs")
    print("=" * 70)
