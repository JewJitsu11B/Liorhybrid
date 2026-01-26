"""Trainer2: single-file manual trainer for the Bayesian Cognitive Field stack.

CUDA-only and no-autograd are enforced at import time. Training is two-phase
(free + nudged) with manual updates; contrastive stats are primary and SPSA is
a fallback only.

Geometry is configurable via a mode menu:
- frame: derived | learned_lowrank | rotor
- metric: diag_rot | block_rot
- R source: constitutive | curvature

Retrieval cost uses:
- cost = R_sc * sqrt(g(v,v) + eps)  (SPD geometry)
- weights = softmax(-beta * cost)

R_sc is a scalar curvature/resilience collapse used to weight costs even when
richer structures exist.

Constraints: no CPU fallback, no optimizer, no grads, no extra files.

Table of sections:
- Section 1: runtime constraints and guards
- Section 2: config + mode menu + validation
"""

# ==============================================================================
# SECTION 1: HARD runtime constraints + CUDA-only enforcement + no-autograd enforcement
# ==============================================================================
import os
import time
import math
import json
import hashlib
import dataclasses
import inspect
import datetime
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .geometric_diagnostics import PathBuffer, format_diagnostics


def _ts() -> str:
    """Timestamp helper for log messages."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def _ts_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

# NOTE: No einsum in Sections 1-2. Hot-path kernels will use precomputed
# contractions and matmul later.

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this trainer. No CPU fallback is allowed. "
        "Install a CUDA-enabled PyTorch build, verify your GPU driver, and set "
        "CUDA_VISIBLE_DEVICES if needed."
    )

# Debug tracing for first-run of each step
_TRACE_FIRST = {}
_TRACE_TIMES = {}

def _trace(tag: str, force: bool = False) -> None:
    """Print trace message with timestamp on first call for each tag."""
    if tag not in _TRACE_FIRST or force:
        _TRACE_FIRST[tag] = True
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Calculate elapsed time from last trace
        elapsed = ""
        if _TRACE_TIMES:
            last_time = max(_TRACE_TIMES.values())
            delta_ms = (time.time() - last_time) * 1000
            elapsed = f" (+{delta_ms:.0f}ms)"
        _TRACE_TIMES[tag] = time.time()
        print(f"[{now}] TRACE: {tag}{elapsed}", flush=True)
torch.set_grad_enabled(False)
torch.autograd.set_detect_anomaly(False)

# Module-level device (cached). No CPU fallback allowed.
DEVICE = torch.device("cuda")

def apply_backend_flags(cfg: "TrainConfig") -> None:
    # cuDNN knobs
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)

    # TF32 knobs (Ampere+)
    tf32 = bool(cfg.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.matmul_allow_tf32)
    torch.backends.cudnn.allow_tf32 = tf32
    # NOTE: Do not print every step; print once at startup if desired.


def inference_context():
    # Stronger than no_grad: prevents autograd graph creation and reduces overhead.
    return torch.inference_mode()


def assert_no_autograd(params: Optional[Iterable[nn.Parameter]] = None) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError("Autograd must remain disabled for trainer2.")
    if params is not None:
        for param in params:
            if getattr(param, "grad", None) is not None:
                raise RuntimeError("Gradient buffer detected; trainer2 forbids autograd.")

def disable_gradients(*objs: Any) -> None:
    seen: set[int] = set()

    def _disable(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, nn.Module):
            for param in obj.parameters():
                if param.requires_grad:
                    param.requires_grad_(False)
        if isinstance(obj, torch.Tensor):
            if obj.requires_grad:
                obj.requires_grad_(False)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                _disable(value)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                _disable(value)
            return
        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                _disable(value)

    for obj in objs:
        _disable(obj)


def get_device() -> torch.device:
    return DEVICE


def to_device(batch: Any, device: torch.device = DEVICE) -> Any:
    if isinstance(batch, torch.Tensor):
        if batch.is_cuda:
            return batch
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device=device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device=device) for v in batch)
    return batch


def gpu_sync_for_timing() -> None:
    torch.cuda.synchronize()


def assert_cuda_tensor(x: Any, name: str) -> None:
    # Only call in debug or low-frequency checks, not per-step.
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if not x.is_cuda:
        raise RuntimeError(f"{name} must be on CUDA; CPU tensors are not allowed in the hot path.")


assert_no_autograd()

# ==============================================================================
# SECTION 2: CONFIG + MODE MENU + VALIDATION
# ==============================================================================
# coord_dim_n is the manifold dimension used for metric contraction and retrieval geometry.
# CxH ⊕ CxH being 16 real dimensions does not force n=4; n is chosen by how you represent coordinates.
# lowrank_r is the number of nonlocal directions in learned low-rank mode.
# rotor_k is the number of plane rotations (Givens-like) composing the interface frame.


@dataclass
class TrainConfig:
    # 2.1 Geometry Config
    # coord_dim_n is the chart dimension used for g(v,v) and retrieval geometry.
    # It is NOT forced by biquat/CxH dimensionality; it is chosen by your coordinate representation.
    coord_dim_n: int = 8          # Manifold dimension (n)
    eps: float = 1e-8             # Sqrt/Inversion stability
    retrieval_beta: float = 5.0   # Softmin sharpness

    # Mode Menu
    frame_mode: str = "rotor"        # derived | learned_lowrank | rotor
    metric_mode: str = "diag_rot"    # diag_rot | block_rot
    R_source: str = "constitutive"   # constitutive | curvature
    rotor_mode: str = "stateful"     # off | derived | stateful

    lowrank_r: int = 4            # Rank for learned nonlocal low-rank
    rotor_k: int = 6              # Number of plane rotations
    theta_wrap: float = 2 * math.pi

    # 2.2 Training Loop (Manual Updates)
    tbptt_window_steps: int = 64
    beta_nudge: float = 1e-3      # Division by this occurs in updates
    eta_update: float = 1e-2      # Learning rate for manual updates (increased from 1e-4 for visible learning)
    nudge_every_windows: int = 1
    max_epochs: int = 1
    max_windows: int = 0

    # SPSA Fallback Toggles
    enable_spsa_fallback: bool = False
    spsa_sigma: float = 0.01
    spsa_lr: float = 0.01
    spsa_directions: int = 4

    # 2.2b Symplectic Dynamics (Phase-Space Preserving Integration)
    # When dynamics_mode="symplectic", field.T evolves via Stormer-Verlet integration
    # preserving phase space volume (Liouville's theorem) to prevent memory rot.
    dynamics_mode: str = "dissipative"      # "dissipative" | "symplectic"
    symplectic_dt: float = 0.005            # Integration timestep
    symplectic_m_cog: float = 1.0           # Effective mass (cognitive inertia)
    symplectic_hbar_cog: float = 0.1        # Cognitive Planck constant (for Laplacian term)
    symplectic_potential: str = "harmonic"  # "zero" | "harmonic" | "gaussian_well"
    symplectic_stiffness: float = 0.01      # Spring constant k for restoring force -k(T - T_eq)

    # 2.2c Nudge Signal Settings
    # Controls how the "force of truth" is computed for contrastive learning.
    # Nudge = k * (target_coord - current_coord) pulls system toward ground truth.
    nudge_mode: str = "target_embedding"    # "off" | "target_embedding" | "template"
    nudge_scale: float = 0.1                # k in F = k * (target - current)
    nudge_use_shifted_target: bool = True   # Use shifted (next token) as target

    # 2.3 Performance Toggles
    # Backend / numerics toggles (applied once, not at import time)
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    allow_tf32: bool = False
    matmul_allow_tf32: bool = False
    use_torch_compile: bool = False
    use_cudagraphs: bool = False
    warmup_steps: int = 10
    capture_batch_size: int = 0
    static_shapes: bool = True
    run_smoke_tests: bool = False

    # I/O
    log_every_windows: int = 10
    save_every_windows: int = 100
    run_dir: str = "./runs"
    run_name: str = "experiment_01"
    telemetry_jsonl: bool = True
    telemetry_jsonl_filename: str = "telemetry.jsonl"

    # Timing and debugging
    timing_debug: bool = False
    step_progress_every: int = 0


def validate_config(cfg: TrainConfig) -> None:
    """Rigorous validation of the geometry and training menu."""
    if not torch.cuda.is_available():
        raise RuntimeError("Configuration validation failed: CUDA not found.")

    valid_options = {
        "frame_mode": ["derived", "learned_lowrank", "rotor"],
        "metric_mode": ["diag_rot"],
        "R_source": ["constitutive", "curvature"],
        "rotor_mode": ["off", "derived", "stateful"],
        "dynamics_mode": ["dissipative", "symplectic"],
        "symplectic_potential": ["zero", "harmonic", "gaussian_well"],
        "nudge_mode": ["off", "target_embedding", "template"],
    }

    for field_name, options in valid_options.items():
        value = getattr(cfg, field_name)
        if value not in options:
            raise ValueError(f"Invalid {field_name}: '{value}'. Must be one of {options}")

    if cfg.coord_dim_n <= 0:
        raise ValueError("coord_dim_n must be positive.")
    if cfg.eps <= 0:
        raise ValueError("eps must be > 0.")
    if cfg.retrieval_beta <= 0:
        raise ValueError("retrieval_beta must be > 0.")
    if cfg.beta_nudge <= 0:
        raise ValueError("beta_nudge must be > 0 (required for divide-by-beta updates).")
    if cfg.tbptt_window_steps < 1:
        raise ValueError("tbptt_window_steps must be >= 1.")

    if cfg.frame_mode == "learned_lowrank" and cfg.lowrank_r < 1:
        raise ValueError("learned_lowrank requires lowrank_r >= 1")
    if cfg.metric_mode != "diag_rot":
        raise ValueError("metric_mode must be 'diag_rot' (other modes are disabled).")
    if cfg.rotor_mode != "off" and cfg.rotor_k < 1:
        raise ValueError("Rotor-based modes require rotor_k >= 1")
    if cfg.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1.")
    if cfg.max_windows < 0:
        raise ValueError("max_windows must be >= 0.")

    # Compile / CUDA Graph constraints
    if cfg.use_torch_compile or cfg.use_cudagraphs:
        if cfg.warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1 when using torch.compile or cudagraphs.")

    if cfg.use_cudagraphs:
        if not cfg.static_shapes:
            raise ValueError("use_cudagraphs=True requires static_shapes=True.")
        if cfg.capture_batch_size <= 0:
            raise ValueError("use_cudagraphs=True requires capture_batch_size > 0.")
        if cfg.capture_batch_size < 1:
            raise ValueError("capture_batch_size must be >= 1.")

    if cfg.use_torch_compile and cfg.use_cudagraphs:
        print(
            "WARNING: use_torch_compile and use_cudagraphs are both True. "
            "This is valid only if you have a clear capture plan (static shapes, stable allocations, "
            "and compiled step function that does not graph-break)."
        )

    if cfg.R_source == "curvature" and cfg.coord_dim_n > 8:
        print(
            "WARNING: R_source='curvature' with coord_dim_n > 8 can be expensive if implemented densely. "
            "Prefer structured/low-rank curvature, constitutive R, or reduce n."
        )

    # Symplectic dynamics validation
    if cfg.dynamics_mode == "symplectic":
        if cfg.symplectic_dt <= 0:
            raise ValueError("symplectic_dt must be > 0.")
        if cfg.symplectic_m_cog <= 1e-6:
            raise ValueError("symplectic_m_cog must be positive (non-zero inertia).")
        if cfg.symplectic_stiffness < 0:
            raise ValueError("symplectic_stiffness must be non-negative.")

        # Physics Check: Stability condition for Harmonic Oscillator
        # Stormer-Verlet stability requires: dt < 2 * sqrt(m / k)
        if cfg.symplectic_potential == "harmonic" and cfg.symplectic_stiffness > 0:
            limit = 2.0 * math.sqrt(cfg.symplectic_m_cog / cfg.symplectic_stiffness)
            if cfg.symplectic_dt >= limit:
                print(
                    f"WARNING: symplectic_dt ({cfg.symplectic_dt}) exceeds stability limit "
                    f"({limit:.4f}) for current mass/stiffness. Integration may diverge."
                )

    # Nudge signal validation
    if cfg.nudge_scale < 0:
        raise ValueError("nudge_scale must be non-negative.")


def mode_signature(cfg: TrainConfig) -> str:
    """Returns a unique string representing the geometric architecture."""
    return (
        f"frame={cfg.frame_mode}|metric={cfg.metric_mode}|R={cfg.R_source}|"
        f"rotor={cfg.rotor_mode}|n={cfg.coord_dim_n}|r={cfg.lowrank_r}|k={cfg.rotor_k}"
    )

def trainer2_entrypoint(
    cfg: TrainConfig,
    *,
    model: Optional[nn.Module] = None,
    field: Any = None,
    memory: Any = None,
    train_loader: Any = None,
    hooks: Optional["StepHooks"] = None,
    rotor_state: Any = None,
    val_loader: Any = None,
    tokenizer: Any = None,
) -> None:
    """Entry hook for wiring from main; do not call at import time."""
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("trainer2_entrypoint", __file__)
    _trace("entrypoint:start")
    validate_config(cfg)
    _trace("entrypoint:config_validated")
    apply_backend_flags(cfg)
    sig = mode_signature(cfg)
    print(f"trainer2 mode: {sig}")
    _trace("entrypoint:precompute_geometry:start")
    geom = precompute_geometry(cfg)
    _trace("entrypoint:precompute_geometry:done")
    if cfg.run_smoke_tests:
        run_smoke_tests(cfg, geom)
    if hooks is None:
        raise NotImplementedError(
            "trainer2 requires StepHooks wiring (hooks=None). "
            "Provide StepHooks and a training loop before running."
        )
    if train_loader is None:
        raise NotImplementedError("trainer2 requires a train_loader.")
    if model is None:
        raise NotImplementedError("trainer2 requires a model.")
    if field is None:
        raise NotImplementedError("trainer2 requires a field.")
    if memory is None:
        raise NotImplementedError("trainer2 requires a memory object.")

    _trace("entrypoint:disable_grads:start")
    torch.set_grad_enabled(False)
    disable_gradients(model, field, memory, rotor_state)
    if isinstance(model, nn.Module):
        assert_no_autograd(model.parameters())
    _trace("entrypoint:disable_grads:done")

    model.train()
    print(
        f"[trainer2] hooks: build={hooks.build_retrieval_batch.__qualname__}, "
        f"step={hooks.step_dynamics.__qualname__}, "
        f"vel={hooks.get_velocity.__qualname__}"
    )
    print(
        f"[trainer2] memory={type(memory).__name__} field={type(field).__name__} "
        f"model={type(model).__name__}"
    )
    telemetry = TelemetryState()
    _ensure_jsonl(telemetry, cfg)
    window_idx = 0
    nudge_every = max(int(cfg.nudge_every_windows), 1)
    did_log_first_batch = False

    print(f"[{_ts()}] entering epoch loop (max_epochs={cfg.max_epochs})")
    batch_count = 0
    t_epoch_start = time.time()

    for epoch_idx in range(int(cfg.max_epochs)):
        print(f"[{_ts()}] EPOCH {epoch_idx}/{cfg.max_epochs-1} started")
        t_batch_start = time.time()
        batches_this_epoch = 0

        for batch in train_loader:
            batch_count += 1
            batches_this_epoch += 1

            # Batch received indicator
            t_recv = time.time()
            if batch_count <= 3 or batch_count % 50 == 0:
                print(f"[{_ts()}] batch {batch_count} received (epoch {epoch_idx})")

            batch = to_device(batch)
            t_device = time.time()

            if not did_log_first_batch:
                if isinstance(batch, dict):
                    shape_info = {
                        k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
                        for k, v in batch.items()
                    }
                    print(f"[{_ts()}] first batch keys: {list(batch.keys())}")
                    print(f"[{_ts()}] first batch shapes: {shape_info}")
                else:
                    print(f"[{_ts()}] first batch type: {type(batch).__name__}")
                print(f"[{_ts()}] to_device took {(t_device - t_recv)*1000:.1f}ms")
                print(f"[{_ts()}] starting window 0, epoch {epoch_idx}")
                did_log_first_batch = True

            # Run window (free or free+nudge)
            t_window_start = time.time()
            if window_idx % nudge_every == 0:
                if batch_count <= 3:
                    print(f"[{_ts()}] window {window_idx}: running two-phase (free+nudge)")
                free, _nudged = run_two_phase_and_update(
                    model=model,
                    field=field,
                    memory=memory,
                    rotor_state=rotor_state,
                    batch=batch,
                    cfg=cfg,
                    geom=geom,
                    hooks=hooks,
                    window_idx=window_idx,
                    epoch_idx=epoch_idx,
                )
                metrics = free.metrics
            else:
                if batch_count <= 3:
                    print(f"[{_ts()}] window {window_idx}: running free-only")
                free = run_window(
                    model=model,
                    field=field,
                    memory=memory,
                    rotor_state=rotor_state,
                    batch=batch,
                    cfg=cfg,
                    geom=geom,
                    hooks=hooks,
                    external_nudge=None,
                    window_idx=window_idx,
                    epoch_idx=epoch_idx,
                )
                metrics = free.metrics
            t_window_end = time.time()

            # Log timing for first few batches
            if batch_count <= 3:
                print(f"[{_ts()}] window {window_idx} took {(t_window_end - t_window_start)*1000:.1f}ms")

            # Single commit point (after free-only or free+nudge window)
            _maybe_update_memory(memory, free)

            mem_norm = memory.bank_coord.norm().item() if hasattr(memory, 'bank_coord') else 0.0
            maybe_log_metrics(
                window_idx,
                metrics,
                cfg,
                telemetry,
                epoch_idx=epoch_idx,
                window_ms=(t_window_end - t_window_start) * 1000.0,
                mem_norm=mem_norm,
                batch_idx=batch_count,
            )
            maybe_checkpoint(
                window_idx=window_idx,
                epoch_idx=epoch_idx,
                cfg=cfg,
                model=model,
                field=field,
                memory=memory,
                rotor_state=rotor_state,
                tokenizer=tokenizer,
            )

            window_idx += 1
            # Window progression diagnostic
            print(f"[PROGRESS] window={window_idx} mem_norm={mem_norm:.4f} lior={metrics.lior_mean.item():.6f}")

            if cfg.max_windows > 0 and window_idx >= cfg.max_windows:
                print(f"[{_ts()}] max_windows reached: {cfg.max_windows}")
                _close_telemetry(telemetry)
                return

        # End of epoch summary
        t_epoch_end = time.time()
        epoch_time = t_epoch_end - t_batch_start
        batches_per_sec = batches_this_epoch / max(epoch_time, 0.001)
        print(f"[{_ts()}] EPOCH {epoch_idx} done: {batches_this_epoch} batches in {epoch_time:.1f}s ({batches_per_sec:.2f} batch/s)")

    total_time = time.time() - t_epoch_start
    print(f"[{_ts()}] training complete: {batch_count} total batches in {total_time:.1f}s")
    _close_telemetry(telemetry)

# ==============================================================================
# SECTION 3: GEOMETRY PRECOMPUTE (GPU-only caches, no hot-path einsum)
# ==============================================================================

@dataclass
class GeometryCache:
    g0: torch.Tensor
    g0_inv: torch.Tensor
    g2_vec: torch.Tensor
    g0_diag: Optional[torch.Tensor] = None
    rotor_pairs: Optional[torch.Tensor] = None
    rotor_layers: Optional[torch.Tensor] = None
    U_init: Optional[torch.Tensor] = None


def build_base_metric(cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    """
    Baseline chart metric for this model. This is a model choice, not a GR rule.
    """
    n = cfg.coord_dim_n
    return torch.eye(n, device=device, dtype=torch.float32)


def precompute_geometry(cfg: TrainConfig) -> GeometryCache:
    """
    Runs once at startup (or when cfg changes). Everything stays on GPU.
    """
    device = DEVICE
    n = cfg.coord_dim_n

    g0 = build_base_metric(cfg, device=device)
    g0_inv = torch.linalg.inv(g0)

    g2 = torch.kron(g0_inv, g0_inv).contiguous()
    g2_vec = g2.reshape(-1).contiguous()

    g0_diag = torch.diagonal(g0).contiguous()

    rotor_pairs = None
    rotor_layers = None
    if cfg.rotor_mode != "off":
        if n < 2:
            raise ValueError("rotor_mode requires coord_dim_n >= 2.")
        k = cfg.rotor_k
        pairs_per_layer = n // 2
        layers = (k + pairs_per_layer - 1) // pairs_per_layer
        base = torch.arange(pairs_per_layer, device=device, dtype=torch.int64)
        layer_idx = torch.arange(layers, device=device, dtype=torch.int64)
        shift = layer_idx % n
        a = (2 * base[None, :] + shift[:, None]) % n
        b = (a + 1) % n
        rotor_layers = torch.stack([a, b], dim=-1)
        rotor_pairs = rotor_layers.reshape(-1, 2)[:k].contiguous()

    U_init = None
    if cfg.frame_mode == "learned_lowrank":
        r = cfg.lowrank_r
        U_init = (0.01 * torch.randn(n, r, device=device, dtype=torch.float32)).contiguous()

    return GeometryCache(
        g0=g0,
        g0_inv=g0_inv,
        g2_vec=g2_vec,
        g0_diag=g0_diag,
        rotor_pairs=rotor_pairs,
        rotor_layers=rotor_layers,
        U_init=U_init,
    )

# ==============================================================================
# SECTION 4: RETRIEVAL COST + ATTENTION (LIoR-consistent)
# ==============================================================================

def quad_form_batch(
    v: torch.Tensor,
    g: torch.Tensor,
    eps: float,
    g_diag: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v: [B,K,n]
    g: [n,n] or [B,n,n]
    returns: sqrt(g(v,v) + eps) -> [B,K]
    """
    if g_diag is not None and g.dim() == 2:
        quad = (v * v) * g_diag.view(1, 1, -1)
        q = quad.sum(dim=-1)
        q = torch.clamp(q, min=0.0)
        return torch.sqrt(q + eps)

    if g.dim() == 2:
        gv = torch.matmul(v, g)
        q = (gv * v).sum(dim=-1)
        q = torch.clamp(q, min=0.0)
        return torch.sqrt(q + eps)

    if g.dim() == 3:
        if g.shape[0] != v.shape[0]:
            raise ValueError("Per-batch g must match v batch size.")
        gv = torch.matmul(v, g)
        q = (gv * v).sum(dim=-1)
        q = torch.clamp(q, min=0.0)
        return torch.sqrt(q + eps)

    raise ValueError("g must be [n,n] or [B,n,n].")


# OPTIMIZATION: JIT compile for better fusion in hot path
@torch.jit.script
def retrieval_weights_from_cost(cost: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.softmax(-beta * cost, dim=-1)


@torch.jit.script
def retrieval_mix(values: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * w.unsqueeze(-1), dim=1)


def retrieval_step(
    q_coord: torch.Tensor,
    cand_coord: torch.Tensor,
    cand_state: torch.Tensor,
    R_sc: torch.Tensor,
    g: torch.Tensor,
    g_diag: Optional[torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    v = cand_coord - q_coord.unsqueeze(1)
    spd = quad_form_batch(v, g=g, eps=cfg.eps, g_diag=g_diag)
    if R_sc.dim() == 1:
        cost = R_sc.unsqueeze(1) * spd
    else:
        cost = R_sc * spd
    w = retrieval_weights_from_cost(cost, beta=cfg.retrieval_beta)
    return retrieval_mix(cand_state, w)


def retrieval_step_with_rotor(
    q_coord: torch.Tensor,
    cand_coord: torch.Tensor,
    cand_state: torch.Tensor,
    R_sc: torch.Tensor,
    g: torch.Tensor,
    g_diag: Optional[torch.Tensor],
    theta1: Optional[torch.Tensor],
    theta2: Optional[torch.Tensor],
    rotor_layers2: Optional[torch.Tensor],
    geom: "GeometryCache",
    cfg: TrainConfig,
) -> torch.Tensor:
    """
    Retrieval step with rotor transformation applied.

    Applies diagonal rotation via theta1/theta2 before computing retrieval cost.
    This is the Option 6 approach: diag/box rotation on coordinates.
    """
    # Apply rotor to query coordinates if available
    if cfg.rotor_mode != "off":
        if theta1 is not None and geom.rotor_layers is not None:
            q_coord = rotor_apply(q_coord, theta1, geom.rotor_layers)
        if theta2 is not None and rotor_layers2 is not None:
            cand_coord = rotor_apply(cand_coord, theta2, rotor_layers2)

    # Delegate to base retrieval step
    return retrieval_step(
        q_coord=q_coord,
        cand_coord=cand_coord,
        cand_state=cand_state,
        R_sc=R_sc,
        g=g,
        g_diag=g_diag,
        cfg=cfg,
    )


# ==============================================================================
# SECTION 5: LIoR PIPELINE (R4 -> scalar collapse, integrator, diagnostics)
# ==============================================================================

def R_sc_from_R4(R4: torch.Tensor, g2_vec: torch.Tensor, n: int, eps: float) -> torch.Tensor:
    B = R4.shape[0]
    R4_flat = R4.reshape(B, -1)
    if torch.is_complex(R4_flat):
        R4_flat = R4_flat.real
    R4_flat = R4_flat.float()
    g2_vec_fp32 = g2_vec.float()
    s = torch.matmul(R4_flat, g2_vec_fp32)
    s = s / float(n * n)
    return torch.sqrt(torch.abs(s) + eps)


def lior_step(
    R_sc: torch.Tensor,
    v: torch.Tensor,
    g0: torch.Tensor,
    g0_diag: Optional[torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    v2 = v.unsqueeze(1)
    spd = quad_form_batch(v2, g=g0, eps=cfg.eps, g_diag=g0_diag).squeeze(1)
    return R_sc * spd


def lior_step_fused(
    R_sc: torch.Tensor,
    v: torch.Tensor,
    g0: torch.Tensor,
    g0_diag: Optional[torch.Tensor],
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    OPTIMIZATION: Fused version that computes both lior and spd in a single pass.
    Returns (dlior, spd) to avoid redundant quad_form_batch computation.
    """
    v2 = v.unsqueeze(1)
    spd = quad_form_batch(v2, g=g0, eps=cfg.eps, g_diag=g0_diag).squeeze(1)
    dlior = R_sc * spd
    return dlior, spd


@dataclass
class WindowMetrics:
    lior_mean: torch.Tensor
    R_mean: torch.Tensor
    spd_mean: torch.Tensor


class GpuDiagnostics:
    def __init__(self) -> None:
        self.ev0 = torch.cuda.Event(enable_timing=True)
        self.ev1 = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.ev0.record()

    def stop_ms(self, sync: bool = False) -> Optional[float]:
        self.ev1.record()
        if not sync:
            return None
        torch.cuda.synchronize()
        return float(self.ev0.elapsed_time(self.ev1))

# ==============================================================================
# SECTION 6: TWO-PHASE WINDOW + MANUAL UPDATES + FALLBACKS
# ==============================================================================

@dataclass
class PhaseStats:
    metrics: WindowMetrics
    act: Any
    mem_update: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    velocity: Optional[torch.Tensor] = None  # Mean velocity for directional learning


class Snapshot:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


def _snapshot_attrs(obj: Any) -> dict:
    payload = {}
    for key, val in obj.__dict__.items():
        if torch.is_tensor(val):
            payload[key] = val.detach().clone()
        elif isinstance(val, (int, float, str, bool)) or val is None:
            payload[key] = val
        elif isinstance(val, dict):
            dict_payload = {}
            ok = True
            for k, v in val.items():
                if torch.is_tensor(v):
                    dict_payload[k] = v.detach().clone()
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    dict_payload[k] = v
                else:
                    ok = False
                    break
            if ok:
                payload[key] = {
                    "_kind": "dict",
                    "data": dict_payload,
                }
        elif isinstance(val, deque):
            items = []
            for item in val:
                if torch.is_tensor(item):
                    items.append(item.detach().clone())
                else:
                    items.append(item)
            payload[key] = {
                "_kind": "deque",
                "data": items,
                "maxlen": val.maxlen,
            }
        elif isinstance(val, (list, tuple)) and all(torch.is_tensor(x) for x in val):
            payload[key] = {
                "_kind": "sequence",
                "data": [x.detach().clone() for x in val],
                "seq_type": "tuple" if isinstance(val, tuple) else "list",
            }
    return payload


def _clone_value(v):
    if torch.is_tensor(v):
        return v.detach().clone()
    return v


def _snapshot_any(obj: Any) -> Tuple[str, Any]:
    if obj is None:
        return ("none", None)
    # Early escape for scalars - never recurse into tensor logic
    if isinstance(obj, (int, float, str, bool)):
        return ("scalar", obj)
    if hasattr(obj, "snapshot") and callable(obj.snapshot):
        return ("snapshot", obj.snapshot())
    if torch.is_tensor(obj):
        return ("tensor", obj.detach().clone())
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        state = obj.state_dict()
        return ("state_dict", {k: _clone_value(v) for k, v in state.items()})
    # DO NOT recurse into arbitrary objects - treat as opaque
    # Only objects with explicit .snapshot() or .state_dict() get snapshotted
    return ("opaque", None)


def _restore_attrs(obj: Any, payload: dict) -> None:
    for key, val in payload.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if torch.is_tensor(current) and torch.is_tensor(val):
            current.copy_(val)
        elif isinstance(current, torch.nn.Parameter) and torch.is_tensor(val):
            current.data.copy_(val)
        elif isinstance(val, dict) and val.get("_kind") == "dict":
            setattr(obj, key, val.get("data", {}))
        elif isinstance(val, dict) and val.get("_kind") == "deque":
            setattr(obj, key, deque(val["data"], maxlen=val.get("maxlen")))
        elif isinstance(val, dict) and val.get("_kind") == "sequence":
            seq = val.get("data", [])
            if val.get("seq_type") == "tuple":
                setattr(obj, key, tuple(seq))
            else:
                setattr(obj, key, list(seq))
        elif isinstance(val, (int, float, str, bool)) or val is None:
            setattr(obj, key, val)


def _restore_any(obj: Any, snap: Tuple[str, Any]) -> Any:
    kind, data = snap
    if kind == "none":
        return obj
    if kind == "scalar":
        return data  # Scalars are immutable, just return the value
    if kind == "opaque":
        return obj  # Opaque objects are not restored (not snapshotted)
    if obj is None:
        raise RuntimeError("Restore failed: target object is None.")
    if kind == "snapshot":
        obj.restore(data)
        return obj
    if kind == "tensor":
        if not torch.is_tensor(obj):
            raise RuntimeError("Restore failed: expected tensor target.")
        obj.copy_(data)
        return obj
    if kind == "state_dict":
        obj.load_state_dict(data, strict=False)
        return obj
    if kind == "attrs":
        _restore_attrs(obj, data)
        return obj
    raise RuntimeError(f"Restore failed: unknown snapshot kind '{kind}'.")


def snapshot_system(field: Any, memory: Any, rotor_state: Any) -> Snapshot:
    payload = {
        "field": _snapshot_any(field),
        "memory": _snapshot_any(memory),
    }
    payload["rotor"] = _snapshot_any(rotor_state)

    # Symplectic state: momentum P and equilibrium T_eq (trainer2-specific)
    # These are dynamically added to field by symplectic_step_dynamics
    symplectic_state = {}
    if hasattr(field, "_symplectic_P") and field._symplectic_P is not None:
        symplectic_state["P"] = field._symplectic_P.detach().clone()
    if hasattr(field, "_symplectic_T_eq") and field._symplectic_T_eq is not None:
        symplectic_state["T_eq"] = field._symplectic_T_eq.detach().clone()
    payload["symplectic"] = symplectic_state

    return Snapshot(payload)


def restore_system(field: Any, memory: Any, rotor_state_ref: Any, snap: Snapshot) -> Any:
    _restore_any(field, snap.payload["field"])
    _restore_any(memory, snap.payload["memory"])

    # Restore symplectic state (if present)
    symplectic_state = snap.payload.get("symplectic", {})
    if "P" in symplectic_state:
        if hasattr(field, "_symplectic_P") and field._symplectic_P is not None:
            field._symplectic_P.copy_(symplectic_state["P"])
        else:
            field._symplectic_P = symplectic_state["P"].clone()
    if "T_eq" in symplectic_state:
        if hasattr(field, "_symplectic_T_eq") and field._symplectic_T_eq is not None:
            field._symplectic_T_eq.copy_(symplectic_state["T_eq"])
        else:
            field._symplectic_T_eq = symplectic_state["T_eq"].clone()
    return _restore_any(rotor_state_ref, snap.payload["rotor"])

# ==============================================================================
# SECTION 6B: STEP HOOKS (single-file contract for field/memory/model evolution)
# ==============================================================================

@dataclass
class StepHooks:
    """
    Single-file integration contract.

    You implement these callables (likely as bound methods from your existing objects),
    and trainer2 stays generic and single-file.

    All tensors must be CUDA tensors. No autograd.
    """
    build_retrieval_batch: Callable[
        [nn.Module, Any, Any, Any, Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    step_dynamics: Callable[[nn.Module, Any, Any, Any, Any, Optional[torch.Tensor]], Any]
    get_velocity: Callable[[nn.Module, Any, Any, Any, Any], torch.Tensor]
    compute_R_sc: Optional[Callable[[nn.Module, Any, Any, Any, Any], torch.Tensor]] = None
    compute_R4: Optional[Callable[[nn.Module, Any, Any, Any, Any], torch.Tensor]] = None
    get_rotor_theta: Optional[Callable[[Any], Optional[torch.Tensor]]] = None
    get_rotor_thetas: Optional[
        Callable[
            [Any],
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        ]
    ] = None


@dataclass
class RotorState:
    theta: torch.Tensor
    theta2: Optional[torch.Tensor] = None
    rotor_layers2: Optional[torch.Tensor] = None


class NullMemory:
    def snapshot(self) -> dict:
        return {}

    def restore(self, _snap: dict) -> None:
        return None


class SimpleSDMMemory:
    """
    Minimal SDM-style memory for trainer2 bring-up.
    Stores pooled coords/states in a fixed-size GPU ring buffer.
    """

    def __init__(
        self,
        coord_dim_n: int,
        capacity: int = 2048,
        device: torch.device = DEVICE,
        static_shapes: bool = False,
    ):
        self.coord_dim_n = int(coord_dim_n)
        self.capacity = int(capacity)
        self.device = device
        self.static_shapes = bool(static_shapes)
        self.bank_coord: Optional[torch.Tensor] = None
        self.bank_state: Optional[torch.Tensor] = None
        self.valid_mask: Optional[torch.Tensor] = None
        self.ptr = 0
        self.filled = 0

    def snapshot(self) -> dict:
        # SDM is read-only during free/nudged windows; no snapshot copy needed.
        return {
            "ptr": int(self.ptr),
            "filled": int(self.filled),
            "coord_dim_n": int(self.coord_dim_n),
            "capacity": int(self.capacity),
            "static_shapes": bool(self.static_shapes),
        }

    def restore(self, snap: dict) -> None:
        self.coord_dim_n = int(snap.get("coord_dim_n", self.coord_dim_n))
        self.capacity = int(snap.get("capacity", self.capacity))
        self.static_shapes = bool(snap.get("static_shapes", self.static_shapes))

        self.ptr = int(snap.get("ptr", self.ptr))
        self.filled = int(snap.get("filled", self.filled))

    def _ensure_bank(self, d_state: int, dtype: torch.dtype) -> None:
        reset = False
        # CRITICAL: Create tensors OUTSIDE inference_mode to allow mutation
        # Use empty().fill_(0) pattern which creates mutable tensors even if
        # called under inference_mode context
        if self.bank_coord is None or self.bank_coord.dtype != dtype:
            with torch.inference_mode(False):
                self.bank_coord = torch.zeros(
                    self.capacity,
                    self.coord_dim_n,
                    device=self.device,
                    dtype=dtype,
                )
            reset = True
        if (
            self.bank_state is None
            or self.bank_state.dtype != dtype
            or self.bank_state.shape[1] != d_state
        ):
            with torch.inference_mode(False):
                self.bank_state = torch.zeros(
                    self.capacity,
                    d_state,
                    device=self.device,
                    dtype=dtype,
                )
            reset = True
        if self.valid_mask is None or self.valid_mask.device != self.device:
            with torch.inference_mode(False):
                self.valid_mask = torch.zeros(
                    self.capacity,
                    device=self.device,
                    dtype=torch.bool,
                )
            reset = True
        if reset:
            self.ptr = 0
            self.filled = 0
            self.valid_mask.zero_()

    def query(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.bank_coord is None or self.bank_state is None or self.valid_mask is None:
            return None, None, None
        # Always return capacity-sized banks; invalidate via mask.
        mem_coord = self.bank_coord.unsqueeze(0).expand(batch_size, -1, -1)
        mem_state = self.bank_state.unsqueeze(0).expand(batch_size, -1, -1)
        invalid = (~self.valid_mask).view(1, -1, 1).to(mem_coord.dtype)
        mem_coord = mem_coord + invalid * 1.0e4
        return mem_coord, mem_state, self.valid_mask

    def update(self, q_coord: torch.Tensor, q_state: torch.Tensor) -> None:
        self._ensure_bank(q_state.shape[-1], q_state.dtype)
        take = min(self.capacity, q_coord.shape[0])
        if take == 0:
            return
        src_coord = q_coord[-take:]
        src_state = q_state[-take:]
        end = self.ptr + take
        if end <= self.capacity:
            self.bank_coord[self.ptr:end] = src_coord
            self.bank_state[self.ptr:end] = src_state
            self.valid_mask[self.ptr:end] = True
        else:
            first = self.capacity - self.ptr
            self.bank_coord[self.ptr:] = src_coord[:first]
            self.bank_state[self.ptr:] = src_state[:first]
            self.valid_mask[self.ptr:] = True
            rest = take - first
            self.bank_coord[:rest] = src_coord[first:first + rest]
            self.bank_state[:rest] = src_state[first:first + rest]
            self.valid_mask[:rest] = True
        self.ptr = (self.ptr + take) % self.capacity
        self.filled = min(self.capacity, self.filled + take)


# ==============================================================================
# SECTION 6C: SYMPLECTIC DYNAMICS (Phase-Space Preserving Integration)
# ==============================================================================
# Implements Stormer-Verlet (Leapfrog) integration to preserve phase space volume
# (Liouville's theorem), preventing information loss ("memory rot").
#
# Physics: H(T, P) = (1/2m)||P||^2 + V(T)
#   - T: field position (cognitive state)
#   - P: conjugate momentum (rate of change)
#   - V: potential energy (restoring force for stability)
#
# The symplectic integrator exactly preserves the 2-form omega = dT ^ dP,
# ensuring that information volume is conserved even over long time horizons.
# ==============================================================================

def spatial_laplacian_trainer2(T: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Compute spatial Laplacian for 4D fields (N_x, N_y, D, D).
    For 2D vectors [B, n], returns zeros (no spatial structure to differentiate).

    Ported from kernels/hamiltonian.py for self-containment.

    Args:
        T: Tensor field - either [B, n] (vector) or [N_x, N_y, D, D] (spatial)
        dx: Grid spacing (default 1.0)

    Returns:
        Laplacian of same shape as T
    """
    if T.dim() == 2:
        # [B, n] - no spatial structure, return zeros
        return torch.zeros_like(T)

    if T.dim() != 4:
        # Unsupported shape - return zeros as safe fallback
        return torch.zeros_like(T)

    # 4D tensor: (N_x, N_y, D, D_out)
    N_x, N_y, D, D_out = T.shape

    # Reshape for 2D convolution: (1, D*D_out, N_x, N_y)
    T_reshaped = T.permute(2, 3, 0, 1).reshape(1, D * D_out, N_x, N_y)

    # Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]] / dx^2
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=T.dtype,
        device=T.device
    ).reshape(1, 1, 3, 3) / (dx * dx)

    # Expand kernel for all channels (depthwise convolution)
    kernel = kernel.repeat(D * D_out, 1, 1, 1)

    # Apply convolution with padding='same' for periodic/reflecting boundary
    import torch.nn.functional as F
    laplacian = F.conv2d(
        T_reshaped,
        kernel,
        padding=1,  # Equivalent to 'same' for 3x3 kernel
        groups=D * D_out
    )

    # Reshape back: (N_x, N_y, D, D_out)
    laplacian = laplacian.reshape(D, D_out, N_x, N_y).permute(2, 3, 0, 1)

    return laplacian


def compute_symplectic_force(
    T: torch.Tensor,
    T_equilibrium: Optional[torch.Tensor],
    cfg: "TrainConfig",
    external_nudge: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute total force F = -grad_T H + external.

    Includes:
    - Kinetic gradient: +(hbar^2/2m) * laplacian(T)  [note: + sign because F = -grad V]
    - Restoring force: -k * (T - T_eq)  (harmonic potential gradient)
    - External nudge: direct force injection

    Physics:
        For H = KE + PE where PE = (k/2)||T - T_eq||^2
        F = -dPE/dT = -k(T - T_eq)

    Args:
        T: Current field state
        T_equilibrium: Reference equilibrium state (attractor)
        cfg: TrainConfig with symplectic parameters
        external_nudge: External force (training signal)

    Returns:
        Total force tensor, same shape as T
    """
    force = torch.zeros_like(T)

    # Kinetic term (quantum potential / smoothness penalty for spatial fields)
    # This prevents "spiky" field configurations
    if T.dim() == 4 and cfg.symplectic_hbar_cog > 0:
        lap = spatial_laplacian_trainer2(T)
        # F_kinetic = +(hbar^2/2m) * laplacian (dispersive term)
        kinetic_coeff = (cfg.symplectic_hbar_cog ** 2) / (2.0 * cfg.symplectic_m_cog)
        force = force + kinetic_coeff * lap

    # Restoring force (prevents drift, creates stable attractor)
    # F_spring = -k * (T - T_eq)
    if T_equilibrium is not None and cfg.symplectic_stiffness > 0:
        force = force - cfg.symplectic_stiffness * (T - T_equilibrium)

    # External driving (training nudge acts as force injection)
    if external_nudge is not None:
        force = force + external_nudge

    return force


def symplectic_leapfrog_step(
    T: torch.Tensor,
    P: torch.Tensor,
    T_equilibrium: Optional[torch.Tensor],
    cfg: "TrainConfig",
    external_nudge: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shape-agnostic Stormer-Verlet (Leapfrog) symplectic integration step.

    Preserves phase space volume (Liouville's theorem) exactly.
    Works for [B, n] vectors or [N_x, N_y, D, D] spatial fields.

    Algorithm (velocity Verlet form):
        1. P_{n+1/2} = P_n + (dt/2) * F(T_n)           [half kick]
        2. T_{n+1}   = T_n + (dt/m) * P_{n+1/2}       [full drift]
        3. P_{n+1}   = P_{n+1/2} + (dt/2) * F(T_{n+1}) [half kick]

    Physics:
        - Exactly preserves symplectic 2-form: omega = dT ^ dP
        - Time-reversible (run backward to recover initial state)
        - Energy oscillates but doesn't drift (no secular growth)

    Args:
        T: Position (field configuration)
        P: Momentum (conjugate to T)
        T_equilibrium: Equilibrium point for restoring force
        cfg: TrainConfig with dt, m_cog, stiffness, etc.
        external_nudge: External force (training signal)

    Returns:
        (T_new, P_new): Updated position and momentum
    """
    dt = cfg.symplectic_dt
    m = cfg.symplectic_m_cog

    # Step 1: Half kick - update momentum using force at current position
    F_t = compute_symplectic_force(T, T_equilibrium, cfg, external_nudge)
    P_half = P + F_t * (0.5 * dt)

    # Step 2: Full drift - update position using half-step momentum
    T_new = T + (P_half / m) * dt

    # Step 3: Half kick - update momentum using force at new position
    F_t_new = compute_symplectic_force(T_new, T_equilibrium, cfg, external_nudge)
    P_new = P_half + F_t_new * (0.5 * dt)

    return T_new, P_new


def compute_symplectic_diagnostics(
    T: torch.Tensor,
    P: torch.Tensor,
    T_equilibrium: Optional[torch.Tensor],
    cfg: "TrainConfig"
) -> dict:
    """
    Compute conservation diagnostics for symplectic integration.

    Use these to verify the integrator is working correctly:
    - Total energy should oscillate but not drift
    - Phase space volume should remain constant

    Args:
        T: Current position
        P: Current momentum
        T_equilibrium: Equilibrium reference
        cfg: TrainConfig

    Returns:
        dict with diagnostic values
    """
    m = cfg.symplectic_m_cog
    k = cfg.symplectic_stiffness

    # Kinetic energy: KE = (1/2m) * ||P||^2
    P_norm_sq = torch.sum(P * P)
    KE = (0.5 / m) * P_norm_sq

    # Potential energy: PE = (k/2) * ||T - T_eq||^2
    PE = torch.tensor(0.0, device=T.device, dtype=T.dtype)
    if T_equilibrium is not None and k > 0:
        delta = T - T_equilibrium
        PE = (0.5 * k) * torch.sum(delta * delta)

    # Total energy (should be conserved)
    total_energy = KE + PE

    # Phase space volume proxy: ||T|| * ||P||
    # For symplectic integration, this should remain approximately constant
    T_norm = torch.sqrt(torch.sum(T * T) + 1e-12)
    P_norm = torch.sqrt(P_norm_sq + 1e-12)
    phase_space_volume = T_norm * P_norm

    return {
        "kinetic_energy": KE.item(),
        "potential_energy": PE.item(),
        "total_energy": total_energy.item(),
        "phase_space_volume": phase_space_volume.item(),
        "T_norm": T_norm.item(),
        "P_norm": P_norm.item(),
    }


def _make_sig_caller(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        raise RuntimeError(f"{name} uses *args; define explicit parameters.")
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        raise RuntimeError(f"{name} uses **args; define explicit parameters.")

    def _caller(
        model: nn.Module,
        field: Any,
        memory: Any,
        rotor_state: Any,
        batch: Any,
        external_nudge: Optional[torch.Tensor] = None,
    ) -> Any:
        T = getattr(field, "T", None)
        context = {
            "model": model,
            "field": field,
            "memory": memory,
            "rotor_state": rotor_state,
            "rotor": rotor_state,
            "batch": batch,
            "external_nudge": external_nudge,
            "external_input": external_nudge,
            "nudge": external_nudge,
            "T": T,
            "field_T": T,
            "field_state": T,
        }

        args, kwargs = [], {}
        for p in params:
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if p.name in context and context[p.name] is not None:
                    args.append(context[p.name])
                elif p.default is inspect._empty:
                    raise RuntimeError(f"{name} missing required param '{p.name}'.")
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                if p.name in context and context[p.name] is not None:
                    kwargs[p.name] = context[p.name]
                elif p.default is inspect._empty:
                    raise RuntimeError(f"{name} missing required kw param '{p.name}'.")

        return fn(*args, **kwargs)

    return _caller


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Capture-safe masked mean.

    Args:
        x:    [B, T, D]
        mask: [B, T] (float or bool; 1=keep, 0=ignore)

    Returns:
        [B, D]
    """
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)

    mask = mask.unsqueeze(-1)  # [B, T, 1]
    weighted_sum = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return weighted_sum / denom


def build_sdm_hooks(model: nn.Module, field: Any, memory: Any) -> "StepHooks":
    missing = []
    if not hasattr(memory, "query"):
        missing.append("memory.query")
    if not (hasattr(field, "step") or hasattr(field, "evolve_step")):
        missing.append("field.step or field.evolve_step")
    if missing:
        raise RuntimeError("trainer2 SDM/connection hooks missing: " + ", ".join(missing))

    if hasattr(field, "velocity"):
        get_velocity = _make_sig_caller(field.velocity, "field.velocity")
    else:
        def get_velocity(
            model: nn.Module,
            field: Any,
            memory: Any,
            rotor_state: Any,
            batch: Any,
        ) -> torch.Tensor:
            if isinstance(batch, dict):
                if "_trainer2_v" in batch:
                    return batch["_trainer2_v"]
                if "_trainer2_q_coord" in batch:
                    return batch["_trainer2_q_coord"]
            B = batch["input_ids"].shape[0] if isinstance(batch, dict) and "input_ids" in batch else 1
            return torch.zeros(B, memory.coord_dim_n, device=DEVICE, dtype=torch.float32)

    if hasattr(model, "compute_R_sc"):
        compute_R_sc = _make_sig_caller(model.compute_R_sc, "model.compute_R_sc")
    else:
        def compute_R_sc(
            model: nn.Module,
            field: Any,
            memory: Any,
            rotor_state: Any,
            batch: Any,
        ) -> torch.Tensor:
            T = getattr(field, "T", None)
            if T is None:
                raise RuntimeError("Field missing T; cannot compute default R_sc.")
            R = torch.mean(torch.abs(T))
            B = batch["input_ids"].shape[0] if isinstance(batch, dict) and "input_ids" in batch else 1
            return R.expand(B)

    step_fn = field.evolve_step if hasattr(field, "evolve_step") else field.step
    step_dynamics = _make_sig_caller(step_fn, "field.step/evolve_step")

    sig = inspect.signature(model.forward)
    supports_diagnose = "diagnose" in sig.parameters
    supports_attention_mask = "attention_mask" in sig.parameters
    supports_mask = "mask" in sig.parameters

    def _embed_batch(batch: Any) -> torch.Tensor:
        if not hasattr(model, "input_embedding") or model.input_embedding is None:
            raise RuntimeError("build_sdm_hooks requires model.input_embedding to be set.")
        modality = batch.get("modality", "text") if isinstance(batch, dict) else "text"
        if isinstance(modality, list):
            modality = modality[0]
        if modality == "text":
            x = model.input_embedding(batch["input_ids"], modality="text")
        elif modality == "image":
            x = model.input_embedding(batch["image"], modality="image")
        elif modality == "video":
            x = model.input_embedding(batch["video"], modality="video")
        else:
            raise ValueError(f"Unknown modality: {modality}")
        target_dtype = next(model.parameters()).dtype
        return x.to(target_dtype)

    def _forward_model(x: torch.Tensor, batch: Any) -> torch.Tensor:
        kwargs = {}
        if supports_diagnose:
            kwargs["diagnose"] = False
        if supports_attention_mask and isinstance(batch, dict):
            kwargs["attention_mask"] = batch.get("attention_mask", None)
        if supports_mask and isinstance(batch, dict):
            kwargs["mask"] = batch.get("attention_mask", None)
        output, _attn = model(x, field.T, time=getattr(field, "t", None), **kwargs)
        return output

    def build_retrieval_batch(
        model: nn.Module,
        field: Any,
        memory: Any,
        rotor_state: Any,
        batch: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _embed_batch(batch)
        output = _forward_model(x, batch)
        n = memory.coord_dim_n if hasattr(memory, "coord_dim_n") else output.shape[-1]
        if output.shape[-1] < n:
            pad = torch.zeros(
                output.shape[0],
                output.shape[1],
                n - output.shape[-1],
                device=output.device,
                dtype=output.dtype,
            )
            coords = torch.cat([output, pad], dim=-1)
        else:
            coords = output[..., :n]

        T = output.shape[1]
        attn_mask = _get_attention_mask(batch, output.device, T)
        q_coord = _masked_mean(coords, attn_mask)
        q_state = _masked_mean(output, attn_mask)

        if isinstance(batch, dict):
            batch["_trainer2_q_coord"] = q_coord
            batch["_trainer2_q_state"] = q_state
            batch["_trainer2_v"] = q_coord

        mem_coord, mem_state, _mask = memory.query(output.shape[0])
        if mem_coord is None or mem_state is None:
            return q_coord, coords, output
        cand_coord = torch.cat([coords, mem_coord], dim=1)
        cand_state = torch.cat([output, mem_state], dim=1)
        return q_coord, cand_coord, cand_state

    def _get_thetas(
        rotor_state: Any,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if rotor_state is None:
            return None, None, None
        if torch.is_tensor(rotor_state):
            return rotor_state, None, None
        theta1 = getattr(rotor_state, "theta", None)
        theta2 = getattr(rotor_state, "theta2", None)
        layers2 = getattr(rotor_state, "rotor_layers2", None)
        return theta1, theta2, layers2

    def _get_attention_mask(batch: Any, device: torch.device, T: int) -> torch.Tensor:
        # Derive batch size once
        if isinstance(batch, dict) and "input_ids" in batch:
            B = batch["input_ids"].shape[0]
        else:
            B = 1

        attn_mask = None
        if isinstance(batch, dict):
            attn_mask = batch.get("attention_mask", batch.get("mask", None))

        if attn_mask is None:
            # Always create [B, T]
            attn_mask = torch.ones((B, T), device=device, dtype=torch.float32)
        else:
            # Normalize to [B, T]
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device=device, dtype=torch.float32)

        return attn_mask

    return StepHooks(
        build_retrieval_batch=build_retrieval_batch,
        step_dynamics=step_dynamics,
        get_velocity=get_velocity,
        compute_R_sc=compute_R_sc,
        get_rotor_thetas=_get_thetas,
    )


def build_default_hooks(model: nn.Module, field: Any, cfg: TrainConfig) -> StepHooks:
    if not hasattr(model, "input_embedding") or model.input_embedding is None:
        raise RuntimeError("build_default_hooks requires model.input_embedding to be set.")

    sig = inspect.signature(model.forward)
    supports_diagnose = "diagnose" in sig.parameters
    supports_attention_mask = "attention_mask" in sig.parameters
    supports_mask = "mask" in sig.parameters

    def _embed_batch(batch: Any) -> torch.Tensor:
        modality = batch.get("modality", "text") if isinstance(batch, dict) else "text"
        if isinstance(modality, list):
            modality = modality[0]
        if modality == "text":
            x = model.input_embedding(batch["input_ids"], modality="text")
        elif modality == "image":
            x = model.input_embedding(batch["image"], modality="image")
        elif modality == "video":
            x = model.input_embedding(batch["video"], modality="video")
        else:
            raise ValueError(f"Unknown modality: {modality}")
        target_dtype = next(model.parameters()).dtype
        return x.to(target_dtype)

    def _forward_model(x: torch.Tensor, batch: Any) -> torch.Tensor:
        kwargs = {}
        if supports_diagnose:
            kwargs["diagnose"] = False
        if supports_attention_mask and isinstance(batch, dict):
            kwargs["attention_mask"] = batch.get("attention_mask", None)
        if supports_mask and isinstance(batch, dict):
            kwargs["mask"] = batch.get("attention_mask", None)
        output, _attn = model(x, field.T, time=getattr(field, "t", None), **kwargs)
        return output

    def build_retrieval_batch(
        model: nn.Module,
        field: Any,
        memory: Any,
        rotor_state: Any,
        batch: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _embed_batch(batch)
        output = _forward_model(x, batch)
        n = cfg.coord_dim_n
        if output.shape[-1] < n:
            pad = torch.zeros(
                output.shape[0],
                output.shape[1],
                n - output.shape[-1],
                device=output.device,
                dtype=output.dtype,
            )
            coords = torch.cat([output, pad], dim=-1)
        else:
            coords = output[..., :n]
        q_coord = coords.mean(dim=1)
        q_state = output.mean(dim=1)
        if isinstance(batch, dict):
            batch["_trainer2_v"] = q_coord
            batch["_trainer2_q_coord"] = q_coord
            batch["_trainer2_q_state"] = q_state
        return q_coord, coords, output

    def _symplectic_step_dynamics_impl(
        field: Any,
        external_nudge: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Symplectic (Stormer-Verlet) evolution preserving phase space volume.

        This replaces dissipative field.evolve_step() with a Hamiltonian
        integrator that conserves information (Liouville's theorem).

        Physics interpretation:
        - Mass (m_cog) = Cognitive inertia (resistance to changing state)
        - Stiffness (k) = Memory retention (pull back to equilibrium)
        - Nudge = External force (training signal injection)
        """
        T = field.T

        # Get or initialize momentum (field starts at rest)
        P = getattr(field, "_symplectic_P", None)
        if P is None:
            P = torch.zeros_like(T)
            field._symplectic_P = P

        # Get or initialize equilibrium point (attractor for restoring force)
        # First state becomes the "truth" that the system oscillates around
        T_eq = getattr(field, "_symplectic_T_eq", None)
        if T_eq is None:
            T_eq = T.detach().clone()
            field._symplectic_T_eq = T_eq

        # Broadcast nudge if shape mismatch (e.g., [B, n] nudge for [N,N,D,D] field)
        nudge_broadcast = None
        if external_nudge is not None:
            nudge_broadcast = external_nudge
            if nudge_broadcast.shape != T.shape:
                # Attempt broadcasting: add leading dims and expand
                while nudge_broadcast.dim() < T.dim():
                    nudge_broadcast = nudge_broadcast.unsqueeze(0)
                try:
                    nudge_broadcast = nudge_broadcast.expand_as(T)
                except RuntimeError:
                    # Shape incompatible - fallback to no nudge
                    nudge_broadcast = None

        # Symplectic leapfrog step (preserves phase space volume)
        T_new, P_new = symplectic_leapfrog_step(
            T=T,
            P=P,
            T_equilibrium=T_eq,
            cfg=cfg,
            external_nudge=nudge_broadcast
        )

        # Update field state in-place
        field.T = T_new
        field._symplectic_P = P_new

        return T_new

    def step_dynamics(
        model: nn.Module,
        field: Any,
        memory: Any,
        rotor_state: Any,
        batch: Any,
        external_nudge: Optional[torch.Tensor],
    ) -> Any:
        # Symplectic mode: preserve phase space volume (Liouville's theorem)
        if cfg.dynamics_mode == "symplectic":
            return _symplectic_step_dynamics_impl(field, external_nudge)

        # Original dissipative mode (default for backward compatibility)
        if hasattr(field, "evolve_step"):
            return field.evolve_step(external_input=external_nudge)
        if hasattr(field, "step"):
            return field.step(external_nudge)
        raise RuntimeError("Field does not expose evolve_step or step.")

    def get_velocity(
        model: nn.Module,
        field: Any,
        memory: Any,
        rotor_state: Any,
        batch: Any,
    ) -> torch.Tensor:
        if isinstance(batch, dict) and "_trainer2_v" in batch:
            return batch["_trainer2_v"]
        if isinstance(batch, dict) and "input_ids" in batch:
            B = batch["input_ids"].shape[0]
        else:
            B = 1
        return torch.zeros(B, cfg.coord_dim_n, device=DEVICE, dtype=torch.float32)

    def compute_R_sc_default(
        model: nn.Module,
        field: Any,
        memory: Any,
        rotor_state: Any,
        batch: Any,
    ) -> torch.Tensor:
        T = getattr(field, "T", None)
        if T is None:
            raise RuntimeError("Field missing T; cannot compute default R_sc.")
        R = torch.mean(torch.abs(T))
        if isinstance(batch, dict) and "input_ids" in batch:
            B = batch["input_ids"].shape[0]
        else:
            B = 1
        return R.expand(B)

    def _get_theta(rotor_state: Any) -> Optional[torch.Tensor]:
        if rotor_state is None:
            return None
        if torch.is_tensor(rotor_state):
            return rotor_state
        return getattr(rotor_state, "theta", None)

    def _get_thetas(
        rotor_state: Any,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if rotor_state is None:
            return None, None, None
        if torch.is_tensor(rotor_state):
            return rotor_state, None, None
        theta1 = getattr(rotor_state, "theta", None)
        theta2 = getattr(rotor_state, "theta2", None)
        layers2 = getattr(rotor_state, "rotor_layers2", None)
        return theta1, theta2, layers2

    return StepHooks(
        build_retrieval_batch=build_retrieval_batch,
        step_dynamics=step_dynamics,
        get_velocity=get_velocity,
        compute_R_sc=compute_R_sc_default,
        get_rotor_theta=_get_theta,
        get_rotor_thetas=_get_thetas,
    )


def _ensure_cuda(x: torch.Tensor, name: str) -> torch.Tensor:
    assert_cuda_tensor(x, name)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"{name} dtype not supported: {x.dtype}")
    return x


def _compute_R_sc_via_hooks(
    cfg: TrainConfig,
    geom: GeometryCache,
    hooks: StepHooks,
    model: nn.Module,
    field: Any,
    memory: Any,
    rotor_state: Any,
    batch: Any,
) -> torch.Tensor:
    if hooks.compute_R_sc is not None:
        R_sc = hooks.compute_R_sc(model, field, memory, rotor_state, batch)
        R_sc = _cast_R_sc(R_sc)
        _ensure_cuda(R_sc, "R_sc")
        return R_sc
    if hooks.compute_R4 is not None:
        R4 = hooks.compute_R4(model, field, memory, rotor_state, batch)
        _ensure_cuda(R4, "R4")
        return R_sc_from_R4(R4, geom.g2_vec, cfg.coord_dim_n, cfg.eps)
    raise NotImplementedError("Need hooks.compute_R_sc or hooks.compute_R4 to produce R_sc.")

# ==============================================================================
# SECTION 6C: RUN WINDOW (GPU-only, no autograd)
# ==============================================================================

def run_window(
    model: nn.Module,
    field: Any,
    memory: Any,
    rotor_state: Any,
    batch: Any,
    cfg: TrainConfig,
    geom: GeometryCache,
    hooks: StepHooks,
    external_nudge: Optional[torch.Tensor],
    window_idx: int = 0,
    epoch_idx: int = 0,
) -> PhaseStats:
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("trainer2_run_window", __file__)
    with inference_context():
        _trace("run_window:enter")
        steps = int(cfg.tbptt_window_steps)
        lior_acc = torch.zeros((), device=DEVICE, dtype=torch.float32)
        R_acc = torch.zeros((), device=DEVICE, dtype=torch.float32)
        spd_acc = torch.zeros((), device=DEVICE, dtype=torch.float32)
        velocity_acc: Optional[torch.Tensor] = None  # For directional metric learning
        ce_acc = torch.zeros((), device=DEVICE, dtype=torch.float32)
        ce_count = 0

        # Geometric diagnostics buffer
        path_buffer = PathBuffer(
            buffer_size=steps,
            dim=cfg.coord_dim_n,
            device=DEVICE
        )

        ev0 = ev1 = None
        if cfg.timing_debug:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        act = None
        mem_update: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        progress_every = int(getattr(cfg, "step_progress_every", 0) or 0)
        
        # OPTIMIZATION: Defer .item() calls to avoid per-step GPU sync
        # Store metrics on GPU, sync only at window boundary
        progress_metrics_gpu = None
        if progress_every > 0:
            progress_metrics_gpu = torch.zeros(3, device=DEVICE, dtype=torch.float32)

        for _t in range(steps):
            if _t == 0:
                _trace("run_window:step0:start")
            if progress_every > 0 and (_t % progress_every == 0) and _t > 0:
                # OPTIMIZATION: Batch metric computation on GPU, single sync
                import math
                inv_t = 1.0 / float(_t)
                progress_metrics_gpu[0] = lior_acc * inv_t
                progress_metrics_gpu[1] = R_acc * inv_t
                progress_metrics_gpu[2] = spd_acc * inv_t
                # Single .item() call for all 3 metrics
                metrics_cpu = progress_metrics_gpu.cpu()
                lior_now, R_now, spd_now = metrics_cpu[0].item(), metrics_cpu[1].item(), metrics_cpu[2].item()
                ppl = math.exp(min(lior_now, 20.0))  # Perplexity = exp(loss), capped
                print(f"[{_ts()}] e{epoch_idx} w{window_idx} step {_t}/{steps} | loss={lior_now:.4f} ppl={ppl:.2f} R={R_now:.4f} spd={spd_now:.4f}", flush=True)

            if _t == 0:
                _trace("run_window:step0:compute_R_sc:start")
            R_sc = _compute_R_sc_via_hooks(cfg, geom, hooks, model, field, memory, rotor_state, batch)
            if _t == 0:
                _trace("run_window:step0:compute_R_sc:done")
            if R_sc.dim() > 1:
                R_sc = R_sc.reshape(R_sc.shape[0])

            if _t == 0:
                _trace("run_window:step0:build_retrieval_batch:start")
            q_coord, cand_coord, cand_state = hooks.build_retrieval_batch(
                model, field, memory, rotor_state, batch
            )
            if _t == 0:
                _trace("run_window:step0:build_retrieval_batch:done")
            q_coord = _ensure_cuda(q_coord, "q_coord")
            cand_coord = _ensure_cuda(cand_coord, "cand_coord")
            cand_state = _ensure_cuda(cand_state, "cand_state")

            # Apply nudge to coordinates (Option 1: coordinate-space nudge)
            # This shifts the observer's position toward the target embedding,
            # creating a measurable difference in LIoR between free and nudged windows.
            if external_nudge is not None:
                # Nudge shape: [B, n], q_coord shape: [B, n]
                if external_nudge.shape == q_coord.shape:
                    q_coord = q_coord + external_nudge
                elif external_nudge.shape[-1] == q_coord.shape[-1]:
                    # Broadcast if batch dims differ
                    q_coord = q_coord + external_nudge.expand_as(q_coord)

            if isinstance(batch, dict):
                q_state = batch.get("_trainer2_q_state", None)
                q_coord_u = batch.get("_trainer2_q_coord", None)
                if q_state is not None and q_coord_u is not None:
                    mem_update = (q_coord_u.detach(), q_state.detach())

            theta1 = None
            theta2 = None
            layers2 = None
            if cfg.rotor_mode != "off":
                if hooks.get_rotor_thetas is not None:
                    theta1, theta2, layers2 = hooks.get_rotor_thetas(rotor_state)
                elif hooks.get_rotor_theta is not None:
                    theta1 = hooks.get_rotor_theta(rotor_state)
                if theta1 is not None:
                    _ensure_cuda(theta1, "theta1")
                if theta2 is not None:
                    _ensure_cuda(theta2, "theta2")

            if _t == 0:
                _trace("run_window:step0:retrieval_step:start")
            y = retrieval_step_with_rotor(
                q_coord=q_coord,
                cand_coord=cand_coord,
                cand_state=cand_state,
                R_sc=R_sc,
                g=geom.g0,
                g_diag=geom.g0_diag,
                theta1=theta1,
                theta2=theta2,
                rotor_layers2=layers2,
                geom=geom,
                cfg=cfg,
            )
            if _t == 0:
                _trace("run_window:step0:retrieval_step:done")
            act = y

            if _t == 0:
                _trace("run_window:step0:get_velocity:start")
            v = hooks.get_velocity(model, field, memory, rotor_state, batch)
            if _t == 0:
                _trace("run_window:step0:get_velocity:done")
            v = _ensure_cuda(v, "v")
            if v.dim() == 3:
                v = v.squeeze(1)
            if _t == 0:
                _trace("run_window:step0:lior_step:start")
            # OPTIMIZATION: Use fused version to compute both dlior and spd in one pass
            dlior, spd = lior_step_fused(R_sc=R_sc, v=v, g0=geom.g0, g0_diag=geom.g0_diag, cfg=cfg)
            if _t == 0:
                _trace("run_window:step0:lior_step:done")

            lior_acc = lior_acc + dlior.mean()
            R_acc = R_acc + R_sc.mean()
            spd_acc = spd_acc + spd.mean()

            # Accumulate velocity for directional metric learning
            # OPTIMIZATION: Use in-place add to avoid creating new tensors
            if velocity_acc is None:
                velocity_acc = v.detach().clone()
            else:
                velocity_acc.add_(v)

            # Push to geometric diagnostics buffer
            path_buffer.push(
                velocity=v.detach(),
                curvature=R_sc.detach(),
                lior=dlior.detach()
            )

            if _t == 0:
                _trace("run_window:step0:step_dynamics:start")
            hooks.step_dynamics(model, field, memory, rotor_state, batch, external_nudge)
            if _t == 0:
                _trace("run_window:step0:step_dynamics:done")

        if cfg.timing_debug and ev1 is not None and ev0 is not None:
            ev1.record()
            torch.cuda.synchronize()
            ms = float(ev0.elapsed_time(ev1))
            print(f"[{_ts()}] window gpu_ms={ms:.2f}", flush=True)

        inv_steps = 1.0 / float(steps)
        metrics = WindowMetrics(
            lior_mean=lior_acc * inv_steps,
            R_mean=R_acc * inv_steps,
            spd_mean=spd_acc * inv_steps,
        )

        # Compute and log geometric diagnostics
        if geom is not None and geom.g0_diag is not None:
            diag = path_buffer.compute_diagnostics(
                metric_diag=geom.g0_diag,
                n_perturbations=8,
                perturbation_scale=0.1
            )
            if diag is not None:
                print(f"[{_ts()}] GEOM: {format_diagnostics(diag)}")

        # Normalize velocity by number of steps
        mean_velocity = velocity_acc * inv_steps if velocity_acc is not None else None
        return PhaseStats(metrics=metrics, act=act, mem_update=mem_update, velocity=mean_velocity)


def build_nudge(
    free_stats: PhaseStats,
    batch: Any,
    cfg: TrainConfig,
    model: nn.Module = None,
) -> Optional[torch.Tensor]:
    """
    Build the nudge signal based on target embeddings.

    Physics interpretation:
        Nudge = k * (Target_coord - Current_coord)

    This creates a "force of truth" that attracts the system
    toward the ground truth embedding during the nudged phase.

    Args:
        free_stats: Statistics from free (unnudged) window run
        batch: Current batch dict with input_ids, attention_mask, etc.
        cfg: Training configuration
        model: Model with input_embedding for target lookup

    Returns:
        Nudge tensor of shape matching q_coord, or None if disabled.
    """
    # Check if nudge is disabled
    if cfg.nudge_mode == "off":
        return None

    # Legacy template mode (for backward compatibility)
    if cfg.nudge_mode == "template":
        if isinstance(batch, dict) and "nudge_template" in batch:
            return torch.zeros_like(batch["nudge_template"])
        return None

    # Target embedding mode
    if cfg.nudge_mode != "target_embedding":
        return None

    # Get current coordinates from batch (set by build_retrieval_batch)
    current_coord = batch.get("_trainer2_q_coord", None) if isinstance(batch, dict) else None
    if current_coord is None:
        return None  # No coordinates available

    # Get target embedding (need model for this)
    if model is None or not hasattr(model, "input_embedding"):
        return None

    # Compute target coordinates from next token
    if isinstance(batch, dict) and "input_ids" in batch:
        input_ids = batch["input_ids"]

        if cfg.nudge_use_shifted_target:
            # Shift right: target[t] = input[t+1]
            # Use last token as target (or pad with itself)
            if input_ids.shape[1] > 1:
                target_ids = input_ids[:, 1:]  # (B, T-1)
                # Pad to maintain shape
                last_token = input_ids[:, -1:]
                target_ids = torch.cat([target_ids, last_token], dim=1)  # (B, T)
            else:
                target_ids = input_ids
        else:
            target_ids = input_ids

        # Embed target tokens
        with torch.no_grad():
            target_emb = model.input_embedding(target_ids, modality="text")  # (B, T, D)

        # Get coordinate dimension
        n = current_coord.shape[-1]
        if target_emb.shape[-1] < n:
            # Pad to match coord dimension
            pad = torch.zeros(
                target_emb.shape[0],
                target_emb.shape[1],
                n - target_emb.shape[-1],
                device=target_emb.device,
                dtype=target_emb.dtype,
            )
            target_coords = torch.cat([target_emb, pad], dim=-1)
        else:
            target_coords = target_emb[..., :n]

        # Average over sequence (match q_coord computation)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device=target_coords.device, dtype=torch.float32)
            mask_sum = attn_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            target_coord = (target_coords * attn_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        else:
            target_coord = target_coords.mean(dim=1)  # (B, n)

        # Compute nudge: F = k * (target - current)
        nudge = cfg.nudge_scale * (target_coord - current_coord)

        return nudge

    return None


def apply_manual_update(
    model: nn.Module,
    free: PhaseStats,
    nudged: PhaseStats,
    cfg: TrainConfig,
    geom: Optional["GeometryCache"] = None,
    velocity: Optional[torch.Tensor] = None,
    rotor_state: Any = None,
) -> bool:
    """
    Apply metric + rotor update based on free vs nudged LIoR difference.

    CRITICAL: This function must NOT be wrapped in inference_context().
    InferenceMode = immutable state. Learning cannot occur inside it.

    Implements:
    1. Directional metric update: Δg_ii ∝ -ΔLIoR · γ̇_i²
    2. Rotor coupling: Δθ ∝ -ΔLIoR · angle(γ̇ in rotor plane)

    This learns both the cost (D) and orientation (R) in g = R^T D R

    Returns True if update applied cleanly, False if rejected.
    """
    # Compute LIoR difference (can be computed earlier in inference mode, but write must be outside)
    lior_diff = nudged.metrics.lior_mean - free.metrics.lior_mean
    lior_diff_val = lior_diff.item() if torch.is_tensor(lior_diff) else float(lior_diff)

    # Check for NaN/inf
    if not math.isfinite(lior_diff_val):
        print(f"[UPDATE] REJECTED: LIoR diff is {lior_diff_val}")
        return False

    eta = float(cfg.eta_update / cfg.beta_nudge)
    updated_something = False

    # Option 1: Update model.theta_update_target if it exists
    if hasattr(model, "theta_update_target"):
        tgt = getattr(model, "theta_update_target")
        if isinstance(tgt, torch.Tensor) and tgt.is_cuda:
            delta = lior_diff_val * eta
            if math.isfinite(delta):
                tgt.add_(delta)
                print(f"[UPDATE] theta_update_target += {delta:.6f} (LIoR diff={lior_diff_val:.6f})")
                updated_something = True

    # Option 2 + 3: Combined metric + rotor update
    # The metric is g = R^T D R where:
    #   - D = diag(g0_diag) learns costs
    #   - R = Givens rotations (theta) learns orientation
    #
    # Update law: Δg_ii ∝ -ΔLIoR · γ̇_i² (in rotated frame)
    # Rotor update: Δθ ∝ -ΔLIoR · angle(γ̇) (in each plane)

    if geom is not None and geom.g0_diag is not None:
        g_diag = geom.g0_diag
        if g_diag.is_cuda and velocity is not None and velocity.numel() > 0:
            # Get mean velocity
            if velocity.dim() > 1:
                v_mean = velocity.mean(dim=0)
            else:
                v_mean = velocity

            # Step 1: Rotate velocity into rotor frame (if rotors exist)
            if (rotor_state is not None and hasattr(rotor_state, 'theta') and
                rotor_state.theta is not None and geom.rotor_layers is not None):
                v_rot = rotor_apply(v_mean.unsqueeze(0), rotor_state.theta, geom.rotor_layers).squeeze(0)
            else:
                v_rot = v_mean

            # Step 2: Compute diagonal metric update in rotated frame
            v_sq = v_rot.pow(2)
            v_sq_norm = v_sq / (v_sq.sum() + 1e-8)

            # Δg_ii: if nudge helped (ΔLIoR < 0), make these directions cheaper
            delta_g = eta * (lior_diff_val) * v_sq_norm

            # Apply update to diagonal metric
            g_diag.add_(delta_g)
            g_diag.clamp_(min=0.01, max=100.0)

            v_sq_max = v_sq.max().item()
            delta_norm = delta_g.abs().mean().item()
            print(f"[UPDATE] g0_diag += directional (|Δ|={delta_norm:.6f}, v²_max={v_sq_max:.4f}, LIoR diff={lior_diff_val:.6f})")
            updated_something = True

            # Step 3: Update rotor angles to align with velocity direction
            if (rotor_state is not None and hasattr(rotor_state, 'theta') and
                rotor_state.theta is not None and geom.rotor_layers is not None):

                theta = rotor_state.theta
                if theta.is_cuda:
                    layers = geom.rotor_layers.shape[0]
                    pairs_per_layer = geom.rotor_layers.shape[1]
                    total_updates = 0
                    total_delta_theta = 0.0

                    # Rotor learning rate (smaller than metric lr)
                    rotor_lr = eta * 0.01

                    # OPTIMIZATION: Vectorize rotor update to eliminate nested .item() calls
                    # Flatten rotor_layers to process all pairs at once
                    rotor_pairs = geom.rotor_layers.reshape(-1, 2)  # [layers*pairs, 2]
                    valid_pairs = []
                    valid_k_idx = []
                    
                    for k_idx, (i_j) in enumerate(rotor_pairs):
                        i, j = int(i_j[0].item()), int(i_j[1].item())
                        if i < v_mean.shape[-1] and j < v_mean.shape[-1] and k_idx < theta.shape[-1]:
                            valid_pairs.append((i, j))
                            valid_k_idx.append(k_idx)
                    
                    if valid_pairs:
                        # Vectorized angle computation
                        i_indices = torch.tensor([p[0] for p in valid_pairs], device=DEVICE)
                        j_indices = torch.tensor([p[1] for p in valid_pairs], device=DEVICE)
                        
                        v_i = v_mean[i_indices]
                        v_j = v_mean[j_indices]
                        v_plane_mag = torch.sqrt(v_i**2 + v_j**2)
                        
                        # Filter out tiny magnitudes
                        valid_mask = v_plane_mag >= 1e-6
                        
                        if valid_mask.any():
                            v_angle = torch.atan2(v_j[valid_mask], v_i[valid_mask])
                            delta_theta = rotor_lr * (-lior_diff_val) * v_angle * v_plane_mag[valid_mask]
                            
                            # Filter finite values
                            finite_mask = torch.isfinite(delta_theta)
                            if finite_mask.any():
                                valid_k = torch.tensor([valid_k_idx[i] for i, m in enumerate(valid_mask) if m], device=DEVICE)
                                valid_k = valid_k[finite_mask]
                                delta_theta = delta_theta[finite_mask]
                                
                                # Apply updates
                                theta.index_add_(theta.dim() - 1, valid_k, delta_theta)
                                total_updates = len(delta_theta)
                                total_delta_theta = delta_theta.abs().sum().item()

                    if total_updates > 0:
                        # Wrap theta to [-π, π] (in-place)
                        theta.remainder_(2 * math.pi)
                        theta.sub_(math.pi)
                        theta.add_(2 * math.pi)
                        theta.remainder_(2 * math.pi)
                        theta.sub_(math.pi)

                        avg_delta = total_delta_theta / total_updates
                        print(f"[UPDATE] rotor: {total_updates} planes (avg |Δθ|={avg_delta:.6f})")
                        updated_something = True

        elif g_diag.is_cuda:
            # Fallback: scalar update if no velocity available
            delta_mag = abs(lior_diff_val) * eta * 0.01
            if lior_diff_val < 0:
                adjustment = 1.0 + delta_mag
            else:
                adjustment = 1.0 - delta_mag
            adjustment = max(0.95, min(1.05, adjustment))
            g_diag.mul_(adjustment)
            g_diag.clamp_(min=0.01, max=100.0)
            print(f"[UPDATE] g0_diag *= {adjustment:.6f} (scalar, no velocity, LIoR diff={lior_diff_val:.6f})")
            updated_something = True

    if not updated_something:
        print(f"[UPDATE] SKIPPED: no update target (LIoR diff={lior_diff_val:.6f})")

    return True


def _maybe_update_memory(memory: Any, stats: PhaseStats) -> None:
    if stats.mem_update is None:
        return
    if not hasattr(memory, "update"):
        return
    q_coord, q_state = stats.mem_update
    try:
        memory.update(q_coord, q_state)
    except TypeError:
        memory.update(coord=q_coord, state=q_state)


def spsa_fallback_step(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("SPSA fallback is not implemented.")


def run_two_phase_and_update(
    model: nn.Module,
    field: Any,
    memory: Any,
    rotor_state: Any,
    batch: Any,
    cfg: TrainConfig,
    geom: GeometryCache,
    hooks: StepHooks,
    window_idx: int = 0,
    epoch_idx: int = 0,
) -> Tuple[PhaseStats, PhaseStats]:
    """
    Two-phase training: free window + nudged window, then update geometry.

    CRITICAL: Updates (apply_manual_update, _maybe_update_memory) MUST happen
    OUTSIDE inference_context(). InferenceMode = immutable state.
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("trainer2_two_phase", __file__)
    # Phase 1: Run both windows inside inference mode (read-only is fine)
    snap = None
    try:
        with inference_context():
            _trace("two_phase:snapshot:start")
            snap = snapshot_system(field, memory, rotor_state)
            _trace("two_phase:snapshot:done")

            _trace("two_phase:free_window:start")
            free = run_window(
                model=model, field=field, memory=memory, rotor_state=rotor_state,
                batch=batch, cfg=cfg, geom=geom, hooks=hooks, external_nudge=None,
                window_idx=window_idx, epoch_idx=epoch_idx
            )
            _trace("two_phase:free_window:done")

            _trace("two_phase:build_nudge:start")
            nudge = build_nudge(free, batch, cfg, model=model)
            _trace("two_phase:build_nudge:done")
            _trace("two_phase:restore:start")
            restore_system(field, memory, rotor_state, snap)
            _trace("two_phase:restore:done")

            _trace("two_phase:nudged_window:start")
            nudged = run_window(
                model=model, field=field, memory=memory, rotor_state=rotor_state,
                batch=batch, cfg=cfg, geom=geom, hooks=hooks, external_nudge=nudge,
                window_idx=window_idx, epoch_idx=epoch_idx
            )
            _trace("two_phase:nudged_window:done")
    finally:
        # Explicit cleanup of snapshot to free GPU memory
        if snap is not None:
            del snap
            snap = None

    # Phase 2: Apply updates OUTSIDE inference mode (mutations allowed)
    _trace("two_phase:apply_update:start")

    # Get velocity from PhaseStats if available
    velocity = getattr(free, 'velocity', None)

    # Apply metric + rotor update (directional learning)
    ok = apply_manual_update(
        model=model,
        free=free,
        nudged=nudged,
        cfg=cfg,
        geom=geom,
        velocity=velocity,
        rotor_state=rotor_state,
    )
    _trace("two_phase:apply_update:done")

    # SPSA fallback if update failed
    if (not ok) and cfg.enable_spsa_fallback and hasattr(model, "theta_update_target"):
        theta = getattr(model, "theta_update_target")
        if isinstance(theta, torch.Tensor) and theta.is_cuda:
            def obj_fn(th: torch.Tensor) -> torch.Tensor:
                old = theta.clone()
                theta.copy_(th)
                snap2 = None
                try:
                    with inference_context():
                        snap2 = snapshot_system(field, memory, rotor_state)
                        stats = run_window(
                            model=model, field=field, memory=memory, rotor_state=rotor_state,
                            batch=batch, cfg=cfg, geom=geom, hooks=hooks, external_nudge=None,
                            window_idx=window_idx, epoch_idx=epoch_idx
                        )
                        restore_system(field, memory, rotor_state, snap2)
                finally:
                    if snap2 is not None:
                        del snap2
                theta.copy_(old)
                return stats.metrics.lior_mean

            spsa_step(theta, obj_fn, cfg)

    # Memory update OUTSIDE inference mode
    _maybe_update_memory(memory, free)

    # OPTIMIZATION: Adaptive GPU memory cleanup based on usage, not periodic
    # Only clear cache if memory usage is above 90% to avoid blocking stalls
    if window_idx > 0 and window_idx % 50 == 0:  # Check less frequently
        mem_allocated = torch.cuda.memory_allocated(DEVICE)
        mem_reserved = torch.cuda.memory_reserved(DEVICE)
        if mem_allocated / mem_reserved > 0.9:
            torch.cuda.empty_cache()

    return free, nudged

# ==============================================================================
# SECTION 7: STRUCTURED R / CURVATURE SOURCES AND REPRESENTATIONS
# ==============================================================================

@dataclass
class RHooks:
    constitutive_R_sc: Optional[Callable[..., torch.Tensor]] = None
    constitutive_R4: Optional[Callable[..., torch.Tensor]] = None
    curvature_R_sc: Optional[Callable[..., torch.Tensor]] = None
    curvature_R4: Optional[Callable[..., torch.Tensor]] = None


def R_repr_signature(cfg: TrainConfig) -> str:
    return f"R={cfg.R_source}|n={cfg.coord_dim_n}|r={cfg.lowrank_r}"


def _cast_R_sc(R_sc: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(R_sc):
        R_sc = R_sc.real
    R_sc = R_sc.float()
    return R_sc.contiguous()


def compute_R_sc(cfg: TrainConfig, geom: GeometryCache, hooks: RHooks, **kwargs: Any) -> torch.Tensor:
    if cfg.R_source == "constitutive":
        if hooks.constitutive_R_sc is not None:
            R_sc = hooks.constitutive_R_sc(**kwargs)
        elif hooks.constitutive_R4 is not None:
            R4 = hooks.constitutive_R4(**kwargs)
            R_sc = R_sc_from_R4(R4, geom.g2_vec, cfg.coord_dim_n, cfg.eps)
        else:
            raise NotImplementedError("Constitutive R source requires a compute hook.")
    else:
        if hooks.curvature_R_sc is not None:
            R_sc = hooks.curvature_R_sc(**kwargs)
        elif hooks.curvature_R4 is not None:
            R4 = hooks.curvature_R4(**kwargs)
            R_sc = R_sc_from_R4(R4, geom.g2_vec, cfg.coord_dim_n, cfg.eps)
        else:
            raise NotImplementedError("Curvature R source requires a compute hook.")

    R_sc = _cast_R_sc(R_sc)
    assert_cuda_tensor(R_sc, "R_sc")
    return R_sc


def compute_R4(cfg: TrainConfig, hooks: RHooks, **kwargs: Any) -> torch.Tensor:
    if cfg.R_source == "constitutive" and hooks.constitutive_R4 is not None:
        R4 = hooks.constitutive_R4(**kwargs)
    elif cfg.R_source == "curvature" and hooks.curvature_R4 is not None:
        R4 = hooks.curvature_R4(**kwargs)
    else:
        raise NotImplementedError("R4 not available for this source; provide a compute hook.")
    assert_cuda_tensor(R4, "R4")
    return R4

# ==============================================================================
# SECTION 8: FRAME / ROTOR SUBSYSTEM (INTERFACE ORIENTATION)
# ==============================================================================

def wrap_theta(theta: torch.Tensor, wrap: float) -> torch.Tensor:
    period = float(wrap)
    return torch.remainder(theta + 0.5 * period, period) - 0.5 * period


def rotor_compose(theta: torch.Tensor, dtheta: torch.Tensor, wrap: float) -> torch.Tensor:
    return wrap_theta(theta + dtheta, wrap)


def rotor_apply(
    v: torch.Tensor,
    theta: Optional[torch.Tensor],
    rotor_layers: Optional[torch.Tensor],
) -> torch.Tensor:
    if theta is None or rotor_layers is None:
        return v
    v_out = v
    squeeze = False
    if v_out.dim() == 2:
        v_out = v_out.unsqueeze(1)
        squeeze = True

    B = v_out.shape[0]
    if theta.dim() == 1:
        theta_work = theta.unsqueeze(0).expand(B, -1)
    else:
        theta_work = theta
        if theta_work.shape[0] != B:
            raise ValueError("theta batch dimension must match v batch dimension.")

    layers, pairs_per_layer, _ = rotor_layers.shape
    total_pairs = layers * pairs_per_layer
    k = theta_work.shape[1]
    if k < total_pairs:
        theta_pad = torch.zeros(B, total_pairs, device=theta_work.device, dtype=theta_work.dtype)
        theta_pad[:, :k] = theta_work
    elif k > total_pairs:
        theta_pad = theta_work[:, :total_pairs]
    else:
        theta_pad = theta_work

    theta_layers = theta_pad.view(B, layers, pairs_per_layer)

    for layer_idx in range(layers):
        pairs = rotor_layers[layer_idx]
        a = pairs[:, 0]
        b = pairs[:, 1]
        angle = theta_layers[:, layer_idx, :]
        sin_a = torch.sin(angle).unsqueeze(1)
        cos_a = torch.cos(angle).unsqueeze(1)
        va = v_out.index_select(-1, a)
        vb = v_out.index_select(-1, b)
        va_new = cos_a * va - sin_a * vb
        vb_new = sin_a * va + cos_a * vb
        v_out.index_copy_(-1, a, va_new)
        v_out.index_copy_(-1, b, vb_new)

    if squeeze:
        v_out = v_out.squeeze(1)
    return v_out


def apply_rotor_if_present(
    v: torch.Tensor,
    theta: Optional[torch.Tensor],
    rotor_layers: Optional[torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    if cfg.rotor_mode == "off" or theta is None or rotor_layers is None:
        return v
    return rotor_apply(v, theta, rotor_layers)


def apply_rotors_parallel(
    v: torch.Tensor,
    theta1: Optional[torch.Tensor],
    layers1: Optional[torch.Tensor],
    theta2: Optional[torch.Tensor],
    layers2: Optional[torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    v1 = apply_rotor_if_present(v, theta1, layers1, cfg)
    v2 = apply_rotor_if_present(v, theta2, layers2, cfg)
    if theta1 is None and theta2 is None:
        return v
    if theta1 is None:
        return v2
    if theta2 is None:
        return v1
    return v1 + v2 - v


def maybe_apply_rotor(
    v: torch.Tensor,
    theta: Optional[torch.Tensor],
    geom: GeometryCache,
    cfg: TrainConfig,
) -> torch.Tensor:
    if cfg.rotor_mode == "off":
        return v
    return rotor_apply(v, theta, geom.rotor_layers)

# ==============================================================================
# SECTION 9: ACTIVITY LOGGING (GPU-SAFE, NO HOT-PATH HOOKS)
# ==============================================================================

@dataclass
class LinearActivityBuffers:
    yx: torch.Tensor
    y: torch.Tensor
    count: torch.Tensor


@dataclass
class LinearActivityStats:
    yx_mean: torch.Tensor
    y_mean: torch.Tensor
    count: torch.Tensor


def make_linear_activity_buffers(
    out_dim: int,
    in_dim: int,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> LinearActivityBuffers:
    return LinearActivityBuffers(
        yx=torch.zeros(out_dim, in_dim, device=device, dtype=dtype),
        y=torch.zeros(out_dim, device=device, dtype=dtype),
        count=torch.zeros((), device=device, dtype=torch.float32),
    )


def activity_reset_linear(buf: LinearActivityBuffers) -> None:
    buf.yx.zero_()
    buf.y.zero_()
    buf.count.zero_()


def activity_accumulate_linear(buf: LinearActivityBuffers, x: torch.Tensor, y: torch.Tensor) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    y2 = y.reshape(-1, y.shape[-1])
    buf.yx.add_(torch.matmul(y2.transpose(0, 1), x2))
    buf.y.add_(y2.sum(dim=0))
    buf.count.add_(x2.shape[0])


def activity_finalize_linear(buf: LinearActivityBuffers) -> LinearActivityStats:
    count = torch.clamp(buf.count, min=1.0)
    return LinearActivityStats(
        yx_mean=buf.yx / count,
        y_mean=buf.y / count,
        count=buf.count,
    )

# ==============================================================================
# SECTION 10: MANUAL UPDATE LIBRARY (CONTRASTIVE + STRUCTURED PARAMS)
# ==============================================================================

def clamp_update_norm(delta: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    if max_norm is None or max_norm <= 0:
        return delta
    norm = torch.linalg.vector_norm(delta)
    scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return delta * scale


def apply_contrastive_linear_update(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],#stop  fucking  changing  this!  leave  it  in  brackets
    free_stats: LinearActivityStats,
    nudged_stats: LinearActivityStats,
    cfg: TrainConfig,
    max_norm: Optional[float] = None,
) -> None:
    with inference_context():
        scale = cfg.eta_update / cfg.beta_nudge
        delta_w = (nudged_stats.yx_mean - free_stats.yx_mean) * scale
        delta_w = clamp_update_norm(delta_w, max_norm)
        if not torch.isfinite(delta_w).all():
            return
        weight.add_(delta_w)

        if bias is not None:
            delta_b = (nudged_stats.y_mean - free_stats.y_mean) * scale
            if torch.isfinite(delta_b).all():
                bias.add_(delta_b)

# ==============================================================================
# SECTION 11: SPSA FALLBACK (EMERGENCY ONLY, GRAPH-AWARE)
# ==============================================================================

def spsa_step(
    theta: torch.Tensor,
    objective_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    cfg: TrainConfig,
) -> torch.Tensor:
    if cfg.use_cudagraphs:
        raise NotImplementedError("SPSA is not supported under CUDA graphs in this skeleton.")
    if objective_fn is None:
        raise NotImplementedError("SPSA requires an objective_fn hook.")

    with inference_context():
        for _ in range(cfg.spsa_directions):
            delta = torch.sign(torch.randn_like(theta))
            loss_plus = objective_fn(theta + cfg.spsa_sigma * delta)
            loss_minus = objective_fn(theta - cfg.spsa_sigma * delta)
            grad_est = (loss_plus - loss_minus) / (2.0 * cfg.spsa_sigma) * delta
            theta.add_(-cfg.spsa_lr * grad_est)
    return theta

# ==============================================================================
# SECTION 12: COMPILE / INDUCTOR / CUDA GRAPH INTEGRATION
# ==============================================================================

def maybe_compile_step_fn(step_fn: Callable[..., Any], cfg: TrainConfig) -> Callable[..., Any]:
    if not cfg.use_torch_compile:
        return step_fn
    return torch.compile(step_fn, mode="reduce-overhead")


def maybe_capture_cudagraph(step_fn: Callable[..., Any], static_inputs: Any, cfg: TrainConfig) -> Callable[..., Any]:
    """
    Capture a CUDA graph for the step function if enabled.
    
    Requirements:
    - static_shapes=True: All tensors must have fixed shapes
    - capture_batch_size > 0: Fixed batch size for capture
    - warmup_steps >= 1: Number of warmup iterations before capture
    
    Returns a wrapper that replays the captured graph.
    """
    if not cfg.use_cudagraphs:
        return step_fn
    
    # CUDA graphs require PyTorch 1.10+ and CUDA 11+
    if not hasattr(torch.cuda, 'CUDAGraph'):
        print("WARNING: CUDA graphs not available in this PyTorch version. Falling back to eager mode.")
        return step_fn
    
    print(f"[CUDAGRAPH] Preparing CUDA graph capture (batch_size={cfg.capture_batch_size})")
    
    # State for graph replay
    graph_state = {
        'graph': None,
        'static_input': None,
        'static_output': None,
        'warmup_done': False,
        'warmup_count': 0,
    }
    
    def wrapped_fn(*args, **kwargs):
        # Warmup phase: run eagerly to stabilize allocations
        if not graph_state['warmup_done']:
            graph_state['warmup_count'] += 1
            result = step_fn(*args, **kwargs)
            
            if graph_state['warmup_count'] >= cfg.warmup_steps:
                graph_state['warmup_done'] = True
                print(f"[CUDAGRAPH] Warmup complete ({cfg.warmup_steps} steps). Capturing graph...")
                
                # Capture graph
                try:
                    graph = torch.cuda.CUDAGraph()
                    
                    # Create static tensors for capture (clone inputs)
                    static_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            static_args.append(arg.clone())
                        else:
                            static_args.append(arg)
                    
                    # Synchronize before capture
                    torch.cuda.synchronize()
                    
                    # Capture the graph
                    with torch.cuda.graph(graph):
                        static_output = step_fn(*static_args, **kwargs)
                    
                    graph_state['graph'] = graph
                    graph_state['static_input'] = static_args
                    graph_state['static_output'] = static_output
                    
                    print("[CUDAGRAPH] Graph captured successfully!")
                    
                except Exception as e:
                    print(f"[CUDAGRAPH] Capture failed: {e}")
                    print("[CUDAGRAPH] Falling back to eager mode")
                    graph_state['graph'] = None
            
            return result
        
        # Graph replay phase
        if graph_state['graph'] is not None:
            # Copy input data to static buffers
            for static_arg, arg in zip(graph_state['static_input'], args):
                if isinstance(static_arg, torch.Tensor) and isinstance(arg, torch.Tensor):
                    static_arg.copy_(arg)
            
            # Replay graph
            graph_state['graph'].replay()
            
            # Return static output (no copy needed, it's updated in-place)
            return graph_state['static_output']
        else:
            # Fallback to eager if capture failed
            return step_fn(*args, **kwargs)
    
    return wrapped_fn


@dataclass
class SpectralHeadConfig:
    spectral_dim: int = 8
    spectral_beta: float = 1.0
    use_spectral_in_retrieval: bool = False


@dataclass
class SpectralCoords:
    spec: torch.Tensor  # [B, T, d_spec]


class SpectralCoordinateHead(nn.Module):
    """
    Minimal spectral head: projects token states -> spectral coordinates.

    Design constraints:
    - stateless in forward (no side effects)
    - deterministic
    - CUDA-only tensors (enforced by caller)
    - no nonlinear routing/gating
    """
    def __init__(self, d_model: int, cfg: SpectralHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(d_model, cfg.spectral_dim, bias=False)

    def forward(self, x: torch.Tensor) -> SpectralCoords:
        spec = self.proj(x)
        return SpectralCoords(spec=spec)


def _spectral_cost(
    q_spec: torch.Tensor,
    cand_spec: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Simple SPD-ish spectral distance proxy.

    q_spec:    [B, dS]
    cand_spec: [B, K, dS]
    returns:   [B, K]
    """
    dv = cand_spec - q_spec.unsqueeze(1)
    q = (dv * dv).sum(dim=-1)
    q = torch.clamp(q, min=0.0)
    return torch.sqrt(q + eps)


def retrieval_step_with_spectral(
    *,
    q_coord: torch.Tensor,
    cand_coord: torch.Tensor,
    cand_state: torch.Tensor,
    R_sc: torch.Tensor,
    g: torch.Tensor,
    g_diag: Optional[torch.Tensor],
    q_spec: Optional[torch.Tensor],
    cand_spec: Optional[torch.Tensor],
    cfg: TrainConfig,
    spectral_cfg: SpectralHeadConfig,
) -> torch.Tensor:
    """
    Base geometric retrieval cost + optional spectral additive cost.

    Important: this only changes *relative weights* via softmax over a cost;
    it does not inject energy/mass.
    """
    v = cand_coord - q_coord.unsqueeze(1)
    spd = quad_form_batch(v, g=g, eps=cfg.eps, g_diag=g_diag)

    if R_sc.dim() == 1:
        cost = R_sc.unsqueeze(1) * spd
    else:
        cost = R_sc * spd

    if spectral_cfg.use_spectral_in_retrieval and (q_spec is not None) and (cand_spec is not None):
        s_cost = _spectral_cost(q_spec=q_spec, cand_spec=cand_spec, eps=cfg.eps)
        cost = cost + float(spectral_cfg.spectral_beta) * s_cost

    w = retrieval_weights_from_cost(cost, beta=cfg.retrieval_beta)
    return retrieval_mix(cand_state, w)

# ==============================================================================
# SECTION 13: TELEMETRY, LOGGING, AND CHECKPOINT POLICY
# ==============================================================================

def _telemetry_config_hash(cfg: TrainConfig) -> str:
    payload = dataclasses.asdict(cfg)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _telemetry_path(cfg: TrainConfig) -> str:
    filename = cfg.telemetry_jsonl_filename or "telemetry.jsonl"
    if not filename.endswith(".jsonl"):
        filename = f"{filename}.jsonl"
    prefix = cfg.run_name or "run"
    return os.path.join(cfg.run_dir, f"{prefix}_{filename}")


def _write_jsonl(state: "TelemetryState", record: dict) -> None:
    if state.jsonl_fp is None:
        return
    state.jsonl_fp.write(json.dumps(record, ensure_ascii=True) + "\n")
    state.jsonl_fp.flush()


def _ensure_jsonl(state: "TelemetryState", cfg: TrainConfig) -> None:
    if not cfg.telemetry_jsonl:
        return
    if state.jsonl_fp is not None:
        return
    os.makedirs(cfg.run_dir, exist_ok=True)
    state.jsonl_path = _telemetry_path(cfg)
    state.config_hash = state.config_hash or _telemetry_config_hash(cfg)
    state.jsonl_fp = open(state.jsonl_path, "a", encoding="ascii")
    meta = {
        "type": "run_meta",
        "ts_utc": _ts_utc(),
        "run_name": cfg.run_name,
        "config_hash": state.config_hash,
        "mode_signature": mode_signature(cfg),
        "tbptt_window_steps": cfg.tbptt_window_steps,
        "log_every_windows": cfg.log_every_windows,
        "save_every_windows": cfg.save_every_windows,
    }
    _write_jsonl(state, meta)


def _close_telemetry(state: "TelemetryState") -> None:
    if state.jsonl_fp is None:
        return
    try:
        state.jsonl_fp.flush()
    finally:
        state.jsonl_fp.close()
    state.jsonl_fp = None


@dataclass
class TelemetryState:
    last_log_window: int = -1
    buffer: Optional[torch.Tensor] = None  # shape [3, N]
    ptr: int = 0
    flush_stream: Optional[torch.cuda.Stream] = None
    staging_cpu: Optional[torch.Tensor] = None  # pinned [3]
    jsonl_path: Optional[str] = None
    jsonl_fp: Optional[object] = None
    config_hash: Optional[str] = None


def _ensure_flush_stream(state: TelemetryState) -> None:
    if state.flush_stream is None:
        state.flush_stream = torch.cuda.Stream()


def _ensure_staging_cpu(state: TelemetryState, dtype: torch.dtype) -> None:
    if state.staging_cpu is None or state.staging_cpu.dtype != dtype or state.staging_cpu.numel() != 3:
        state.staging_cpu = torch.empty((3,), device="cpu", dtype=dtype, pin_memory=True)


def _log_metrics_buffered(
    window_idx: int,
    metrics: WindowMetrics,
    cfg: TrainConfig,
    state: TelemetryState,
) -> Optional[Tuple[float, float, float]]:
    """Buffer metrics on GPU; async flush to host at log interval (side stream + pinned staging)."""
    if cfg.log_every_windows <= 0:
        return None

    if state.buffer is None:
        buf_len = max(int(cfg.log_every_windows), 1)
        dtype = metrics.lior_mean.dtype
        device = metrics.lior_mean.device
        state.buffer = torch.zeros((3, buf_len), device=device, dtype=dtype)
        state.ptr = 0

    buf = state.buffer
    idx = state.ptr % buf.shape[1]
    buf[0, idx] = metrics.lior_mean
    buf[1, idx] = metrics.R_mean
    buf[2, idx] = metrics.spd_mean
    state.ptr += 1

    if window_idx % cfg.log_every_windows != 0:
        return None
    if state.last_log_window == window_idx:
        return None
    state.last_log_window = window_idx

    means_gpu = buf.mean(dim=1)

    _ensure_flush_stream(state)
    _ensure_staging_cpu(state, means_gpu.dtype)

    stream = state.flush_stream
    staging = state.staging_cpu
    assert stream is not None
    assert staging is not None

    with torch.cuda.stream(stream):
        staging.copy_(means_gpu, non_blocking=True)

    # Sync only the side stream (printing is outside any capture path)
    stream.synchronize()

    lior = float(staging[0])
    r_mean = float(staging[1])
    spd = float(staging[2])
    print(f"[{_ts()}] window={window_idx} lior={lior:.6f} R={r_mean:.6f} spd={spd:.6f}")
    return (lior, r_mean, spd)


def maybe_log_metrics(
    window_idx: int,
    metrics: WindowMetrics,
    cfg: TrainConfig,
    state: TelemetryState,
    *,
    epoch_idx: Optional[int] = None,
    window_ms: Optional[float] = None,
    mem_norm: Optional[float] = None,
    batch_idx: Optional[int] = None,
) -> None:
    # Buffered logging; avoid per-step host syncs.
    means = _log_metrics_buffered(window_idx, metrics, cfg, state)
    if means is None:
        return
    if not cfg.telemetry_jsonl:
        return

    _ensure_jsonl(state, cfg)
    lior, r_mean, spd = means
    record = {
        "type": "window_metrics",
        "ts_utc": _ts_utc(),
        "run_name": cfg.run_name,
        "config_hash": state.config_hash,
        "epoch": epoch_idx,
        "window": window_idx,
        "batch": batch_idx,
        "total_loss": lior,
        "lior_mean": lior,
        "R_mean": r_mean,
        "spd_mean": spd,
        "window_ms": window_ms,
        "mem_norm": mem_norm,
        "tbptt_window_steps": cfg.tbptt_window_steps,
    }
    _write_jsonl(state, record)


def _get_tokenizer_dict(tokenizer: Any) -> Optional[dict]:
    """Extract tokenizer state for checkpoint."""
    if tokenizer is None:
        return None
    return {
        'vocab_size': getattr(tokenizer, 'vocab_size', None),
        'vocab': getattr(tokenizer, 'vocab', None),
        'inverse_vocab': getattr(tokenizer, 'inverse_vocab', None),
    }


def maybe_checkpoint(*args: Any, **kwargs: Any) -> None:
    window_idx = kwargs.get("window_idx", None)
    cfg = kwargs.get("cfg", None)
    model = kwargs.get("model", None)
    field = kwargs.get("field", None)
    memory = kwargs.get("memory", None)
    rotor_state = kwargs.get("rotor_state", None)
    epoch_idx = kwargs.get("epoch_idx", 0)
    tokenizer = kwargs.get("tokenizer", None)

    if cfg is None or window_idx is None:
        return
    if cfg.save_every_windows <= 0:
        return
    if window_idx % cfg.save_every_windows != 0:
        return

    os.makedirs(cfg.run_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.run_dir, f"{cfg.run_name}_window{window_idx}.pt")

    # Build inference-compatible config
    inference_config = {
        'd_model': getattr(model, 'd_model', 256),
        'n_layers': getattr(model, 'n_layers', cfg.coord_dim_n),
        'n_heads': getattr(model, 'n_heads', 4),
        'n_attention_layers': getattr(model, 'n_attention_layers', 2),
        'field_dim': 16,  # Hardcoded in GeometricStack
        'vocab_size': model.lm_head.out_features if hasattr(model, 'lm_head') and model.lm_head is not None else 32000,
        'max_seq_len': 512,
        'spatial_size': [8, 8],
        'use_mamba': True,
    }

    field_state_dict = None
    if hasattr(field, "state_dict") and callable(field.state_dict):
        try:
            field_state_dict = field.state_dict()
        except Exception:
            field_state_dict = None
    if field_state_dict is None:
        field_state_dict = _snapshot_any(field)

    ckpt = {
        # Inference-compatible keys (required by inference/inference.py)
        "epoch": epoch_idx,
        "global_step": window_idx,
        "config": inference_config,
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
        "field_state_dict": field_state_dict,
        "input_embedding_state_dict": model.input_embedding.state_dict() if hasattr(model, 'input_embedding') and model.input_embedding is not None else None,
        "lm_head_state_dict": model.lm_head.state_dict() if hasattr(model, 'lm_head') and model.lm_head is not None else None,
        "tokenizer": _get_tokenizer_dict(tokenizer),
        # trainer2-specific keys (for resume)
        "window_idx": window_idx,
        "epoch_idx": epoch_idx,
        "cfg": dataclasses.asdict(cfg),
        "model_state": model.state_dict() if hasattr(model, "state_dict") else None,
        "field_state": _snapshot_any(field),
        "memory_state": _snapshot_any(memory),
        "rotor_state": _snapshot_any(rotor_state),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[{_ts()}] checkpoint saved: {ckpt_path}")

# ==============================================================================
# SECTION 14: ENTRYPOINT, SMOKE TESTS, AND STRICT FAILURE SEMANTICS
# ==============================================================================

def test_R_sc_from_R4_shapes(cfg: TrainConfig, geom: GeometryCache) -> None:
    B = 2
    n = cfg.coord_dim_n
    R4 = torch.randn(B, n, n, n, n, device=DEVICE, dtype=torch.float32)
    R_sc = R_sc_from_R4(R4, geom.g2_vec, n, cfg.eps)
    if R_sc.shape != (B,):
        raise RuntimeError("R_sc_from_R4 output shape mismatch.")
    if not torch.isfinite(R_sc).all():
        raise RuntimeError("R_sc_from_R4 produced non-finite values.")


def test_quad_form_diag_vs_dense(cfg: TrainConfig, geom: GeometryCache) -> None:
    B = 2
    K = 3
    n = cfg.coord_dim_n
    v = torch.randn(B, K, n, device=DEVICE, dtype=torch.float32)
    diag = torch.rand(n, device=DEVICE, dtype=torch.float32) + 0.1
    g = torch.diag(diag)
    spd_diag = quad_form_batch(v, g=g, eps=cfg.eps, g_diag=diag)
    spd_dense = quad_form_batch(v, g=g, eps=cfg.eps, g_diag=None)
    if not torch.allclose(spd_diag, spd_dense, rtol=1e-4, atol=1e-4):
        raise RuntimeError("quad_form_batch diag vs dense mismatch.")


def test_rotor_apply_norm_preservation(cfg: TrainConfig, geom: GeometryCache) -> None:
    if geom.rotor_layers is None:
        return
    B = 2
    n = cfg.coord_dim_n
    v = torch.randn(B, n, device=DEVICE, dtype=torch.float32)
    k = geom.rotor_layers.shape[0] * geom.rotor_layers.shape[1]
    theta = torch.zeros(k, device=DEVICE, dtype=torch.float32)
    v_rot = rotor_apply(v, theta, geom.rotor_layers)
    n0 = torch.linalg.vector_norm(v, dim=-1)
    n1 = torch.linalg.vector_norm(v_rot, dim=-1)
    if not torch.allclose(n0, n1, rtol=1e-5, atol=1e-5):
        raise RuntimeError("rotor_apply does not preserve norm.")


def run_smoke_tests(cfg: TrainConfig, geom: GeometryCache) -> None:
    test_R_sc_from_R4_shapes(cfg, geom)
    test_quad_form_diag_vs_dense(cfg, geom)
    test_rotor_apply_norm_preservation(cfg, geom)

# Original outline preserved below for reference during implementation.
_OUTLINE = r"""
Trainer2 Outline (single-file plan)

Key function variables (top-level):
- config: dict for all mode switches and numeric knobs
- device: torch.device, must be cuda
- dtype: torch.dtype for model and geometry
- model: geometric transformer or equivalent
- field: CognitiveTensorField or equivalent
- train_loader: DataLoader for training
- val_loader: DataLoader for validation (optional)
- tokenizer: tokenizer for text paths (optional)
- frame_mode: "derived" | "learned_lowrank" | "rotor"
- metric_mode: "diag_rot"
- R_source: "constitutive" | "curvature"
- rotor_mode: "off" | "derived" | "stateful"
- g0: base metric (n x n)
- g0_inv: inverse metric (n x n)
- K: rank-4 contraction kernel for R_sc (n x n x n x n)
- lambda_local: diagonal local stiffness (n,)
- lambda_mem: diagonal memory stiffness (n,)
- U_mem: low-rank memory directions (n x r)
- D_mem: low-rank weights (r,)
- Q: frame / rotor matrix (n x n) or implicit rotor params
- rotor_planes: list of plane index pairs
- rotor_thetas: list/vec of angles
- lior_state: alpha, integral, dtau (all tensors on GPU)
- global_step, epoch: training counters
- output_dir: checkpoint/log path
- metrics_logger: logging helper

Section TOC:
1) Entry and invariants (CUDA only, no autograd)
2) Config and menu switches
3) Geometry precompute (g0, g0_inv, K)
4) Curvature and collapse (R4 -> R_sc, operator form)
5) Frame and metric construction (derived, learned, rotor)
6) Retrieval cost and attention weights
7) Two-phase unroll (free and nudged)
8) Manual updates (contrastive stats, rotor, low-rank)
9) Metrics and logging
10) Checkpointing and resume
11) Validation and evaluation
12) Integration and entrypoints

Section headers and formulas index:
1) Entry and invariants
   - No formulas; constraints only.
2) Config and menu switches
   - No formulas; configuration only.
3) Geometry precompute
   - g0_inv = inverse(g0)
   - K_{mu nu rho sigma} = (1/n^2) * g0_inv^{mu rho} * g0_inv^{nu sigma}
4) Curvature and collapse
   - R_sc(x) = sqrt(|(1/n^2) * g^{mu rho} * g^{nu sigma} * R_{mu nu rho sigma}(x)| + eps)
   - Optional operator map (n=4):
     R_op[a,b] = R_{mu_a nu_a rho_b sigma_b}
5) Frame and metric construction
   - Derived frame: C = E[z z^T], Q = eigvecs(C)
   - Rotor: Q = product_k G(i_k, j_k, theta_k)
   - Low-rank metric: g(v,v) = Omega^2 * v^T g0 v + (U^T v)^T D (U^T v)
6) Retrieval cost and attention weights
   - Quadratic form: g(v,v) = v^T g v
   - Cost: cost = R_sc * sqrt(|g(v,v)| + eps)
   - Weights: w_i = softmax(-beta * cost_i)
7) Two-phase unroll
   - LIoR increment: dLIoR = R_sc * sqrt(|g(v,v)| + eps) * dtau
8) Manual updates
   - Rotor FD update: theta_k <- theta_k - eta * (J_plus - J_minus) / (2 * eps)
9) Metrics and logging
   - No formulas; stats aggregation only.
10) Checkpointing and resume
   - No formulas; serialization only.
11) Validation and evaluation
   - No formulas; reuse training loss definitions.
12) Integration and entrypoints
   - No formulas; wiring only.

SECTION 1) Entry and invariants (CUDA only, no autograd)
Formulas:
- None.
Expected bugs / pitfalls:
- Running on CPU or mixed CPU/GPU tensors causes silent slowdowns or crashes.
- Accidentally enabling autograd or creating optimizer state breaks the "no autograd" rule.
- Using numpy in the hot path forces CPU sync.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: main.py, training/__init__.py (if exported).
Calls into (current / intended):
- torch, torch.cuda for device checks.
Plain English:
- This section enforces non-negotiable runtime rules: CUDA only and no autograd.
- It protects performance and keeps the training loop deterministic and lightweight.
Subheaders:
- 1.1 CUDA hard requirement
- 1.2 Disable autograd and grads
- 1.3 Device/dtype guards

SECTION 2) Config and menu switches
Formulas:
- None.
Expected bugs / pitfalls:
- Invalid mode combinations (e.g., R_source="curvature" with no curvature provider).
- Missing defaults for frame_mode, metric_mode, rotor_mode.
- Configuration drift between YAML and runtime config dict.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: main.py config builder, configs/*.yaml.
Calls into (current / intended):
- configs/*.yaml, argparse or YAML loader (if used).
Plain English:
- This section defines the mode menu and safety clamps in one place.
- It prevents ad hoc branching and keeps the trainer behavior explicit.
Subheaders:
- 2.1 Frame and metric mode options
- 2.2 R_source and curvature options
- 2.3 Rotor and low-rank parameters
- 2.4 Safety clamps and numeric epsilons

SECTION 3) Geometry precompute (g0, g0_inv, K)
Formulas:
- g0_inv = inverse(g0)
- K_{mu nu rho sigma} = (1/n^2) * g0_inv^{mu rho} * g0_inv^{nu sigma}
Expected bugs / pitfalls:
- Non-invertible g0 leads to NaNs in g0_inv and K.
- Building K on CPU by mistake (very slow and causes device mismatch).
- g0 not matching model dimension (n mismatch).
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: trainer2 initialization, geometry core.
Calls into (current / intended):
- torch.linalg.inv, torch.eye, torch.zeros.
Plain English:
- Precompute constant geometry tensors once on GPU to avoid repeated work.
- K is the key contraction kernel used to collapse rank-4 curvature.
Subheaders:
- 3.1 Base metric definition (g0)
- 3.2 Inverse metric (g0_inv)
- 3.3 Contraction kernel (K)

SECTION 4) Curvature and collapse (R4 -> R_sc, operator form)
Formulas:
- R_sc(x) = sqrt(|(1/n^2) * g^{mu rho} * g^{nu sigma} * R_{mu nu rho sigma}(x)| + eps)
- Optional operator map (n=4):
  R_op[a,b] = R_{mu_a nu_a rho_b sigma_b}
Expected bugs / pitfalls:
- R_sc negative inside sqrt (use abs and eps).
- Allocating dense R4 in the hot path (blows memory).
- Using einsum in hot path when a precomputed K would suffice.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: retrieval cost, LIoR accumulator.
Calls into (current / intended):
- geometry precompute (K), field state provider.
Plain English:
- This section defines how curvature becomes a scalar for cost weighting.
- It keeps the physics (curvature) without creating large tensors per step.
Subheaders:
- 4.1 Constitutive R_source (direct R4 or R_sc)
- 4.2 Curvature R_source (derived from metric)
- 4.3 Debug-only operator form (bivector)

SECTION 5) Frame and metric construction (derived, learned, rotor)
Formulas:
- Derived frame: C = E[z z^T], Q = eigvecs(C)
- Rotor: Q = product_k G(i_k, j_k, theta_k)
- Low-rank metric: g(v,v) = Omega^2 * v^T g0 v + (U^T v)^T D (U^T v)
Expected bugs / pitfalls:
- Eigenvector sign flips causing jitter (fix sign deterministically).
- Rotor angles drifting without wrapping to [-pi, pi].
- U or D unconstrained causing negative or exploding costs.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: retrieval cost and LIoR step.
Calls into (current / intended):
- torch.linalg.eigh, geometry precompute.
Plain English:
- This section builds the frame and metric used for distances.
- It supports derived frames, learned low-rank anisotropy, and rotor-only rotations.
Subheaders:
- 5.1 Derived frame from stats
- 5.2 Learned low-rank anisotropy
- 5.3 Rotor-only frame

SECTION 6) Retrieval cost and attention weights
Formulas:
- Quadratic form: g(v,v) = v^T g v
- Cost: cost = R_sc * sqrt(|g(v,v)| + eps)
- Weights: w_i = softmax(-beta * cost_i)
Expected bugs / pitfalls:
- Wrong application of Q vs Q^T (frame mismatch).
- Negative or zero costs leading to invalid sqrt.
- Softmax overflow if beta is too large.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: model attention layers, memory retrieval.
Calls into (current / intended):
- Frame/metric construction, curvature collapse.
Plain English:
- This is the hot path that computes distances and attention weights.
- It must be GPU-only, stable, and consistent with the chosen geometry.
Subheaders:
- 6.1 Displacement build (v = x_i - x_q)
- 6.2 Frame rotation (apply Q or rotor)
- 6.3 Cost and softmax

SECTION 7) Two-phase unroll (free and nudged)
Formulas:
- dLIoR = R_sc * sqrt(|g(v,v)| + eps) * dtau
Expected bugs / pitfalls:
- Forgetting to snapshot and restore state between free and nudged runs.
- Mixing stats across phases, invalidating contrastive updates.
- Accidental CPU sync when aggregating stats.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: training loop step function.
Calls into (current / intended):
- Retrieval cost, LIoR accumulator, field evolution.
Plain English:
- This section runs the model twice to measure the effect of nudging.
- The difference between phases drives manual updates without autograd.
Subheaders:
- 7.1 Snapshot and restore
- 7.2 Free phase
- 7.3 Nudged phase
- 7.4 Stats aggregation

SECTION 8) Manual updates (contrastive stats, rotor, low-rank)
Formulas:
- Rotor FD update: theta_k <- theta_k - eta * (J_plus - J_minus) / (2 * eps)
Expected bugs / pitfalls:
- Updating too many parameters per step (destabilizes learning).
- Forgetting to clamp positive parameters (lambda, D, Omega).
- Using large eps for finite differences (noisy gradients).
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: end of each training step or window.
Calls into (current / intended):
- Two-phase stats, frame/metric state.
Plain English:
- This section applies small, local parameter updates from contrastive signals.
- It keeps parameter counts tiny and avoids autograd.
Subheaders:
- 8.1 Contrastive stats update
- 8.2 Rotor angle update
- 8.3 Low-rank update
- 8.4 Clamp and stabilize

SECTION 9) Metrics and logging
Formulas:
- None.
Expected bugs / pitfalls:
- Frequent .item() calls causing GPU sync overhead.
- Logging inside the hot path (slows training).
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: training loop, validation loop.
Calls into (current / intended):
- training/metrics.py, logging utilities.
Plain English:
- This section collects and logs training statistics at safe intervals.
- It avoids syncing every step to keep the GPU busy.
Subheaders:
- 9.1 Train metrics
- 9.2 Validation metrics
- 9.3 Timing diagnostics

SECTION 10) Checkpointing and resume
Formulas:
- None.
Expected bugs / pitfalls:
- Serializing large tensors too frequently (I/O stalls).
- Missing new geometry params in checkpoints (incomplete restore).
- Saving CPU tensors by mistake (device mismatch on load).
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: training loop on interval, signal handler.
Calls into (current / intended):
- training/checkpoint_utils.py, torch.save/load.
Plain English:
- This section saves and restores minimal trainer state for recovery.
- It is optional and should default to off if strict GPU-only is required.
Subheaders:
- 10.1 Save state
- 10.2 Load state
- 10.3 Versioning and schema

SECTION 11) Validation and evaluation
Formulas:
- None.
Expected bugs / pitfalls:
- Running validation on CPU by accident.
- Using training-time nudges during validation.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: training loop, CLI evaluation flow.
Calls into (current / intended):
- model forward, retrieval cost, metrics logger.
Plain English:
- This section measures model quality without training updates.
- It reuses the same geometry but runs in eval mode.
Subheaders:
- 11.1 Eval loop
- 11.2 Metric aggregation
- 11.3 Early stop criteria

SECTION 12) Integration and entrypoints
Formulas:
- None.
Expected bugs / pitfalls:
- Forgetting to export trainer2 in training/__init__.py when needed.
- CLI menu wiring into the wrong trainer class.
Callers (current / intended):
- Current callers: none (new file).
- Intended callers: main.py, training/__init__.py, tests.
Calls into (current / intended):
- training/datasets.py, inference/geometric_attention.py, core/tensor_field.py.
Plain English:
- This section documents how trainer2 plugs into the existing codebase.
- It keeps the wiring clear so integration is low-risk.
Subheaders:
- 12.1 main.py menu wiring
- 12.2 training/__init__.py export
- 12.3 Tests and smoke runs
"""
