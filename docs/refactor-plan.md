# Refactor Plan: Capture-Friendly Trainer2 & Geometric Attention

## Goals
- Keep live metrics without breaking cudagraph/inductor capture.
- Enforce memory query/update separation: no model calls inside memory, no self-insertion, single commit point after free/nudge.
- Ensure masked pooling and fixed shapes for retrieval (capture-safe, no padding leakage).
- Make logging/device sync discipline explicit.
- Clarify geometry modes (diag rot / block rot / low-rank) and their cost/benefit.
- Prepare for `torch.compile` and/or explicit `torch.cuda.CUDAGraph`.

## Decisions (locked)
- **No model calls in memory**: memory exposes `query(batch_size)` and `update(q_coord, q_state)`. Model forward happens exactly once per phase.
- **No updates between free and nudge**: commit once per window (field + memory). Free/nudged runs are read-only.
- **Masked pooling required**: always pool q_coord/q_state with attention_mask; treat mask as all-ones if none provided.
- **Static shapes for capture**: fixed `B`, `T`, `coord_dim_n`, `capacity`; memory returns capacity-sized banks + `valid_mask`.
- **Logging**: accumulate metrics on GPU; host-print only at coarse intervals via async copy; no `cuda.synchronize()` in hot path.

## Why dynamic shapes still break capture
- **Variable-length batches**: dataloaders emit variable `T` unless padded/truncated upstream. Fix: pad/truncate before trainer2.
- **Memory filled dimension**: slicing `mem[:filled]` changes shapes step-to-step. Fix: always return `[B, capacity, ...]` plus `valid_mask`.
- **Conditional branches on masks/modalities**: differing graph per call under `torch.compile`. Fix: prebuild default masks and unify code paths with fixed signatures.

## TODO (implementation checklist)
- [ ] **Logging refactor (trainer2)**
  - [ ] Remove hot-path `torch.cuda.synchronize()` and `.item()` in `maybe_log_metrics`.
  - [x] ~~Add device-side metric ring buffer (e.g., store `lior_mean`, `R_mean`, `spd_mean` for last N windows).~~
  - [ ] Add async host-flush every N windows (configurable), using side stream + `non_blocking=True` copy.
  - [ ] Keep debug mode that can force syncs, but default path must be sync-free.

- [ ] **Memory API enforcement**
  - [x] ~~Ensure trainer uses `memory.query(batch_size)` before retrieval; uses `memory.update(q_coord, q_state)` only after commit.~~
  - [ ] Remove/avoid any model invocation inside memory.
  - [x] ~~Guarantee `static_shapes=True` path returns capacity-sized `mem_coord/mem_state` + `valid_mask` (mask-based invalidation, no shape change).~~

- [ ] **Masked pooling**
  - [x] ~~In `build_retrieval_batch`, always pool `q_coord/q_state` with attention_mask; if mask missing, use all-ones of fixed shape.~~
  - [ ] Ensure pooled tensors keep fixed shape/dtype/device (no Python branching on mask rank).

- [ ] **Free/Nudge isolation**
  - [ ] Restructure so field/memory are read-only during free/nudge; no snapshot clone of `field.T` in hot path.
  - [ ] Apply updates once at commit point; if rollback needed, snapshot only lightweight metadata (ptr/filled), not full tensors.

- [ ] **Static shape guarantees**
  - [ ] Upstream: pad/truncate sequences to fixed `T` before trainer2.
  - [x] ~~Always pass a fixed-shaped attention mask (even if all ones).~~
  - [x] ~~Avoid any slicing based on `filled`; use `valid_mask` instead.~~

- [ ] **Geometry modes (doc + config sanity)**
  - [ ] Document costs: diag (O(D)), block rot (O(D * layers)), low-rank (O(D*r)); warn dense (O(D^2)) is capture-hostile.
  - [ ] Keep retrieval metric diagonal in rotated/structured frame for capture-friendliness.

- [ ] **Compile/CUDAGraph path**
  - [ ] After the above, add optional `torch.compile(mode="reduce-overhead")` around the step function.
  - [ ] If graph breaks remain, add explicit `torch.cuda.CUDAGraph` capture with static input pools.
  - [ ] Warmup K steps to stabilize allocator before capture.

## Geometry / rotor tradeoffs
| Mode             | Cost          | Expressivity | Capture risk | So what |
| ---------------- | ------------- | ------------ | ------------ | ------- |
| Diag rot         | O(D)          | Low          | Minimal      | Cheap/stable for large-D memory and sequential physics. |
| Block rot        | O(D · blocks) | Medium       | Low          | Adds local coupling without D² blowup; matches localized coherence. |#This one should be default, 2x2 or 4x4 blocks.
| Low-rank rot     | O(D · r)      | Med–High     | Low–Medium   | Captures a few latent correlations cheaply; good if only a few dominant modes. |
| Full dense rot   | O(D²)         | High         | High         | Capture-hostile, encourages spurious coupling/hallucinations. |

## Similarity options (retrieval) #What about L2, Energy, spinor/tensor, etc. ?
| Type                        | Formula (cost)                  | Cost  | Use-case / So what |
| --------------------------- | ------------------------------- | ----- | ------------------ |
| Diagonal quad form          | Σ g_i (Δx_i)²                   | O(D)  | Default: fast, stable, huge K. |
| Block-diagonal quad form    | Small 2×2/4×4 blocks            | O(D·b)| Local cross-talk; structured physics. |
| Low-rank Mahalanobis        | ‖Δx‖² + ‖Δx U‖² (U∈R^{D×r})      | O(D·r)| Factorized correlation; r≪D. |
| Elementwise temporal        | per-channel over window W       | O(D·W)| Sequential memory; avoids cross-channel contamination. |
| RBF/softmax(-β * cost)      | exp(-β * cost)                  | O(KD) | Your retrieval weighting; keeps probabilities physical. |
| Cosine (diagnostic)         | (x·y)/(‖x‖‖y‖)                  | O(KD) | Scale-free diagnostics; not for training weights. | #This should be the curved version. 

## Implementation blueprint
- **Logging**: GPU metric buffer + async host flush every N windows; no `.item()`/sync in hot path; debug mode may sync sparsely.
- **Memory**: `query` returns fixed `[B,capacity,…]` + `valid_mask`; `update` after commit only; no model calls in memory.
- **Masked pooling**: always use attention_mask (or all-ones) to pool q_coord/q_state; fixed shapes, no rank-branching.
- **Free/Nudge**: run both on read-only state; commit once; if rollback, snapshot only metadata (ptr/filled), not tensors.
- **Static shapes**: pad/truncate upstream; fixed masks; `static_shapes=True` for SDM; avoid slicing by `filled`.
- **Compile/CUDAGraph**: after above, try `torch.compile`; if still breaking, wrap step in explicit `torch.cuda.CUDAGraph` with warmup.

## Metrics to verify (cheap, capture-safe) #Don't delete the existing metrics, just move them out of the hot path. Make interval 1/4 of steps in epoch.
| Metric       | Expectation              | Interval |
| ------------ | ------------------------ | -------- |
| R_mean       | Stable or gentle decay   | 100–500  |
| lior_mean    | Smooth increase/stability| 100–500  |
| spd_mean     | Small bounded oscillation| 100–500  |
| entropy (opt)| Slow decay (collapse)    | debug    |

## Recommended stack
| Layer        | Choice                                   | Reason |
| ------------ | ---------------------------------------- | ------ |
| Geometry     | Diagonal or block rot                    | Capture-safe, interpretable. |
| Memory       | Ring buffer (static shapes + valid_mask) | Stable capacity, fixed layout. |
| Retrieval    | RBF of quadratic cost                    | Physical, consistent. |
| Attention    | Elementwise/structured (not dense)       | Limits hallucinations. |
| Runtime      | torch.compile → CUDAGraph                | Debuggable, then max perf. |
| Logging      | GPU accumulation + async host flush      | Zero sync in hot path. |
| Free/Nudge   | Read-only until commit                   | True two-phase isolation. |

## Notes on live metrics (cheap & capture-friendly)
- Accumulate metrics on GPU: `ema = ema + (x - ema) * alpha` or store per-window into a small CUDA tensor buffer.
- Every N windows (e.g., 500/1000), copy a small metrics tensor to CPU using a side stream and `non_blocking=True`; synchronize that stream only for printing.
- Do not call `.item()`/`.cpu()` in the captured step; only in the periodic flush.
- Checkpointing should stay outside capture or at very low frequency (will sync).

## Risk log / watchpoints
- Any data-dependent Python branching in `build_retrieval_batch` (mask rank, presence of memory) will break capture; standardize shapes and always-return branches.
- Growing `filled` dimension changes shape unless masked; use `valid_mask` to avoid recompile.
- Field snapshots/clones in hot path kill bandwidth and capture; prefer deferred commit.

## Next actions (suggested order)
1) Patch logging hot path to remove syncs/items; add GPU metric buffer + periodic async flush.
2) Enforce memory query/update split in trainer2; fix masked pooling and static mask shape.
3) Remove field.clone snapshot in free/nudge; adopt commit-only updates with lightweight metadata snapshot if needed.
4) Standardize shapes (pad/truncate, static_masks, static_shapes=True for SDM).
5) Try `torch.compile`; if unstable, add explicit CUDAGraph capture around the step.

---

## 7) Telemetry: widen buffered logging (optional, inference-only)

If you want to print `wedge_norm` and `contract_mean` only when present:
