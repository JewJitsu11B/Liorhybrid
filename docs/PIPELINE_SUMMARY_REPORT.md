# Liorhybrid Main Pipeline Summary (vs. JAMBA Hybrid)

## Scope
- Summarizes the primary pipeline files and their key sections.
- Highlights how the physics-first design offers advantages over JAMBA's MoE + Mamba hybrid.

## File-by-File Outline

### `main.py`
- **Interactive / CLI entry**: Menu-driven config and automation hooks.
- **Preflight checklist**: Device + parameter guards; audit hook (`audit_file_once`) for pipeline tracing.
- **Cost calculator**: Quick params/memory/compute estimates to size runs.
- **Why better than JAMBA**: Enforces physics invariants and device consistency before training starts; JAMBA relies on generic launcher scripts without physics-aware preflight.

### `core/tensor_field.py`
- **Field state and evolution**: Bayesian cognitive tensor field, time stepping, state dict IO.
- **Memory-aware ops**: Hooks for fractional/long-memory dynamics.
- **Why better than JAMBA**: True field dynamics with fractional memory; JAMBA SSM state is O(N) and empirical, not physics-grounded.

### `kernels/fractional_memory.py`
- **Fractional kernel weights**: Power-law memory and gradient modulation.
- **Stability controls**: Bounded exponents and safe parameterization.
- **Why better than JAMBA**: Constant-memory finite-pole recurrence (O(1)) vs JAMBA’s growing SSM state; preserves long-range correlations without sequence-length scaling.

### `training/trainer.py` (standard)
- **Data plumbing**: Tokenizer, loaders, batching.
- **Training loop**: Optimizer/loss wiring with physics-aware callbacks.
- **Why better than JAMBA**: Keeps geometric/physics hooks in loop; JAMBA training loop is MoE/attention centric without manifold constraints.

### `training/trainer2.py` (physics-complete path)
- **Entry & invariants**: CUDA-only, no autograd hot path.
- **Geometry config**: Frame/metric modes, rotor/low-rank params, safety clamps.
- **Curvature & collapse**: Constitutive/curvature sources without dense R4 in hot path.
- **Retrieval weights**: Geodesic costs → softmax with physics-aware displacements.
- **Two-phase unroll**: Free → nudged contrastive stats; manual updates without autograd.
- **Metrics/checkpointing**: CUDA-safe logging, resume hooks.
- **Why better than JAMBA**: Geometry-true retrieval and contrastive physics loop; JAMBA hybrid optimizes attention/SSM but lacks differential-geometric costs and free/nudged physics supervision.

### `inference/geometric_stack.py`
- **Geometric transformer inference**: O(N) dominant stack with causal field integration.
- **Memory kernel reuse**: Shares fractional kernel behavior for consistency with training.
- **Why better than JAMBA**: Unified train/infer geometry (same field + kernel) vs JAMBA’s train/infer split between attention and SSM blocks.

### `utils/pipeline_audit.py` + `pipeline_audit.md`
- **Runtime audit**: First-touch file logging, markdown event trail, unique file summary.
- **Why better than JAMBA**: Built-in provenance of pipeline participation; JAMBA lacks per-file audit for physics integrity.

## Cross-Cutting Advantages over JAMBA Hybrid
- **Physics-unified stack**: Field → geometry → retrieval → loss share one manifold; JAMBA blends MoE + Mamba without a governing physical metric.
- **Constant-memory long context**: Fractional kernel O(1) state vs JAMBA’s O(N) SSM expansion.
- **Geodesic costs instead of attention heuristics**: Retrieval weights derived from metric curvature, not token similarity alone.
- **Train/Infer symmetry**: Same kernels and geometry reused; JAMBA hybrids often diverge between training (attention+MoE) and inference (compressed SSM).
- **Auditability**: Pipeline audit trail ensures integrity across stages; JAMBA offers no comparable first-touch provenance.
