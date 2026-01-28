## Multi-role Audit – Key Runtime Files

Roles covered per file: Code Reviewer (CR), Physics Expert (PE), Crazy Idea (CI). Ratings: 1 (low risk/ready) – 5 (high risk/needs work).

### training/trainer.py — Rating: 3
- **CR:** Solid invariants for attached modules and device moves; mixed precision/compile guarded. Signal handler does side-effect work during interrupt—acceptable but consider async-safe logging. Checkpoint path handling fine. Watch for torch.compile fallback print spam in long runs.
- **PE:** No physics-heavy math here; relies on external field/model. Uses cleanup thread when CUDA available; safe for tensor field state. Ensure field tensor `T` stays normalized upstream.
- **CI:** Add live curvature diagnostics from field into metrics logger to auto-tune grad clip per batch.

### inference/inference.py — Rating: 3
- **CR:** Robust checkpoint introspection; infers shapes from state_dict. Potential risk if unexpected keys (defensive). Consider validating `spatial_size` tuple length before use. No lazy imports for adapters—OK for CLIs.
- **PE:** Field reconstruction assumes checkpoint tensor consistency; no entropy gating implemented yet. Retrieval roadmap noted but not wired.
- **CI:** Plug a lightweight SDM cache that learns on-the-fly with entropy gates, exposed as a flag.

### kernels/ (gradients.py, hamiltonian.py, metric_context.py, fractional_memory.py, tetrad.py, bayesian.py) — Rating: 4
- **CR:** Math-heavy; multiple chained tensor ops with limited shape assertions. Suggest adding quick asserts for device/shape alignment near public entry points.
- **PE:** Hamiltonian/tetrad interactions sensitive to metric signature; ensure consistency with manifold config. Fractional memory kernel needs stability checks for high-order terms.
- **CI:** Explore adaptive metric that anneals between Minkowski-like and learned Riemannian curvature based on batch entropy.

### models/ (biquaternion.py, manifold.py, lior_kernel.py, causal_field.py, rank_reduction.py, complex_metric.py, language_head.py, activations.py) — Rating: 4
- **CR:** Rich component set; some modules likely bypassed in configs—could drift untested. Encourage unit guards for optional paths. Pay attention to parameter init alignment with trainer expectations (embedding/lm_head).
- **PE:** Manifold/complex metric pieces must align with kernel assumptions; check biquaternion ops for unit norm and associativity in backprop. Rank reduction should preserve causal field constraints.
- **CI:** Try gating manifold curvature by token-level uncertainty, with a low-rank adapter that modulates activation nonlinearity.

### core/ (config.py, tensor_field.py) — Rating: 3
- **CR:** Central tensor field definitions; ensure config validation covers device and spatial sizes. Thread-safety seems implicit—document if shared across loaders.
- **PE:** Tensor field evolution rules must maintain causality; verify any diffusion/decay terms respect timestep ordering.
- **CI:** Introduce reversible updates (symplectic-like) to conserve information volume, toggleable via config.

### main.py — Rating: 2
- **CR:** CLI/bootstrap logic straightforward; preflight audit hook already present. Ensure sys.path tweak is minimal and safe. Parameter summaries helpful.
- **PE:** Minimal physics impact; defers to downstream modules.
- **CI:** Add a “physics sanity check” mode that runs a tiny manifold consistency probe before full training.

---
Summary: Highest attention areas are **kernels** and **models** (ratings 4) due to math sensitivity and optional paths. trainer/inference/core are moderate (3) and main.py is low (2).
