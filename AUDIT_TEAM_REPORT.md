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

## Six-Agent Specialist Review Plan (How to Proceed)

1. **Coordinator Agent**: Owns scope, creates review queue by risk order (`kernels` → `models` → `training`/`inference` → `core` → CLI/docs), enforces deadlines, and resolves blockers.
2. **Math Agent**: Verifies derivations, tensor-shape invariants, stability bounds, and numerical assumptions in `kernels/`, `models/`, and `utils/` math paths.
3. **Physics Agent**: Checks physical consistency (causality, metric signature, conservation behavior, manifold assumptions) across `kernels/`, `models/`, and training dynamics.
4. **Code Reviewer Agent**: Audits correctness, edge-case handling, error paths, and test coverage gaps; proposes minimal diffs for high-risk defects only.
5. **Scribe Agent**: Produces a single decision log with findings, severity, evidence (file + line), and recommended next actions; keeps cross-agent terminology consistent.
6. **Morale Agent**: Maintains team throughput and focus by flagging overload early, balancing workload, and keeping review cadence sustainable.

Execution cadence: each specialist submits findings in parallel per module batch, then coordinator runs a short synthesis pass; proceed only when math + physics + code reviewer agree on blocking risks and the scribe publishes the consolidated action list.

## Detokenize / Output Agent Team

A dedicated six-agent team (`utils/detokenize_output_agents.py`) scans the
detokenizing and output pipeline end-to-end and produces a phased plan to
finalise and make the pipeline fully operational.

### Agent Roles

1. **DetokenizeScannerAgent**: Scans every `.py` file in the repository and
   classifies files by role (`tokenizer`, `lm_head`, `generation`, `entropy`,
   `test`).  Broadcasts the discovered locations to the three specialist agents.

2. **TokenizerHealthAgent**: Validates `CognitiveTokenizer` (source-text + runtime):
   - Special-token map contains all required entries.
   - `decode()` implemented with HF backend or fallback `inverse_vocab`.
   - `encode()` supports `max_length` truncation.
   - `eos_token_id` property exposes `<|endoftext|>`.
   - Encode → decode roundtrip is lossless; batch encoding matches individual encoding.

3. **LMHeadAuditAgent**: Validates `LanguageModelHead` (source-text + runtime):
   - `LayerNorm` applied before the output projection.
   - `output_projection` is `nn.Linear(d_model, vocab_size)`.
   - Weight tying is optional and guarded by `tie_weights` flag.
   - Forward pass produces correct shape `(batch, seq_len, vocab_size)` with finite logits.

4. **GenerationPipelineAgent**: Validates `InferenceEngine.generate()` (source-text + runtime):
   - EOS termination breaks the loop before `max_tokens` is exhausted.
   - `input_ids` is clipped to `max_seq_len` on each step.
   - `field.evolve_step()` is called every iteration.
   - `tokenizer.decode(generated_ids)` is called to produce the final text.
   - Entropy gating and selector probabilities are applied before sampling.

5. **OperationalizationAgent**: Synthesises all findings into a four-phase plan:
   - Phase 1 – Stabilise (resolve critical blockers).
   - Phase 2 – Validate (address major findings + add tests).
   - Phase 3 – Harden (performance, robustness, monitoring).
   - Phase 4 – Operationalise (deployment readiness, CI integration).

6. **OutputTeamCoordinator**: Orchestrates the full team, collects all findings,
   and returns a `DetokenizeOutputReport` containing locations, findings, an
   action log, and the operationalization plan.

### Running the Team

```python
from utils.detokenize_output_agents import OutputTeamCoordinator

coordinator = OutputTeamCoordinator(repo_root="/path/to/repo")
report = coordinator.run(run_numerical=True)

for line in report.action_log:
    print(line)

for line in report.operationalization_plan:
    print(line)
```

Tests: `tests/test_detokenize_output_agents.py` (36 checks).
