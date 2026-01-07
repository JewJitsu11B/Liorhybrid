# Contributing

This repository implements a CUDA-only, no-autograd dynamical training system. Contributions must preserve state integrity, window semantics, and geometry-native selection behavior.

## Non-negotiables

### CUDA-only + no autograd
- CUDA is required for trainer paths; no CPU fallback in hot paths.
- Autograd must remain disabled; do not introduce `.backward()`, `.grad`, or optimizer state.

### Window semantics
- A “window” is the atomic unit of evolution and the only valid commit boundary for long-lived state mutations.
- Free/nudged comparisons must start from identical state (snapshot/restore must be exact).

## Conservation semantics

Selection, collapse, and memory accumulation must redistribute existing mass/energy;
they must not create it implicitly.

Forbidden:
- hidden renormalization inside kernels
- implicit amplification via softmax misuse
- double-counting across windows

If magnitude changes, it must be attributable to an explicit decay, gate, or injection.

## Causal exclusion guarantee

Delayed accumulation must read only from state written in strictly prior windows.
Rolling or circular buffers are permitted, but the slot written during the active
window must never be visible to delayed reads.

## State integrity rules

### Inference must be state-safe
Inference must not mutate long-lived state (`model`, `field`, `memory`, `rotor_state`) unless explicitly opted-in and clearly labeled as mutating.

### Explicit commit points
Memory updates may only occur at explicit commit points (window boundaries). Retrieval (`memory.query`) must be observational and must not mutate state.

## SPSA policy
SPSA is diagnostic-only by default:
- must be explicitly opt-in
- must log when triggered
- probe effects are assumed to persist unless state-safe bracketing is used

## Future multi-GPU plans (priority order)

The preferred scaling strategy is explicitly constrained by causal ordering, conservation, and the delayed-read exclusion rule.

### Rank 1 (Best): Concept / Basis sharding
Parallelize over concepts/keys/φ-vectors (disjoint subsets per GPU).

Parallelizes:
- inner products ⟨φᵢ | ψ⟩
- Born weights |⟨φᵢ | ψ⟩|²
- softmax logits prior to global reduction

Why preferred:
- zero causal coupling
- no history dependence
- no window ordering concerns
- no shared state mutation

Practical benefit:
- near-linear speedup with GPUs
- minimal synchronization (small reduction)
- scales across many GPUs cleanly

Cost:
- one all-reduce (or gather) for logits / partial softmax statistics

Risk:
- low, provided the reduction is correct (numerically and semantically)

### Rank 2: Agent-level parallelism (multiple fields / belief states)
Parallelize over independent Ψ systems (separate fields, memories, windows).

Why preferred:
- dynamics are local per system
- conservation and LIoR are per-field unless explicitly coupled
- no gradients → minimal synchronization pressure

Practical benefit:
- linear scaling with number of agents
- minimal code coupling

Cost:
- GPU memory per agent

Risk:
- only if inter-agent coupling is added without syncing invariants

### Rank 3: Kernel / history evaluation parallelism
Parallelize over delayed-kernel components and history slices (Δt terms).

Why useful:
- history reads are read-only
- each Δt contribution independent
- preserves causal structure

Cost:
- more kernel launches; careful buffer indexing

Risk:
- stream/order hazards that violate read-before-write or causal exclusion

### Rank 4: Spatial / field partitioning
Partition the tensor field/manifold across GPUs.

Why last:
- delayed terms are not spatially local
- fractional operators may require cross-boundary communication

Cost:
- halo exchanges and boundary synchronization

Risk:
- easy to violate conservation or introduce subtle drift

## Running Tests

Always run Python with the `-Xgil=0` flag to suppress GIL warnings from tokenizers:

```bash
python -Xgil=0 main.py
python -Xgil=0 -m pytest tests/
```

This prevents the RuntimeWarning about the global interpreter lock being enabled for tokenizers.

---

## Option 6 Implementation (REQUIRED READING)

Before modifying any files in `inference/`, you MUST review:

```
~/.claude/plans/option6-address-probing.md
```

This plan documents the **Address-Space Probing** architecture:

- **Core Principle**: All interaction happens in Address space. Embeddings are payloads, not geometry.
- **NO POOLING**: Pooling destroys information. The qualitative costs are inexcusable.
- **NO DENSE MATMUL**: K is the Address structure, not a dense tensor.
- **Attention = Probing**: Loop over neighbors (64 total), not Q @ K.T.
- **O(N × 64 × d')** complexity, not O(N²).

Key files affected:
- `inference/address.py` - Address structure (D=7202 floats)
- `inference/geometric_attention.py` - Must use `probe_address()`, not matmul
- `inference/geometric_stack.py` - Must use `AddressBuilder`, not pooling

Any changes that introduce pooling, softmax collapse, or dense K/V matmul will be rejected.

## PR checklist
- [ ] No autograd or optimizer state introduced
- [ ] CUDA-only constraints preserved
- [ ] Window boundary semantics preserved (commit only at window boundary)
- [ ] Conservation semantics preserved (no implicit amplification/double-counting)
- [ ] Causal exclusion guarantee preserved (no same-window reads from delayed buffers)
- [ ] Inference state-safety preserved
- [ ] SPSA remains opt-in and clearly labeled
- [ ] Multi-GPU changes follow the documented scaling priorities and do not violate causal exclusion