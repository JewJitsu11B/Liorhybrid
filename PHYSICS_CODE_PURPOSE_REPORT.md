# Physics-First Code Audit and Architectural Rationale

**Date:** 2026-01-13  
**Scope:** Repository-wide file audit focused on intent, physics choices, and the architectural replacements required to keep the physics consistent.

## What the system is trying to accomplish
- **Evolve a cognitive tensor field** (`core/tensor_field.py`) whose dynamics obey a Hamiltonian + Bayesian update with fractional memory. The field is treated as a rank-2 complex tensor that must conserve probability and energy while ingesting evidence.
- **Embed ML training inside the physics** (`training/lior_trainer.py`): geodesic cost and entropy-based updates make the learning trajectory follow the field geometry rather than an arbitrary optimizer.
- **Perform inference with geometric algebra** (`inference/geometric_mamba.py`, `inference/geometric_attention.py`, `inference/geometric_products.py`): downstream models read the field using wedge/spinor/tensor products and octonionic state-space updates so that causal structure and orientation are preserved.
- **Keep all components auditable** (`utils/pipeline_audit.py`, `main.py` preflight) by asserting device/parameter invariants before optimizers run.

## Physics rationale behind the key choices
- **Hamiltonian gradient, not direct energy injection** (`core/tensor_field.py`): fractional memory now modulates the Hamiltonian gradient instead of adding a force term, preventing unphysical energy growth and preserving Bayesian normalization.
- **Bayesian collapse with prior memory** (`kernels/bayesian.py`, `kernels/fractional_memory.py`): evidence enters via a Λ_QR term while history contributes only as a scalar weight, matching \(P(\psi|D) \propto P(D|\psi)P(\psi_\text{prior})\).
- **Geodesic action as the learning signal** (`training/lior_trainer.py`): trajectories are penalized when they deviate from the field-induced metric \(g_{\mu\nu}=T^\top T\); this keeps optimization aligned with the physical manifold and avoids exploding gradients that standard CE-only objectives would cause.
- **Geometric algebra for representation** (`models/biquaternion.py`, `models/complex_metric.py`, `models/manifold.py`): using biquaternions and complex metrics preserves orientation, spin, and phase information that ordinary real-valued layers would discard.
- **Three-mode LIoR memory kernel** (`models/lior_kernel.py`): exponential + power-law + oscillatory components approximate long-range memory with a finite recurrence while enforcing stability (\(\rho \in (0,1)\)).

## Where modern ML architecture was replaced to satisfy the physics
- **State-space core:** Standard linear Mamba transition \(h_t = A h_{t-1} + B x_t\) is replaced by **CI8 Trinor evolution** (`inference/geometric_mamba.py`). Octonionic multiplication and learned rotations keep causal arrows and phase correlations that matrix multiplication would destroy.
- **Attention block:** Dot-product attention is replaced by **geometric attention** (`inference/geometric_attention.py`) using wedge/tensor/spinor scores (`inference/geometric_products.py`). This maintains antisymmetry, bivector structure, and phase alignment required by the field.
- **Training objective:** Plain cross-entropy is augmented with **geodesic + entropy terms** (`training/lior_trainer.py`), because pure CE ignores the field geometry and breaks norm/energy conservation during backprop.
- **Phase representation:** Instead of complex dtype softmax tricks, the **exponential generator head** (`inference/geometric_attention.ExponentialPhaseExtractor`) maps vectors into triality-aligned biquaternions, avoiding NaNs and preserving Lie-group structure.

## File-by-file audit map
- **Field evolution:** `core/tensor_field.py`, `core/config.py`
- **Physics operators:** `kernels/hamiltonian.py`, `kernels/bayesian.py`, `kernels/fractional_memory.py`, `kernels/gradients.py`
- **Memory kernel:** `models/lior_kernel.py`
- **Geometric algebra primitives:** `models/biquaternion.py`, `models/manifold.py`, `models/complex_metric.py`
- **Inference stack:** `inference/geometric_mamba.py`, `inference/geometric_attention.py`, `inference/geometric_products.py`, `inference/geometric_stack.py`
- **Training loop & metrics:** `training/lior_trainer.py`, `training/metrics.py`, `training/cost_terms.py`
- **Audit & safety gates:** `utils/pipeline_audit.py`, `main.py` (preflight checks)

## How the pieces interact (physics-first flow)
1. **Field evolution:** `CognitiveTensorField.evolve_step` updates \(T_{ij}\) with Hamiltonian + Bayesian + memory weighting, storing history for future steps.
2. **Metric construction:** Averaged field produces \(g_{\mu\nu}\) and density matrices (`training/lior_trainer.py`, `models/complex_metric.py`), defining the space the learner must follow.
3. **Inference readout:** Geometric Mamba encodes sequences in CI8 space; geometric attention scores queries against field-derived key/value tensors using wedge/spinor/tensor products.
4. **Learning signal:** Geodesic and entropy terms penalize paths that deviate from the field manifold; gradients stay bounded because the metric is detached and dimension-matched to the field.
5. **Audit hooks:** Each critical entry point records a one-time audit (`utils/pipeline_audit.audit_file_once`), ensuring reproducibility and device consistency.

## Summary
The repository rewires standard ML components to obey a Hamiltonian/Bayesian cognitive field. Octonionic state evolution, geometric attention, and geodesic-cost training are substitutions made specifically to preserve energy, probability conservation, causal orientation, and long-range memory that ordinary Transformer/Mamba blocks would violate.
