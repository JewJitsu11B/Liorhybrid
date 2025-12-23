# Trainer2 Outline Summary

File: `training/trainer2.py`  
Status: runnable (manual trainer with SDM + connection hooks required)

## Wiring
- Exported as a module via `training/__init__.py` under `trainer2`.
- Import path: `from Liorhybrid.training import trainer2`.

## Key Variables (top-level)
- config: dict of mode switches and numeric knobs
- device: torch.device (must be cuda)
- dtype: model/geometry dtype
- model: geometric transformer (or equivalent)
- field: CognitiveTensorField (or equivalent)
- train_loader, val_loader: DataLoaders
- frame_mode: "derived" | "learned_lowrank" | "rotor"
- metric_mode: "diag_rot"
- R_source: "constitutive" | "curvature"
- rotor_mode: "off" | "derived" | "stateful"
- g0, g0_inv, K: base metric + contraction kernel
- lambda_local, lambda_mem: diagonal stiffness
- U_mem, D_mem: low-rank memory correction
- Q, rotor_planes, rotor_thetas: frame/rotor state
- lior_state: alpha, integral, dtau (GPU tensors)

## Section Map (Purpose + Subheaders)
1) Entry and invariants
   - Purpose: enforce CUDA-only and no autograd.
   - Subheaders: 1.1 CUDA hard requirement, 1.2 Disable autograd and grads, 1.3 Device/dtype guards
2) Config and menu switches
   - Purpose: centralize mode selection and safety clamps.
   - Subheaders: 2.1 Frame/metric options, 2.2 R_source options, 2.3 Rotor/low-rank params, 2.4 Numeric epsilons
3) Geometry precompute
   - Purpose: build g0, g0_inv, K once on GPU.
   - Subheaders: 3.1 Base metric, 3.2 Inverse metric, 3.3 Contraction kernel
4) Curvature and collapse
   - Purpose: get R_sc without dense R4 in the hot path.
   - Subheaders: 4.1 Constitutive R_source, 4.2 Curvature R_source, 4.3 Debug operator form
5) Frame and metric construction
   - Purpose: build Q and metric forms for costs.
   - Subheaders: 5.1 Derived frame, 5.2 Low-rank anisotropy, 5.3 Rotor-only frame
6) Retrieval cost and attention weights
   - Purpose: compute distances and weights with the chosen geometry.
   - Subheaders: 6.1 Displacements, 6.2 Frame rotation, 6.3 Cost + softmax
7) Two-phase unroll
   - Purpose: free vs nudged phases for contrastive stats.
   - Subheaders: 7.1 Snapshot/restore, 7.2 Free phase, 7.3 Nudged phase, 7.4 Stats
8) Manual updates
   - Purpose: tiny parameter updates without autograd.
   - Subheaders: 8.1 Contrastive stats, 8.2 Rotor update, 8.3 Low-rank update, 8.4 Clamps
9) Metrics and logging
   - Purpose: log safely without sync stalls.
   - Subheaders: 9.1 Train metrics, 9.2 Validation metrics, 9.3 Timing
10) Checkpointing and resume
   - Purpose: optional state save/restore.
   - Subheaders: 10.1 Save state, 10.2 Load state, 10.3 Schema
11) Validation and evaluation
   - Purpose: eval-only loop with geometry intact.
   - Subheaders: 11.1 Eval loop, 11.2 Aggregation, 11.3 Early stop
12) Integration and entrypoints
   - Purpose: wiring notes and test hooks.
   - Subheaders: 12.1 main.py wiring, 12.2 training/__init__.py export, 12.3 Tests

## Formula Catalog (explicit)
```
g0_inv = inverse(g0)
K_{mu nu rho sigma} = (1/n^2) * g0_inv^{mu rho} * g0_inv^{nu sigma}
R_sc(x) = sqrt(|(1/n^2) * g^{mu rho} * g^{nu sigma} * R_{mu nu rho sigma}(x)| + eps)
R_op[a,b] = R_{mu_a nu_a rho_b sigma_b}
C = E[z z^T], Q = eigvecs(C)
Q = product_k G(i_k, j_k, theta_k)
g(v,v) = Omega^2 * v^T g0 v + (U^T v)^T D (U^T v)
g(v,v) = v^T g v
cost = R_sc * sqrt(|g(v,v)| + eps)
w_i = softmax(-beta * cost_i)
dLIoR = R_sc * sqrt(|g(v,v)| + eps) * dtau
theta_k <- theta_k - eta * (J_plus - J_minus) / (2 * eps)
```

## Global Pitfalls to Avoid
- CPU tensors or numpy in hot paths cause device syncs and slowdowns.
- Dense rank-4 curvature in the hot path will blow memory.
- Unclamped diag or low-rank terms can make costs negative or explode.
- Rotor angle drift without wrapping can destabilize training.
