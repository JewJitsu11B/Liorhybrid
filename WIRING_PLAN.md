# Liorhybrid Wiring Plan

## Overview

This document captures the plan for connecting disconnected components in the Liorhybrid architecture.

---

## Core Architecture Clarification

### QKV Structure
- **Q** = Concept vector address (query address)
- **K** = Concept vector address (key address)
- **V** = Output vector that does weighted elementwise comparisons

### Address Structure (NOT CompositeK)
Linearized address with fixed-width blocks:
```
[ core | metric | transport | N1-N64 | ecc | timestamps ]
```

**Dimensions (d=512):**
- core: 512 (embedding)
- metric: 512 (diagonal Riemannian)
- transport: 512 (Christoffel coefficients)
- neighbors: 64 × 88 = 5632
- ecc: 32 (BCH error correction)
- timestamps: 2
- **Total: 7202 floats**

### Neighbor Structure (32 + 16 + 16 = 64)
- **N1-N32**: Absolute nearest (similarity grounding)
- **N33-N48**: Attractors (reinforcing evidence)
- **N49-N64**: Repulsors (contrastive evidence)

### Per-Neighbor Block (88 floats)
- **value**: 64 (interaction output vector, NOT raw embedding)
- **scores**: 8 (multiple similarity score types, expandable to ~6 types)
- **coords**: 16 (routing info)

### Neighbor Values
Each neighbor slot gets the **output vector of that vector's interaction** - the result of QKV attention for that neighbor, not the raw embedding.

### Similarity Scores
Plan to add ~6 similarity score types onto the concatenated address:
- Concatenations can do **elementwise** or **pairwise** as desired
- Score types TBD (physics-based and/or geometric)

---

## Components to Wire

### 1. Address Structure
**File:** `inference/address.py`
**Status:** Defined but not connected
**Action:** Wire into geometric attention as the Q/K structure

Components:
- AddressConfig (dimensions)
- Address class (linearized access)
- AddressBuilder (constructs from embeddings + neighbors)

### 2. Coordinate Head
**Location:** `AddressBuilder.coord_proj` and similar
**Status:** Exists in Address/CompositeK
**Action:** Ensure coord projection is generating routing coords for neighbors
**Note:** This is SEPARATE from TrialityCoordinateHead

### 3. TrialityCoordinateHead
**File:** `inference/triality_coordinate_head.py`
**Status:** Imported but never used in forward pass
**Action:** Wire in separately (NOT into coordinate head)
**Purpose:** Provides (geo, spec, mem) 8D coordinates for Spin(8) structure

### 4. ConstitutiveState (Non-Markovian Material Memory)
**File:** `inference/constitutive_state.py`
**Status:** Defined but not connected
**Action:** Wire ConstitutiveLayer into processing stack

Features:
- Bivector decomposition: elastic (decays) / plastic (persists) / excess (staging)
- Material properties: elasticity, yield threshold, fatigue
- Phase transitions for concept formation
- "Same input ≠ same output" - history-dependent responses

### 5. Symplectic (ComplexMetricTensor)
**File:** `models/complex_metric.py`
**Status:** Needs verification
**Action:** Verify A (Riemannian) + iB (Symplectic) is actually used in forward pass

Structure:
- A_μν = symmetric Riemannian metric (configuration space)
- B_μν = antisymmetric symplectic form (phase/momentum space)

### 6. Geometric Diagnostics
**File:** `training/geometric_diagnostics.py`
**Status:** Defined but not called from trainer2
**Action:** Wire E_geo, E_var, E_struct logging into training loop

Metrics:
- E_geo: Geodesic residual |γ̈ + Γ(γ̇, γ̇)| → 0
- E_var: LIoR optimality gap → 0
- E_struct: Curvature-velocity coupling → 0

### 7. LIoR Memory
**Status:** Needs verification
**Action:** Verify fractional memory with power-law decay is hooked up

### 8. Operators (measure_observable, soft_projection)
**File:** `operators/collapse.py`
**Status:** NotImplementedError stubs
**Action:** Implement:
- `measure_observable`: Tr(T† O T) / Tr(T† T) for monitoring
- `soft_projection`: T' = (1-s)T + s*P*T*P† for active inference

---

## Existing Connected Components

### Already Working
- CognitiveTensorField (core/)
- GeometricAttention with Born × Gibbs × Softmax gate (inference/)
- GeometricStack (inference/)
- geometric_products: wedge, tensor, spinor, hodge (inference/)
- Gibbsmax selector (inference/inference.py)
- trainer2 with manual CUDA updates (training/)
- Hamiltonian/Bayesian/Fractional kernels (kernels/)
- pipeline_audit telemetry (utils/)

### Gibbsmax
Located in `inference/inference.py:228`:
```python
if selector == "gibbsmax":
    return torch.softmax(scores / float(max(tau, 1.0e-8)), dim=-1)
```

And in `geometric_attention.py:440-445` (triple gate fusion):
```python
born = similarity.pow(2)                    # |ψ|² amplitude (quantum)
gibbs = torch.exp(-similarity.abs() / tau)  # exp(-E/T) cost (Boltzmann)
soft = torch.softmax(similarity / tau, dim=-1)
weights = born * gibbs * soft               # All three combined
```

---

## NOT Using

- **CompositeK** - Address structure is the core, not CompositeK
- **MoE Framework** - Incompatible with trainer2 (needs optimizer)
- **"New folder" trainer2** - Current trainer2.py IS the new one
- **Stability.py** - Not yet

---

## File Locations

```
inference/
├── address.py                 # Address structure (TO WIRE)
├── triality_coordinate_head.py # Triality coords (TO WIRE separately)
├── constitutive_state.py      # Non-Markovian memory (TO WIRE)
├── geometric_attention.py     # Main attention (WORKING)
├── geometric_products.py      # Wedge/tensor/spinor (WORKING)
├── geometric_stack.py         # Full stack (WORKING)
└── composite_k.py             # NOT USING

models/
└── complex_metric.py          # Symplectic A+iB (TO VERIFY)

training/
├── trainer2.py                # Main trainer (WORKING)
└── geometric_diagnostics.py   # E_geo/E_var/E_struct (TO WIRE)

operators/
└── collapse.py                # measure_observable, soft_projection (TO IMPLEMENT)

kernels/
├── hamiltonian.py             # (WORKING)
├── bayesian.py                # (WORKING)
└── fractional_memory.py       # LIoR memory (TO VERIFY)
```

---

## Priority Order

1. **Address structure** - Core Q/K architecture
2. **Coordinate head** - Routing coords for neighbors
3. **LIoR memory verification** - Ensure hooked up
4. **ConstitutiveState** - Non-Markovian material memory
5. **Symplectic verification** - Ensure A+iB used
6. **TrialityCoordinateHead** - Spin(8) structure (separate from coord head)
7. **Geometric diagnostics** - Training monitoring
8. **Operators implementation** - measure_observable, soft_projection

---

*Generated by Claude Code audit session*
