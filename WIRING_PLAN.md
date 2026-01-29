# Liorhybrid Wiring Plan

## Overview

This document captures the plan for connecting disconnected components in the Liorhybrid architecture.

---

## Core Architecture Clarification

### QKV Structure
- **Q** = Concept vector address (query address)
- **K** = Concept vector address (key address)
- **V** = Output vector that does weighted elementwise comparisons

### Address Structure (NOT CompositeK) - OPTION 6 IMPLEMENTATION
Linearized address with fixed-width blocks (mandatory neighbor probing):
```
[ core | metric | transport | N1-N64 | ecc | timestamps ]
```

**Dimensions (d=512, UPDATED):**
- core: 512 (embedding)
- metric: 512 (diagonal Riemannian)
- transport: 512 (Christoffel coefficients)
- neighbors: 64 × 118 = 7552 (expanded with per-neighbor metric/transport)
- ecc: 32 (collision-avoidance hash)
- timestamps: 2
- **Total: 9122 floats**

### Neighbor Structure (32 + 16 + 16 = 64) - MANDATORY, NO FALLBACKS
- **N1-N32**: Absolute nearest (32 neighbors, highest similarity)
- **N33-N48**: Attractors (16 neighbors, top similarity after nearest)
- **N49-N64**: Repulsors (16 neighbors, lowest similarity for contrast)

**Role Assignment:**
- Nearest: Selected via top-32 similarity ranking
- Attractors: Next top-16 after excluding nearest
- Repulsors: Bottom-16 similarity for contrastive evidence
- All 64 slots MUST be filled (repeats candidates if fewer than 64 available)

### Per-Neighbor Block (118 floats, UPDATED)
- **value**: 64 (interaction output vector, projected from neighbor embedding)
- **neighbor_metric**: 16 (metric features of this neighbor via projection)
- **neighbor_transport**: 16 (transport features of this neighbor via projection)
- **scores**: 6 (6 similarity score types, MANDATORY)
- **coords**: 16 (routing info)

**Block Layout:** `[value | neighbor_metric | neighbor_transport | scores | coords]`

### Similarity Scores (6 types, MANDATORY)
Exactly 6 similarity scores per neighbor (no optionality at runtime):
1. **Score 0**: Cosine similarity (geometric baseline, computed from embeddings)
2. **Scores 1-5**: Learned similarity metrics (5 learned scores via projection)

**Computation:**
- If `neighbor_similarities` provided externally: Use for score 0
- If not provided: Compute cosine similarity internally (no empty slots)
- Scores 1-5 always computed via learned projections (`similarity_proj`)

### Neighbor Metric/Transport Features (NEW)
Each neighbor block stores geometric features OF THAT NEIGHBOR:
- `neighbor_metric`: 16-dim projection of neighbor embedding through `neighbor_metric_proj`
- `neighbor_transport`: 16-dim projection of neighbor embedding through `neighbor_transport_proj`
- These are NOT the main token's metric/transport, but features derived from each neighbor
- Allows neighbor blocks to carry geometric context about their own embeddings

### ECC and Timestamps (Present, Excluded from Scoring)
- **ECC field**: 32 bits storing collision-avoidance hash (first 32 of 64-bit route hash)
- **Timestamps**: 2 floats (internal_time, wall_time) for causality
- **Behavior**: Present in address for integrity/causality
- **Exclusion**: NOT used in neighbor similarity scoring or selection
- **Purpose**: ECC provides address-space entropy for uniqueness; timestamps for temporal ordering

### Collision Avoidance & Uniqueness
**Challenge:** With N tokens, avoid address collisions (same address for different content)

**Solution:**
1. **Route hash projection**: 64-bit hash via learned projection (`route_hash_proj`)
2. **Storage**: First 32 bits stored in ECC field
3. **Uniqueness helpers**:
   - `check_address_collisions(addr, threshold=0.99)`: Detects collision pairs
   - `compute_address_uniqueness_score(addr)`: Returns score ∈ [0,1], higher = more unique
4. **Entropy**: Hash adds extra bits beyond embedding similarity for collision mitigation

### Optionality Definition (CLARIFIED)
**"Optional" means:**
- ONLY during transitional blackout while wiring new features (temporary disable flag)
- NEVER at runtime when `enable_address_probing=True` (default)
- When probing enabled: all 64 neighbor slots MUST be filled, 6 scores MUST be computed
- ECC/timestamps always present but excluded from neighbor scoring logic

**Runtime Behavior:**
- `AddressConfig.enable_address_probing = True` (default): Full 64-slot probing active
- If disabled (testing only): Core/metric/transport populated, neighbors zero (not recommended)

---

## Components to Wire

### 1. Address Structure (OPTION 6 - IMPLEMENTED ✓)
**File:** `inference/address.py`
**Status:** ✓ IMPLEMENTED - Mandatory 64-slot neighbor probing active
**Implementation Date:** 2026-01-28

**What's Implemented:**
- ✓ AddressConfig with 6 similarity scores (m=6)
- ✓ 64 neighbors: 32 nearest, 16 attractors, 16 repulsors (mandatory)
- ✓ Per-neighbor metric/transport features (16 dims each)
- ✓ 6 similarity scores computed per neighbor (cosine + 5 learned)
- ✓ Collision-avoidance hash (64-bit route hash, 32 bits in ECC field)
- ✓ Helper functions: check_address_collisions, compute_address_uniqueness_score
- ✓ Comprehensive tests in test_address_builder.py (all passing)

**Components:**
- AddressConfig (dimensions, enable_address_probing flag)
- Address class (linearized access with role-typed neighbor views)
- AddressBuilder (constructs from embeddings + neighbors with mandatory probing)

**New Methods:**
- `compute_similarity_scores()`: 6 scores per neighbor (cosine + learned)
- `select_neighbors()`: Role-typed selection (32+16+16)
- `compute_collision_hash()`: Uniqueness via route hash
- Access methods: `all_neighbor_metrics`, `all_neighbor_transports`, etc.

**Next Steps:**
- Wire into geometric_attention.py as Q/K structure
- Integrate probing path to consume neighbor scores/coords (no dense matmul)


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

1. ✓ **Address structure (Option 6)** - COMPLETED 2026-01-28
   - Mandatory 64-slot neighbor probing implemented
   - 6 similarity scores per neighbor
   - Metric/transport features per neighbor
   - Collision-avoidance hash system
   - Comprehensive tests passing
2. **Coordinate head** - Routing coords for neighbors (Implemented in AddressBuilder.coord_proj)
3. **LIoR memory verification** - Ensure hooked up
4. **ConstitutiveState** - Non-Markovian material memory
5. **Symplectic verification** - Ensure A+iB used
6. **TrialityCoordinateHead** - Spin(8) structure (separate from coord head)
7. **Geometric diagnostics** - Training monitoring
8. **Operators implementation** - measure_observable, soft_projection

---

## Implementation Summary: Address-Based Neighbor Probing (Option 6)

**Implemented:** 2026-01-28

### Core Features
1. **64-slot neighbor structure** with mandatory role typing:
   - 32 nearest neighbors (highest similarity)
   - 16 attractors (top similarity after nearest)
   - 16 repulsors (lowest similarity for contrast)

2. **6 similarity score types** per neighbor (no optionality):
   - Score 0: Cosine similarity (baseline)
   - Scores 1-5: Learned metrics via projection

3. **Per-neighbor geometric features**:
   - 16-dim metric features (neighbor's own geometry)
   - 16-dim transport features (neighbor's own connection)
   - Allows each neighbor to carry geometric context

4. **Collision avoidance system**:
   - 64-bit route hash via learned projection
   - 32 bits stored in ECC field
   - Helper functions for collision detection and uniqueness scoring

5. **Total address dimension**: 9122 floats (d=512)
   - Core: 512
   - Metric: 512
   - Transport: 512
   - Neighbors: 64 × 118 = 7552
   - ECC: 32
   - Timestamps: 2

### Implementation Files
- `inference/address.py`: Core implementation (AddressConfig, Address, AddressBuilder)
- `tests/test_address_builder.py`: Comprehensive pytest tests
- `test_address_standalone.py`: Standalone validation script

### Test Results
All 10 test categories passing:
- ✓ Config dimensions
- ✓ Address builder shape
- ✓ 64 neighbors populated
- ✓ 6 similarity scores per neighbor
- ✓ Role-typed partitions (32+16+16)
- ✓ Metric/transport per neighbor
- ✓ ECC and timestamps present
- ✓ Collision checking
- ✓ Uniqueness score computation
- ✓ Individual neighbor access

### Next Integration Steps
1. Wire AddressBuilder into geometric_attention.py as Q/K structure
2. Update GeometricAttention to consume neighbor scores/coords
3. Implement probing path (no dense matmul, neighbor-based routing)
4. Add config flag integration for address_probing mode
5. Validate end-to-end with attention mechanism

---

*Generated by Claude Code audit session*
*Updated with Option 6 implementation - 2026-01-28*
