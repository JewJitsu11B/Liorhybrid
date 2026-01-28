# Replacing MoE with Physics-Based Architecture

## The Problem with Standard MoE

Traditional Mixture-of-Experts (MoE):
```
Input → Router (softmax gating) → Top-K Experts → Weighted Sum → Output
```

Issues:
- Router is a learned neural network (black box)
- Load balancing requires auxiliary losses
- Expert selection is discrete/discontinuous
- No physical interpretation of "expertise"
- Experts don't interact - just weighted averaging

---

## Physics-Based Replacement

### Core Insight
Instead of "experts" selected by a learned router, use **physics fields** where:
- **Routing** = geodesic flow on learned manifold
- **Expert selection** = field collapse via Born rule
- **Expert interaction** = geometric products (wedge, tensor, spinor)
- **Load balancing** = natural from energy conservation

---

## Architecture Sketch

### Standard MoE
```
x → Router(x) → [e1, e2, ..., en] → Σ w_i * Expert_i(x) → y
     ↓
   softmax
   top-k
```

### Physics Replacement
```
x → Address(x) → Field Evolution → Geometric Attention → y
        ↓              ↓                    ↓
    Q/K coords    T_ij dynamics      Born×Gibbs×Soft
    metric        Hamiltonian        wedge/tensor/spinor
    transport     Bayesian update    field contraction
    neighbors     LIoR memory
```

---

## Component Mapping

| MoE Component | Physics Replacement | How It Works |
|---------------|---------------------|--------------|
| **Router** | Address metric + transport | Geodesic routing via Christoffel symbols |
| **Expert selection** | Born gate |ψ|² | Quantum-like collapse selects "experts" |
| **Expert weights** | Gibbs e^(-E/T) | Energy-based weighting (Boltzmann) |
| **Experts** | Neighbor slots (32+16+16) | Each neighbor = localized "expert" |
| **Expert FFN** | Geometric products | Wedge/tensor/spinor compute interactions |
| **Load balance loss** | Energy conservation | Hamiltonian naturally distributes load |
| **Capacity factor** | Yield threshold | Constitutive material limits activation |

---

## Detailed Replacement

### 1. Router → Address Structure

**MoE Router:**
```python
router_logits = W_router @ x  # learned weights
gates = softmax(router_logits)
top_k_gates, top_k_indices = topk(gates, k)
```

**Physics Replacement:**
```python
# Build address with metric (how to measure distance)
addr = AddressBuilder(x)
# addr.metric = local Riemannian metric
# addr.transport = Christoffel symbols (how coords change)

# Geodesic routing: find nearest points in curved space
# NOT Euclidean distance - uses learned metric g_ij
distances = geodesic_distance(addr.metric, addr.transport, x, neighbors)
```

### 2. Expert Selection → Born + Gibbs Gate

**MoE Selection:**
```python
# Hard top-k or soft gating
selected = top_k(router_output, k=2)
```

**Physics Replacement:**
```python
# Born rule: quantum amplitude → probability
born = similarity.pow(2)  # |ψ|²

# Gibbs: energy-based selection
gibbs = torch.exp(-energy / temperature)  # e^(-E/T)

# Combined gate (no hard cutoff)
weights = born * gibbs * softmax(similarity / T)
```

This is **continuous** (differentiable everywhere) and has **physical meaning**:
- Born: "How much does this neighbor resonate with query?"
- Gibbs: "How energetically favorable is this interaction?"

### 3. Experts → Neighbor Interaction Outputs

**MoE Experts:**
```python
expert_outputs = [expert_i(x) for i in selected]
output = sum(w_i * expert_outputs[i])
```

**Physics Replacement:**
```python
# Each neighbor slot holds INTERACTION OUTPUT (not raw embedding)
# The "expert computation" IS the geometric product

for i, neighbor in enumerate(neighbors):
    # Geometric products compute the interaction
    wedge_i = wedge_product(query, neighbor, T_field)   # antisymmetric
    tensor_i = tensor_product(query, neighbor, T_field)  # full correlation
    spinor_i = spinor_product(query, neighbor, T_field)  # rotational

    # Store interaction result in neighbor value slot
    neighbor.value = combine(wedge_i, tensor_i, spinor_i)

# Weighted combination using physics gates
output = sum(weights[i] * neighbors[i].value)
```

### 4. Load Balancing → Energy Conservation

**MoE Load Balancing:**
```python
# Auxiliary loss to prevent expert collapse
load_balance_loss = coefficient * variance(expert_usage)
total_loss = task_loss + load_balance_loss
```

**Physics Replacement:**
```python
# Energy is naturally conserved by Hamiltonian evolution
# No auxiliary loss needed!

# Field energy distributes across neighbors automatically
E_total = compute_energy(T_field)  # conserved quantity

# If one "expert" (neighbor) gets too much energy,
# Hamiltonian dynamics redistributes it
# This IS the physics - not an added loss term
```

### 5. Capacity → Constitutive Material Properties

**MoE Capacity:**
```python
# Hard capacity limit per expert
if expert_usage[i] > capacity:
    drop_tokens()
```

**Physics Replacement:**
```python
# Material yield threshold (constitutive state)
if stress > material.yield_threshold:
    # Excess goes to staging (plastic deformation)
    # NOT dropped - stored for later
    excess = decompose_bivector(interaction).excess
    staging.append(excess)

# Fatigue tracking prevents overuse naturally
material.fatigue += stress * fatigue_rate
```

---

## Complete Forward Pass

```python
def physics_moe_forward(x, T_field, neighbors):
    """
    Physics-based MoE replacement.

    Args:
        x: Input (batch, seq, d_model)
        T_field: Cognitive tensor field (N_x, N_y, D, D)
        neighbors: 64 neighbor addresses (32 nearest + 16 attract + 16 repulse)
    """
    # 1. Build query address (replaces router input)
    Q_addr = AddressBuilder(x)

    # 2. Compute interactions with all neighbors (replaces expert FFNs)
    for i, K_addr in enumerate(neighbors):
        # Geometric products = "expert computation"
        interaction = geometric_attention(
            Q=Q_addr.core,
            K=K_addr.core,
            T_field=T_field
        )
        # Store in V slot
        neighbors[i].value = interaction

    # 3. Physics-based gating (replaces router softmax)
    similarity = compute_similarity(Q_addr, neighbors, metric=Q_addr.metric)

    born = similarity.pow(2)                    # |ψ|²
    gibbs = torch.exp(-energy(similarity) / T)  # e^(-E/T)
    soft = softmax(similarity / T)

    weights = born * gibbs * soft  # combined gate

    # 4. Weighted combination (same as MoE, but V is interaction output)
    output = einsum('bn,bnd->bd', weights, neighbors.values)

    # 5. No load balance loss needed - Hamiltonian conserves energy

    return output
```

---

## Benefits Over Standard MoE

| Aspect | Standard MoE | Physics MoE |
|--------|-------------|-------------|
| **Routing** | Learned black box | Interpretable geodesics |
| **Selection** | Discrete top-k | Continuous Born+Gibbs |
| **Experts** | Independent FFNs | Interacting via geometry |
| **Balance** | Auxiliary loss | Natural conservation |
| **Capacity** | Hard drop | Soft yield + staging |
| **Memory** | Stateless | Non-Markovian (constitutive) |
| **Gradients** | Through router only | Through field everywhere |

---

## Implementation Checklist

- [ ] Replace `SupervisorGating` with `AddressBuilder` metric routing
- [ ] Replace `expert.forward()` with geometric products
- [ ] Replace softmax gating with Born × Gibbs × Soft
- [ ] Remove load balance loss (Hamiltonian handles it)
- [ ] Add constitutive state for capacity/yield
- [ ] Wire neighbor structure (32+16+16)
- [ ] Add ~6 similarity score types to address

---

## File Changes Needed

```
moe_framework/supervisor.py    → DELETE or archive
moe_framework/expert.py        → DELETE or archive
moe_framework/constellation.py → DELETE or archive

inference/address.py           → KEEP, wire as Q/K
inference/geometric_attention.py → KEEP, this IS the "expert"
inference/constitutive_state.py  → WIRE for capacity/yield
```

---

*Physics doesn't need a router - geodesics know where to go.*
