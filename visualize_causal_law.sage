#!/usr/bin/env sage
"""
Causal Accumulation Law Visualization (SageMath)

Core equation:
T^{μν}_{ρσ}(x) = α J^{μν}_{ρσ}(x) - (1-α) ∫_{J^-(x)} k(τ;x,x') Π Γ J(x') Φ d^4x'

Where:
  - J = associator = (ψ_Σ * ψ_Λ) * ψ_α - ψ_Σ * (ψ_Λ * ψ_α)
  - k(τ) = LIoR kernel with three modes
  - α interpolates: QM (α→1, Markovian) ↔ GR (α→0, causal propagator)
"""

from sage.all import *
import numpy as np

# ============================================================================
# PART 1: LIoR Memory Kernel k(τ)
# ============================================================================
# k(τ) = w(τ) · G_del(x,x')
# w(τ) has three characteristic modes:
#   - Exponential decay: e^{-τ/τ_0}
#   - Power-law (fractional): τ^{-β}
#   - Oscillatory: cos(ωτ)

var('tau, tau0, beta, omega')

# Define the three kernel modes
k_exp = exp(-tau / tau0)
k_power = tau^(-beta)
k_osc = cos(omega * tau)

# Combined kernel (weighted sum)
var('w1, w2, w3')
k_combined = w1 * k_exp + w2 * k_power + w3 * k_osc

print("=" * 60)
print("LIoR MEMORY KERNEL k(τ)")
print("=" * 60)
print(f"Exponential mode:  k_exp(τ) = exp(-τ/τ₀)")
print(f"Power-law mode:    k_pow(τ) = τ^(-β)")
print(f"Oscillatory mode:  k_osc(τ) = cos(ωτ)")
print(f"\nCombined: k(τ) = w₁·k_exp + w₂·k_pow + w₃·k_osc")

# Plot the three modes
p1 = plot(exp(-x/2), (x, 0.1, 10), color='blue', legend_label='Exponential (τ₀=2)', thickness=2)
p2 = plot(x^(-0.5), (x, 0.1, 10), color='red', legend_label='Power-law (β=0.5)', thickness=2)
p3 = plot(cos(2*x), (x, 0.1, 10), color='green', legend_label='Oscillatory (ω=2)', thickness=2)

kernel_plot = p1 + p2 + p3
kernel_plot.axes_labels(['$\\tau$ (proper time)', '$k(\\tau)$'])
kernel_plot.set_legend_options(loc='upper right')
kernel_plot.save('kernel_modes.png', figsize=[10, 6], dpi=150)
print("\nSaved: kernel_modes.png")

# ============================================================================
# PART 2: α Parameter - QM ↔ GR Interpolation
# ============================================================================
# T = α·J + (1-α)·∫ k·Π·Γ·J·Φ
#
# α → 1: QM limit (Markovian, local, no memory)
#        T ≈ J (pure associator dynamics)
#
# α → 0: GR limit (full causal propagator with memory)
#        T ≈ ∫ k·Π·Γ·J·Φ (accumulated history)

var('alpha, x')

# Symbolic representation of the two terms
J_term = var('J')  # Local associator
Memory_term = var('M')  # ∫ k·Π·Γ·J·Φ (memory integral)

T_symbolic = alpha * J_term + (1 - alpha) * Memory_term

print("\n" + "=" * 60)
print("α INTERPOLATION: QM ↔ GR")
print("=" * 60)
print(f"T = α·J + (1-α)·M")
print(f"\nwhere M = ∫_{{J⁻(x)}} k(τ;x,x') Π Γ J(x') Φ d⁴x'")
print(f"\nα → 1: QM limit (Markovian)  →  T ≈ J")
print(f"α → 0: GR limit (causal)     →  T ≈ M")

# Visualize the interpolation
# For visualization, let J=1 (local) and M varies with position
def T_field(alpha_val, j_val, m_val):
    return alpha_val * j_val + (1 - alpha_val) * m_val

# Create interpolation visualization
alpha_range = srange(0, 1.01, 0.01)
j_fixed = 1.0
m_fixed = 0.3  # Memory contribution

T_qm_limit = [T_field(a, j_fixed, m_fixed) for a in alpha_range]
local_contrib = [a * j_fixed for a in alpha_range]
memory_contrib = [(1-a) * m_fixed for a in alpha_range]

p_total = list_plot(list(zip(alpha_range, T_qm_limit)), plotjoined=True,
                    color='purple', thickness=2, legend_label='T (total)')
p_local = list_plot(list(zip(alpha_range, local_contrib)), plotjoined=True,
                    color='blue', thickness=2, linestyle='--', legend_label='α·J (local)')
p_mem = list_plot(list(zip(alpha_range, memory_contrib)), plotjoined=True,
                  color='red', thickness=2, linestyle='--', legend_label='(1-α)·M (memory)')

interp_plot = p_total + p_local + p_mem
interp_plot.axes_labels(['$\\alpha$', 'Contribution'])
interp_plot.set_legend_options(loc='center right')
interp_plot.save('alpha_interpolation.png', figsize=[10, 6], dpi=150)
print("\nSaved: alpha_interpolation.png")

# ============================================================================
# PART 3: Associator J = (ψ_Σ * ψ_Λ) * ψ_α - ψ_Σ * (ψ_Λ * ψ_α)
# ============================================================================
# The associator measures non-associativity of the division algebra
# For octonions: J ≠ 0 (non-associative)
# For quaternions: J = 0 (associative)

print("\n" + "=" * 60)
print("ASSOCIATOR J (Non-Associativity Measure)")
print("=" * 60)
print("J^{μν}_{ρσ} = (ψ_Σ * ψ_Λ) * ψ_α - ψ_Σ * (ψ_Λ * ψ_α)")
print("\nFor octonions (dim 8): J ≠ 0  →  drives dynamics")
print("For quaternions (dim 4): J = 0  →  associative limit")

# Symbolic associator visualization
# Using a simplified 2D representation of associator magnitude
var('psi1, psi2, psi3')

# In the octonionic case, the associator has structure constants
# Let's visualize ||J|| as a function of the three fields
def associator_magnitude(p1, p2, p3, gamma=0.1):
    """Simplified associator magnitude for visualization.
    gamma controls non-associativity strength."""
    return gamma * abs(p1 * p2 * p3) * sin(p1 - p2 + p3)^2

# 2D slice: fix psi3, vary psi1 and psi2
psi3_fixed = 1.0
gamma = 0.3  # Non-associativity strength

associator_plot = contour_plot(
    lambda p1, p2: gamma * abs(p1 * p2 * psi3_fixed) * sin(p1 - p2 + psi3_fixed)^2,
    (0.1, 3), (0.1, 3),
    contours=15,
    colorbar=True,
    cmap='viridis'
)
associator_plot.axes_labels(['$\\psi_\\Sigma$', '$\\psi_\\Lambda$'])
associator_plot.save('associator_magnitude.png', figsize=[8, 8], dpi=150)
print("\nSaved: associator_magnitude.png")

# ============================================================================
# PART 4: O(1) Recurrence Implementation
# ============================================================================
# The key computational insight: infinite memory → O(1) recurrence
#
# m_t = ρ·m_{t-1} + η·x_t - ξ·x_{t-p_eff}
#
# This captures the full integral ∫k·Π·Γ·J·Φ with constant memory!

print("\n" + "=" * 60)
print("O(1) RECURRENCE (Computational Implementation)")
print("=" * 60)
print("m_t = ρ·m_{t-1} + η·x_t - ξ·x_{t-p_eff}")
print("\nwhere:")
print("  ρ = decay factor (from kernel)")
print("  η = input weight")
print("  ξ = delayed subtraction (finite horizon)")
print("  p_eff = effective memory window")
print("\nThis gives O(1) computation for infinite memory!")

# Demonstrate the recurrence
def lior_recurrence(x_seq, rho=0.95, eta=0.1, xi=0.05, p_eff=10):
    """O(1) recurrence for causal accumulation."""
    T = len(x_seq)
    m = [0.0] * T  # Memory state

    for t in range(T):
        if t == 0:
            m[t] = eta * x_seq[t]
        else:
            delayed = x_seq[t - p_eff] if t >= p_eff else 0
            m[t] = rho * m[t-1] + eta * x_seq[t] - xi * delayed

    return m

# Generate sample input (e.g., associator current J)
T_steps = 200
x_input = [sin(0.1 * t) + 0.3 * cos(0.05 * t) + 0.1 * (hash(str(t)) % 100 - 50) / 50
           for t in range(T_steps)]

# Run recurrence with different parameters
m_qm = lior_recurrence(x_input, rho=0.5, eta=0.5, xi=0.0, p_eff=10)   # QM-like (short memory)
m_gr = lior_recurrence(x_input, rho=0.99, eta=0.1, xi=0.02, p_eff=50)  # GR-like (long memory)
m_mid = lior_recurrence(x_input, rho=0.9, eta=0.2, xi=0.05, p_eff=20)  # Intermediate

time_axis = list(range(T_steps))

p_input = list_plot(list(zip(time_axis, x_input)), plotjoined=True,
                    color='gray', thickness=1, legend_label='Input J(t)')
p_qm = list_plot(list(zip(time_axis, m_qm)), plotjoined=True,
                 color='blue', thickness=2, legend_label='QM-like (ρ=0.5)')
p_gr = list_plot(list(zip(time_axis, m_gr)), plotjoined=True,
                 color='red', thickness=2, legend_label='GR-like (ρ=0.99)')
p_mid = list_plot(list(zip(time_axis, m_mid)), plotjoined=True,
                  color='green', thickness=2, legend_label='Intermediate (ρ=0.9)')

recurrence_plot = p_input + p_qm + p_gr + p_mid
recurrence_plot.axes_labels(['Time step $t$', 'Memory state $m_t$'])
recurrence_plot.set_legend_options(loc='upper left')
recurrence_plot.save('recurrence_dynamics.png', figsize=[12, 6], dpi=150)
print("\nSaved: recurrence_dynamics.png")

# ============================================================================
# PART 5: Causal Light-Cone Structure J^-(x)
# ============================================================================
# The integral is over the past light-cone J^-(x)
# This encodes causal structure: only past events contribute

print("\n" + "=" * 60)
print("CAUSAL STRUCTURE: Past Light-Cone J⁻(x)")
print("=" * 60)
print("∫_{J⁻(x)} ... d⁴x'")
print("\nIntegration domain: past light-cone of event x")
print("Ensures: causality (only past influences present)")

# Visualize 2D light-cone (t vs x)
var('t_coord, x_coord')

# Light cone: t^2 - x^2 = 0 (in c=1 units)
# Past cone: t < 0 and |x| < |t|

# Create light-cone plot
light_cone = implicit_plot(t_coord^2 - x_coord^2, (x_coord, -3, 3), (t_coord, -3, 3),
                           color='black', linewidth=2)

# Shade past light-cone region
past_cone = region_plot(
    [t_coord < 0, t_coord^2 > x_coord^2],
    (x_coord, -3, 3), (t_coord, -3, 0),
    incol='lightblue', bordercol='blue', alpha=0.5
)

# Mark observation point
point_x = point([(0, 0)], color='red', size=50, zorder=10)
text_x = text('x (observation)', (0.3, 0.2), fontsize=10, color='red')
text_past = text('J⁻(x) (integration domain)', (0, -1.5), fontsize=10, color='blue')

lightcone_plot = light_cone + past_cone + point_x + text_x + text_past
lightcone_plot.axes_labels(['Space $x$', 'Time $t$'])
lightcone_plot.save('lightcone_structure.png', figsize=[8, 8], dpi=150)
print("\nSaved: lightcone_structure.png")

# ============================================================================
# PART 6: Full Tensor Structure T^{μν}_{ρσ}
# ============================================================================
print("\n" + "=" * 60)
print("FULL CAUSAL ACCUMULATION LAW")
print("=" * 60)
print("""
T^{μν}_{ρσ}(x) = α J^{μν}_{ρσ}(x)
                - (1-α) ∫_{J⁻(x)} k(τ;x,x') Π^{μν}_{ρσ||αβ}^{γδ}
                                   Γ^γ_δ(x,x') J^{αβ}_{γδ}(x')
                                   Φ^{[ρσ]}(x) d⁴x'

Components:
-----------
T^{μν}_{ρσ}  : Rank-4 field tensor (output)
J^{μν}_{ρσ}  : Associator current (non-associativity)
k(τ;x,x')    : LIoR memory kernel
Π            : Parallel transport operator
Γ            : Clifford connection
Φ^{[ρσ]}     : Bivector field (antisymmetric)
α            : QM↔GR interpolation parameter
J⁻(x)        : Past light-cone (causal domain)
""")

# Create summary diagram
summary_text = """
CAUSAL ACCUMULATION LAW STRUCTURE
=================================

INPUT                    KERNEL                   OUTPUT
─────                    ──────                   ──────
     ┌─────────────┐     ┌───────────────┐
J(x')│ Associator  │────▶│   k(τ) LIoR   │
     │ Current     │     │   Memory      │
     └─────────────┘     │   Kernel      │
                         └───────┬───────┘
                                 │
                                 ▼
           ┌─────────────────────────────────────┐
     α ───▶│  T = αJ + (1-α)∫ k·Π·Γ·J·Φ         │───▶ T^{μν}_{ρσ}
           │                                     │
           │  QM ◀────────────────────────▶ GR   │
           │  α=1         α           α=0        │
           └─────────────────────────────────────┘

COMPUTATIONAL TRICK: O(1) Recurrence
───────────────────────────────────────
m_t = ρ·m_{t-1} + η·x_t - ξ·x_{t-p_eff}

Infinite memory → Constant computation!
"""

print(summary_text)

# Save summary to file
with open('causal_law_summary.txt', 'w') as f:
    f.write(summary_text)
print("Saved: causal_law_summary.txt")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("""
Generated files:
  1. kernel_modes.png        - LIoR kernel three modes
  2. alpha_interpolation.png - QM↔GR interpolation
  3. associator_magnitude.png - Associator ||J|| structure
  4. recurrence_dynamics.png - O(1) recurrence behavior
  5. lightcone_structure.png - Causal domain J⁻(x)
  6. causal_law_summary.txt  - Text summary
""")
