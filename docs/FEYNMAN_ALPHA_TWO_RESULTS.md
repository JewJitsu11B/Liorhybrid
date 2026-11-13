# One-pager: the Feynman path sum at closure order α = 2, light-cone kernel width

**Paper:** *An Axiomatic Derivation of the Feynman Path Sum at the Light-Cone Kernel Width and Closure Order α = 2 of the Causal Accumulation Law*. PDF at `docs/paper/cal_feynman_alpha2.pdf`, source in the `.tex` next to it.

**Reproduce:** `python3 scripts/cal_feynman_alpha2.py` (numpy only, about 25 s). Units: ħ = c = 1 and τ_c = 1/√2, so μ = 1.

## Inputs
- The CAL at α = 2: T = 2J + K₂J. In the free, flat sector the transport chain is 1, so there is no coupling constant.
- A light-cone kernel: at lag τ the profile is the light front of radius cτ, with Fourier transform sinc(c|k|τ) in d = 1, 2, 3.
- The balance reading: T is the new state; a free carrier balances its light-cone past, T = 0; a source is the imbalance. The edge's update rule is not used, because at α = 2 the 2J term alone would double the current every update.
- The phase belongs to the source. The double accumulation gives the half turn by itself, and no factor goes on the memory; an added −1 would make the mass tachyonic.
- For the readout only: one sign of frequency, inherited from the source's past, as at the edge.

The kernel's lag scale τ_c is the only scale. Time is relational, so only ratios such as μτ_c are physical.

## Results

| Result | Statement | Check |
|---|---|---|
| Lorentz invariance selects α = 2 | The light-cone symbol depends on s² + c²k² alone only at α = 2 (and α = 0). The leading defect is α(2−α)φ²/6. | defect 0 at α = 0, 2; 0.10 at 1.9 and 2.1 |
| Kernel = retarded Green's function | Kirchhoff's shell in 3+1; one order-1 accumulation per null leg in 1+1 (2 = 1 + 1). | closed form matches direct integration to 6 digits |
| Klein–Gordon | The balanced carrier obeys (□ + μ²)J = 0 with μτ_c = 1/√2. The mass is the kernel's own rate. | roots = √(k² + μ²) to 10 digits |
| Conservative exactly at 2 | Both signs decay at (π/4)(2−α)μ²/ω below 2 and grow at that rate above 2 (runaway). Nothing near 2 selects a sign. | Im ω: −5.5×10⁻³ at 1.99, 0 at 2, +5.6×10⁻³ at 2.01 |
| Path sum | The Neumann series is a sum over piecewise-null histories with −μ² per corner. It converges absolutely (bound ½cosh μt) and resums to G_μ. | 1+1 residual 3.35×10⁻⁵ → 8.38×10⁻⁶ as h halves |
| Checkerboard | In 1+1 the n-th term is the (2n+1)-corner amplitude over 2icμ, since (−μ²)ⁿ = (iμ)²ⁿ. | lattice error halves with ε: 3.7×10⁻² → 4.5×10⁻³ |
| Corner tail | In 3+1, histories without corners sit on the cone; the corners alone build the J₁ tail inside it. | ≤ 2.5×10⁻⁶ |
| Pair composition | The past enters only through (J, ∂ₜJ). That is not the faithful limit. | 1.1×10⁻¹⁴ |
| Unitarity and readout | Each sign of frequency is unitary. On one sign, ∫JJ† is conserved, so the trace readout is Born-normalized in every frame; mixed signs break it. | spread 2.5×10⁻¹⁶ vs 1.6×10⁻¹ |
| Feynman's kernel | The positive-frequency path sum, rest phase removed, gives K_F as c → ∞ with error O(1/c²). | error·c² → 3.667 (predicted 3.672) |

**Where the *i* comes from.** The α = 2 dynamics is real. The *i* of K_F enters only with the sign of frequency inherited from the source. At the edge it is the rotation produced by negative feedback.

## Not claimed / open
- Interaction between carriers, the reciprocal return leg.
- Spin (a Dirac form in 3+1).
- Mass ratios between carriers: with the same kernel every carrier has the same rate.
- The relation to the edge's rest rotation. The equivalence condition (kernel width √6 Compton wavelengths) is a forward pointer only.
- A frame-independent position readout (the Newton–Wigner problem).
- Prior art: the 1+1 checkerboard and its slow limit (Feynman–Hibbs, Jacobson–Schulman, Gaveau–Jacobson–Kac–Schulman) and causal-set chain sums (Johnston). What is new is the source: the history sum is the CAL's own memory series.
