# Results: the CAL field equation collapses to Schrödinger as $\alpha\to1^-$

**The paper** (title: *Schrödinger evolution at the α→1⁻ edge of the Causal Accumulation Law: derived unitarity, the free Galilean limit, and carrier entanglement as time dilation*). `docs/paper/cal_schrodinger_collapse.pdf` (source: `docs/paper/cal_schrodinger_collapse.tex`) contains the full statements and proofs. This file summarizes them.

**Reproduce.** `python3 scripts/cal_schrodinger_collapse.py` (numpy only, about 10 seconds). Units: $\hbar=\tau_c=1$, and $c=1$ in §4.

**Integration.** Weyl integration over the whole past is used throughout, because that is the integral the CAL writes. No Caputo (initial-value) problems are used. A Caputo problem is the Weyl derivative of a past frozen at $J_0$. A frozen past has zero-frequency content, on which the field equation's Weyl integral diverges, so a Caputo problem never solved the field equation. Starting a solution is instead modelled by a source pulse acting on a zero past.

**Update (readout and Markov limits; paper Sec. 6).**
- The two Markov limits are the forgetful (Dirac) limit and the faithful (full light-cone) limit. A closed system cannot reach the faithful one.
- A Dirac memory kernel on the carrier damps every mode.
- The edge is Markov through the fundamental theorem of calculus, which is neither of the two limits. It keeps the accumulated sum, not each contribution. For $\alpha<1$ the power-law memory needs a continuum of auxiliary modes; as $\alpha\to1^-$ these collapse onto one mode, the state itself. So the edge needs no larger state space and no records.
- A single carrier $J\in M_2(\mathbb C)\cong\mathbb C^2\otimes\mathbb C^2$ is two factors. $JJ^\dagger$ is the reduced state of the left factor, and the concurrence between the factors equals $1/\gamma$.
- Lorentz action on the closure: left $SL(2,\mathbb C)$ acts on $JJ^\dagger$ as the Lorentz group. Rotations leave the entanglement unchanged; boosts change it, with $C=\operatorname{sech}\eta$ and Schmidt weights $\propto e^{\pm\eta}$ (Doppler factors). So a carrier's internal entanglement is the time dilation of its current. The Lorentz structure of qubits is itself known (Verstraete–Dehaene–De Moor; Samuel–Shivam–Sinha); the reading is the new part.
- The trace readout ${\rm Tr}(P_mJJ^\dagger)/{\rm Tr}(JJ^\dagger)$ survives composition at any rank. Product carriers give product statistics, and marginals are consistent.
- Intrinsic normalization above $M_2$ needs $\Lambda^2J=0$ (all $2\times2$ minors vanish), not just $\det J=0$.
- The edge operator contains only single-carrier, one-way transport, so it cannot create entanglement between carriers. The reason is the transport structure, not linearity: a coupling would need its adjoint, the return leg, to stay unitary, and reciprocal structure belongs to $\alpha=2$.
- The kinetic term is a scaling limit, not a truncation. With the full spatial symbol $r_1\hat g(-i\nabla)$, shrinking the width at fixed $\hbar^2/2m$ gives $\hbar^2k^2/2m$ exactly; with a Gaussian profile no energy cap remains.

---

## 1. Summary

1. **Collapse.** With $\varepsilon=1-\alpha\to0$ and $\Lambda=\varepsilon\hat R$ held fixed, the bounded solutions converge to Schrödinger evolution, linearly in $\varepsilon$.
2. **Unitarity is derived.** For every $\alpha<1$, the bounded solutions on the whole time line form a unitary group. The measured norm drift is $3\times10^{-16}$.
3. **The limit is one-sided.**
   - The Weyl integral converges only for $\alpha<1$.
   - The memory-generated coefficients change sign across $\alpha=1$.
   - Above $\alpha=1$ the one-way edge equation has no causal, bounded continuation: its response to a source grows or precedes the source. This is a property of that equation, not of the framework on $(1,2]$, and no third derivative is involved. The runs show both branches.
4. **Two Markovian regimes, neither sufficient.**
   - At $\alpha=1$ the past disconnects. With the kernel's time unit $\tau_c$ fixed, it disconnects even from below: the memory rate is about $\varepsilon$.
   - A normalizable kernel, including any $\delta$-like one, has no stationary band, because $\lvert\hat w\rvert\le1<\alpha/(\varepsilon\lvert\lambda\rvert)$. A narrow exponential kernel gets the real part of the energy right but decays at $1/\tau_w$.
   - The power law works because it is not normalizable, and its symbol's phase is the channel angle. The edge has a memory law with a Markovian flow, at fixed memory rate $(1-\alpha)/\tau_c$.
5. **The edge is Galilean and free.**
   - The Lorentz signature belongs to the $\alpha=2$ closure (via $\det JJ^\dagger$). Born uses only the trace, a frame-dependent density. So the edge has time ordering but no light cone.
   - The generator is built from one-way transports. It gives the free Schrödinger form, at most in a fixed one-way background. Interaction and entanglement need $\alpha=2$.
6. **Withdrawn:**
   - The light-cone dispersion is now in Appendix C of the paper, labelled as $\alpha=2$ structure; it is not an edge result.
   - The claim "a causal $\delta$-kernel gives no kinetic term" assumed a cone that the edge does not have.
   - The Caputo $1/\alpha$ norm.

---

## 2. The two Markovian regimes

| Regime | What happens at the edge |
|---|---|
| $\alpha=1$ corner | past disconnected, no generator (skeleton) |
| $\delta$ / narrow kernel | no stationary band; modes decay at $1/\tau_w$ |
| power law, rate $\varepsilon/\tau_c$ fixed | flat accumulator; exact unitary band; Schrödinger |

---

## 3. Lag-independent kernel ($\sigma=-1$, Gaussian profile, $\langle r^2\rangle=0.75$, $A=4$, $\hbar^2/2m=1/2$)

**A. Bounded solutions on the whole time line**, with the full symbol. Every mode is stationary, so there is no cap. Packet: $k_0=1$, width 2.

| $\varepsilon$ | $\max\lvert E_\alpha-E_1\rvert$ | $\varepsilon\max\Lambda\lvert1+\ln\Lambda\rvert$ | ratio | phase-aligned packet error, $t\le40$ | norm drift |
|---|---|---|---|---|---|
| 1e-1 | 1.246 | 0.954 | 1.305 | 1.41 | 6e-16 |
| 1e-2 | 9.78e-2 | 9.55e-2 | 1.025 | 2.87e-1 | 4e-16 |
| 1e-3 | 9.57e-3 | 9.55e-3 | 1.002 | 2.83e-2 | 6e-16 |
| 3e-4 | 2.87e-3 | 2.86e-3 | 1.001 | 8.47e-3 | 6e-16 |

**A2. Width scaling** (Gaussian, $A\langle r^2\rangle/6=\hbar^2/2m=1/2$ fixed). The table gives the relative error of the kinetic term; the bound is $\langle r^2\rangle k^2/12$.

| $\langle r^2\rangle$ | offset $-A$ | $k=0.5$ | $k=1$ | $k=2$ | $k=4$ | bound at $k=4$ |
|---|---|---|---|---|---|---|
| 1 | −3 | 2.05e-2 | 7.89e-2 | 0.270 | 0.651 | 1.33 |
| 0.1 | −30 | 2.08e-3 | 8.29e-3 | 3.26e-2 | 0.122 | 0.133 |
| 0.01 | −300 | 2.08e-4 | 8.33e-4 | 3.33e-3 | 1.32e-2 | 1.33e-2 |
| 0.001 | −3000 | 2.08e-5 | 8.33e-5 | 3.33e-4 | 1.33e-3 | 1.33e-3 |

**B. Source-driven runs.** The integral equation $T=S$ is time-stepped directly: Weyl integration over a zero past, full history kept, a Gaussian pulse at $t_0=10$ of width 2.

*B1. Modes with $\Lambda=+1$.* After the pulse, the mode oscillates forever at the residue amplitude.

| $\alpha$ | $h$ | mean $\lvert J\rvert$, $t\in[80,120]$ | prediction | ratio |
|---|---|---|---|---|
| 0.900 | 0.02 | 6.95236 | 6.95787 | 0.99921 |
| 0.900 | 0.01 | 6.95674 | 6.95787 | 0.99984 |
| 0.990 | 0.01 | 5.16659 | 5.16724 | 0.99987 |
| 0.999 | 0.01 | 5.02771 | 5.02833 | 0.99988 |
| 1.000 | 0.01 | 5.01264 | 5.01326 | 0.99988 |
| 1.010 | 0.01 | 4.86571 | 4.86630 | 0.99988 |

*B2. Modes with $\Lambda=-1$.* Below $\alpha=1$ these exist only if $\hat g$ changes sign; above $\alpha=1$ they are the whole sector. The zero of the response function, compared with the measured late-time rate.

| $\alpha$ | zero | $-{\rm Im}\,E$ | measured rate |
|---|---|---|---|
| 0.970 | 1.027032 − 0.100104 i | +1.0010e-1 | +1.0007e-1 |
| 0.990 | 1.009695 − 0.032052 i | +3.2052e-2 | +3.2049e-2 |
| 0.999 | 1.000997 − 0.003148 i | +3.1479e-3 | +3.1478e-3 |
| 1.000 | 1.000000 | 0 | +2.5e-7 |
| 1.001 | 0.998997 + 0.003135 i | −3.1353e-3 | −3.1348e-3 |
| 1.010 | 0.989718 + 0.030795 i | −3.0795e-2 | −3.0792e-2 |
| 1.030 | 0.967645 + 0.088790 i | −8.8790e-2 | −8.8782e-2 |

*B3. Bounded responses from real-axis inversion ($\Lambda=-1$).*

| $\alpha$ | $\lvert J(t_0-10)\rvert$ | $\lvert J(t_0-20)\rvert$ | $\lvert J(t_0+40)\rvert$ |
|---|---|---|---|
| 0.99 | 4.0e-6 (the Gaussian's own tail) | 7.5e-11 | 1.44 |
| 1.01 | **3.58** (before the pulse) | **2.63** | 4.2e-6 |

From above, a bounded response has to come before its source.

---

## 4. Light cone imposed by hand ($\alpha=2$ structure, not the edge)

This is recorded in Appendix C of the paper. With a cone imposed, the dispersion would be $E=\hbar kv\coth(\hbar kv/E_0)$, with $m=3E_0/(2v^2)$. The framework places the Lorentz signature at $\alpha=2$, so this is not a result about the $\alpha\to1^-$ edge.

## 5. Caveats

- **Proof rigor.** The proofs are at the rigor level of the companion papers; the measurability details of the main unitarity proof are not written out.
- **Interaction and Lorentz structure** are assigned to $\alpha=2$ and are not treated here.
- **Not yet done:** gauge coupling (Hermitian $\hat R$ with $U(1)$ holonomy) and position-dependent potentials beyond a bounded offset.
- **No mass value.** The mass scale is not derived.
