# Results: the CAL field equation collapses to Schrödinger as $\alpha\to1^-$

**The paper.** `docs/paper/cal_schrodinger_collapse.pdf` (source: `docs/paper/cal_schrodinger_collapse.tex`) contains the full statements and proofs. This file summarizes them.

**Reproduce.** `python3 scripts/cal_schrodinger_collapse.py` (numpy only, about 10 seconds). Units: $\hbar=\tau_c=1$, and $c=1$ in §4.

**Integration.** Weyl integration over the whole past is used throughout, because that is the integral the CAL writes. No Caputo (initial-value) problems are used. A Caputo problem is the Weyl derivative of a past frozen at $J_0$. A frozen past has zero-frequency content, on which the field equation's Weyl integral diverges, so a Caputo problem never solved the field equation. Starting a solution is instead modelled by a source pulse acting on a zero past.

---

## 1. Summary

1. **Collapse.** With $\varepsilon=1-\alpha\to0$ and $\Lambda=\varepsilon\hat R$ held fixed, the bounded solutions converge to Schrödinger evolution, linearly in $\varepsilon$.
2. **Unitarity is derived.** For every $\alpha<1$, the bounded solutions on the whole time line form a unitary group. The measured norm drift is $3\times10^{-16}$.
3. **The limit is one-sided.**
   - The Weyl integral converges only for $\alpha<1$.
   - The memory-generated coefficients change sign across $\alpha=1$.
   - From above, the response to a source either runs away or precedes the source. That is the Abraham–Lorentz–Dirac dichotomy, and the runs show both branches.
4. **Markovian without a $\delta$-kernel.** At $\alpha=1$ the kernel is the flat accumulator, whose inverse is $\partial_t$. Separately, a causal $\delta$-kernel has no spatial reach, so it produces no kinetic term. That is why the light cone matters even in a Markovian limit.
5. **The light-cone dispersion.** With memory kept inside the light cone, $E(k)=\hbar kv\coth(\hbar kv/E_0)$. This is causal for all $k$, and it reduces to Schrödinger with $m=3E_0/(2v_{\rm eff}^2)$. It follows that $E_0\le\tfrac23mc^2$.
6. **Withdrawn.** The earlier "switched-on norm $\to\lVert J_0\rVert/\alpha$" came from the Caputo frozen past, which the field equation does not admit.

---

## 2. Markovian evolution and the light cone

There are two different things one could mean by "Markovian":
- **A Markovian flow:** the state at one time fixes all others.
- **A $\delta$-kernel:** the memory weight tends to $\delta(\tau)$.

The limit $\alpha\to1^-$ is the first, not the second. At $\alpha=1$ the Weyl kernel weights the whole past equally, and differentiating that accumulation gives back $\partial_t$, as in $X(t)-X(t_0)=\int\dot X$. The generator is local in time because accumulation and differentiation are inverse to each other. The bounded solutions form a group for every $\alpha<1$ already. Memory shows up only in the dispersion relation and in responses to sources.

The light cone matters because the CAL integrates over $J^-(x)$, and the kinetic term can only come from the memory reaching out in space. Inside the cone, that reach is at most $c$ times the lag. **Paper Lemma 6.1:** the coefficient of $\nabla^2$ is at most $c^2\langle\tau^2\rangle_w/6$, so it vanishes as $w\to\delta$.

A $\delta$-kernel limit with $r_0\neq0$, as in the skeleton draft, therefore assumes instantaneous spatial reach. Inertia needs memory at nonzero lags. The $\alpha\to1^-$ accumulator supplies those lags while keeping the flow Markovian.

---

## 3. Factorized kernel ($\sigma=-1$, $m=1$, $E_{\rm cap}=4$)

**A. Bounded solutions on the whole time line.** Packet: $k_0=1$, width 2.

| $\varepsilon$ | $\max\lvert E_\alpha-E_1\rvert$ | $\varepsilon\max\Lambda\lvert1+\ln\Lambda\rvert$ | ratio | phase-aligned packet error, $t\le40$ | norm drift |
|---|---|---|---|---|---|
| 1e-1 | 1.246 | 0.954 | 1.305 | 1.41 | 3e-16 |
| 1e-2 | 9.78e-2 | 9.55e-2 | 1.025 | 3.30e-1 | 3e-16 |
| 1e-3 | 9.57e-3 | 9.55e-3 | 1.002 | 3.27e-2 | 3e-16 |
| 3e-4 | 2.87e-3 | 2.86e-3 | 1.001 | 9.80e-3 | 3e-16 |

**B. Source-driven runs.** The integral equation $T=S$ is time-stepped directly: Weyl integration over a zero past, full history kept, a Gaussian pulse at $t_0=10$ of width 2.

*B1. Below the cap ($\Lambda=+1$).* After the pulse, the mode oscillates forever at the residue amplitude.

| $\alpha$ | $h$ | mean $\lvert J\rvert$, $t\in[80,120]$ | prediction | ratio |
|---|---|---|---|---|
| 0.900 | 0.02 | 6.95236 | 6.95787 | 0.99921 |
| 0.900 | 0.01 | 6.95674 | 6.95787 | 0.99984 |
| 0.990 | 0.01 | 5.16659 | 5.16724 | 0.99987 |
| 0.999 | 0.01 | 5.02771 | 5.02833 | 0.99988 |
| 1.000 | 0.01 | 5.01264 | 5.01326 | 0.99988 |
| 1.010 | 0.01 | 4.86571 | 4.86630 | 0.99988 |

*B2. Above the cap ($\Lambda=-1$).* The zero of the response function, compared with the measured late-time rate.

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

## 4. Light-cone kernel ($\sigma=+1$, thin shell $v=1$, $E_0=1$)

- On $E>\hbar kv$ the symbol is real to $7\times10^{-15}$, so every stationary mode is unitary.
- From above ($\alpha=1.001$, $1.01$) there is no stationary root.

| $k$ | root | $kv\coth(kv/E_0)$ | $E_0+k^2/2m$ | relative kinetic deviation | $v_g/v$ |
|---|---|---|---|---|---|
| 0.01 | 1.000033 | 1.000033 | 1.000033 | −6.7e-6 | 0.007 |
| 0.10 | 1.003331 | 1.003331 | 1.003333 | −6.7e-4 | 0.067 |
| 0.30 | 1.029822 | 1.029822 | 1.030000 | −6.0e-3 | 0.198 |
| 1.00 | 1.313035 | 1.313035 | 1.333333 | −6.1e-2 | 0.589 |
| 3.00 | 3.014909 | 3.014909 | 4.000000 | −0.33 | 0.975 |

The approach from below is linear in $\varepsilon$, and $m\to3E_0/(2v^2)=1.5$:

| $\varepsilon$ | $E(0)$ | $m$ fit | $3E(0)/((1+\alpha)v^2)$ | $\max\lvert E_\alpha-E_1\rvert$ | cutoff $kv$ |
|---|---|---|---|---|---|
| 1e-1 | 1.124195 | 1.77507 | 1.77504 | 1.24e-1 | 7.3 |
| 1e-2 | 1.010204 | 1.52296 | 1.52292 | 1.02e-2 | 53 |
| 1e-3 | 1.001002 | 1.50229 | 1.50225 | 1.00e-3 | 504 |
| 1e-4 | 1.000100 | 1.50026 | 1.50023 | 1.00e-4 | 5010 |

---

## 5. Caveats

- **Proof rigor.** The proofs are at the rigor level of the companion papers; the measurability details of the main unitarity proof are not written out.
- **Not Lorentz invariant.** The light-cone dispersion is defined in the frame of the foliation. No relativistic claim is made.
- **Not yet done:** gauge coupling (Hermitian $\hat R$ with $U(1)$ holonomy) and position-dependent potentials beyond a bounded offset.
- **No mass value.** The value of $E_0$, and so the mass scale, is not derived. Only the ratio $E_0/(m v_{\rm eff}^2)=2/3$ is.
