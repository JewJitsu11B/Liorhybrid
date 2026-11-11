# Results: the CAL field equation collapses to Schrödinger as $\alpha\to1^-$

This document reports a run of the plan in `docs/SCHRODINGER_ALPHA_TO_ONE_PLAN.md`.

**Reproduce.** `python3 scripts/cal_schrodinger_collapse.py` (numpy only, about 1 minute). Units: $\hbar=\tau_c=1$ (so $E_c=1$), and $c=1$ in §4.

**The field equation that was run.**

$$
T=0:\quad \alpha J=\varepsilon\,(\sigma i\tau_c)^{-\alpha}\,\mathcal M_\alpha[J]
\;\Longleftrightarrow\;
\alpha(\sigma i\tau_c)^\alpha D^\alpha J=\Lambda J\quad(\text{factorized kernel}),
$$

with $\varepsilon=1-\alpha$. The two models are:
- **Factorized kernel:** power-law time memory with causal Weyl integration, times a spatial spread that does not depend on the lag. Then $\Lambda=\varepsilon\hat R_{\mathcal H}$, held fixed in the double scaling.
- **Cone-supported kernel:** the spatial reach at lag $\tau$ is $v\tau\le c\tau$, so the memory stays inside $J^-(x)$.

---

## 1. Summary

1. **Collapse confirmed.** Eternal solutions and switched-on solutions both converge to Schrödinger, linearly in $\varepsilon$.
2. **Unitarity is derived, not posited.** Eternal solutions conserve the norm exactly for every $\alpha<1$; the measured drift is $3\times10^{-16}$.
3. **The limit must be from below.**
   - A mode above the cap decays at $\pi\varepsilon\lvert E\rvert$ from below.
   - At $\alpha=1$ it is unitary.
   - From above it grows at the same rate. With the cone kernel, there is no stationary solution at all from above.
4. **New result: the cone-supported kernel.** Approached as $\alpha\to1^-$, it gives an exact dispersion relation
   $$E(k)=\hbar kv\,\coth\!\big(\hbar kv/E_0\big).$$
   - Schrödinger is its limit for $K\ll E_0$, with $m=3E_0/(2v^2)$.
   - The group velocity stays below $v\le c$ for every $k$.
5. **Three corrections to the plan:**
   - The factorized estimate $\varepsilon\gtrsim3t_C/\tau_{\rm eff}$ was an artifact of a spread that ignores the lag. The cone kernel gives no lower bound on $\varepsilon$; instead it ties the mass to the offset: $E_0=\tfrac23mv_{\rm eff}^2\le\tfrac23mc^2$.
   - Eternal solutions have *no* component above the cap. The decaying root grows into the past, so it is not tempered. The decay applies only to switched-on solutions.
   - Switched-on solutions end at norm $\lVert J_0\rVert/\alpha$ rather than $\lVert J_0\rVert$. This was not in the plan.
6. **Orientation.** A lag-independent spread needs $\sigma=-1$ for positive mass; the cone kernel needs $\sigma=+1$. The causal kernel therefore fixes $\sigma=+1$.

---

## 2. Proofs

**Setting.**
- $\Lambda=\varepsilon\hat R_{\mathcal H}$ is self-adjoint (plan L3).
- $f_\alpha(\lambda)\equiv\sigma E_c(\lambda/\alpha)^{1/\alpha}$ for $\lambda\ge0$.
- $P_+\equiv\mathbb 1_{[0,\infty)}(\Lambda)$, the spectral projection onto $\Lambda\ge0$.
- $g_\alpha(E)\equiv\alpha(\sigma i)^\alpha(-iE\tau_c/\hbar)^\alpha$, the symbol of the left side on $e^{-iEt/\hbar}$.

**L2 (phase cancellation).**
The principal arguments are $\arg(\sigma i)=\sigma\pi/2$ and $\arg(-iE)=-{\rm sign}(E)\,\pi/2$. So
$$\arg g_\alpha=\tfrac{\alpha\pi}{2}\big(\sigma-{\rm sign}\,E\big)\in\{0,\pm\alpha\pi\}.$$
For $0<\alpha<1$, $g_\alpha$ is real only when ${\rm sign}\,E=\sigma$. There it is positive and strictly increasing in $\lvert E\rvert$, running from $0$ to $\infty$. $\square$

**Theorem 2 (eternal solutions; replaces Posit 1).**
*Statement.* Let $J\in C_b(\mathbb R;\mathcal H)$ be bounded, contain no zero frequency, and solve the field equation as a tempered distribution. Then
$$J(t)=e^{-itf_\alpha(\Lambda)/\hbar}P_+J(0),\qquad(1-P_+)J\equiv0 .$$
*Proof.*
1. Fourier-transform in $t$ and decompose spectrally in $\Lambda$. Fiberwise this gives $(g_\alpha(E)-\lambda)\hat J_\lambda=0$.
2. By L2, for $\lambda>0$ the function $g_\alpha-\lambda$ vanishes only at the simple zero $E_\lambda=f_\alpha(\lambda)$. For $\lambda<0$ it never vanishes. At $\lambda=0$ the only zero is $E=0$, which admissibility excludes.
3. A distribution annihilated by such a function is supported on the zero set. So $\hat J_\lambda$ is a sum of $\delta^{(n)}(E-E_\lambda)$.
4. Boundedness in $t$ rules out $n\ge1$, since those terms grow like $t^n$. Hence $J_\lambda(t)=c_\lambda e^{-iE_\lambda t/\hbar}$ for $\lambda>0$, and $J_\lambda=0$ for $\lambda<0$.
5. The family $t\mapsto e^{-itf_\alpha(\Lambda)}$ is a strongly continuous unitary group by functional calculus. Its generator $f_\alpha(\Lambda)$ is self-adjoint (Stone). $\square$

*Rigor note.* Steps 3–4 are fiberwise statements about operator-valued distributions, done here at physics rigor.

**Theorem 1 (collapse).**
For $\psi\in{\rm Ran}\,P_+$:
$$\big\lVert(e^{-itf_\alpha(\Lambda)}-e^{-itf_1(\Lambda)})\psi\big\rVert\le\frac{\lvert t\rvert}{\hbar}\,\lVert(f_\alpha-f_1)(\Lambda)\psi\rVert,$$
where
$$\lvert f_\alpha(\lambda)-f_1(\lambda)\rvert=E_c\,\varepsilon\,\lambda\,\lvert1+\ln\lambda\rvert+O(\varepsilon^2).$$
- The run checks the coefficient: the ratio is $1.001$ at $\varepsilon=3\times10^{-4}$.
- Dominated convergence then gives strong convergence for every $\psi\in{\rm Ran}\,P_+$, uniformly on compact time intervals.
- The limit equation is $\sigma i\hbar\,\partial_tJ=E_c\Lambda J$. That is Schrödinger with $\hbar^2/2m=-\sigma\varepsilon E_c r_0$, the draft's mass formula with $E_c$ restored. $\square$

**Switched-on solutions (Caputo form).**
The solution is $J(t)=E_\alpha(zt^\alpha)J_0$ with $z=\Lambda/(\alpha(\sigma i\tau_c)^\alpha)$. It splits as
$$E_\alpha(zt^\alpha)=\frac1\alpha e^{s_*t}+\int_0^\infty e^{-rt}K_z(r)\,dr ,$$
the pole at $s_*=z^{1/\alpha}$ plus a branch-cut term. The cut term is $O(\varepsilon\,t^{-\alpha})$.

- **Below the cap ($\Lambda>0$):** $s_*$ lies on the imaginary axis, so $\lVert J(t)\rVert\to\lVert J_0\rVert/\alpha$. This matches run B3 to six digits.
- **Above the cap:** ${\rm Re}\,s_*=\lvert s_*\rvert\sin(\pi/\alpha)$. That is $\approx-\pi\varepsilon\lvert E\rvert/\hbar$ (decay) for $\alpha<1$ and $\approx+\pi\lvert\varepsilon\rvert\lvert E\rvert/\hbar$ (growth) for $\alpha>1$ (run B4).

**Theorem 3 (why from below).**
- **(a) Convergence.** $\int_0^\infty s^{\alpha-1}e^{iEs}\,ds$ converges by the Dirichlet test iff $0<\alpha<1$. At $\alpha=1$ its value is defined only as the limit from below.
- **(b) Mass sign.** From above, $\varepsilon<0$ flips the sign of the memory-generated kinetic term (factorized kernel). With the cone kernel and a positive zeroth moment, $E_0<0$, and the run finds no stationary root.
- **(c) Runaway.** From above, the non-stationary sector grows at $\pi\lvert\varepsilon\rvert\lvert E\rvert/\hbar$ (run B4). $\square$

---

## 3. Run output: factorized kernel ($\sigma=-1$, $m=1$, $E_{\rm cap}=4$)

**A. Eternal solutions.** A Gaussian packet with $k_0=1$ and width 2; its probability above the cap is $10^{-13}$, so it is dropped.

| $\varepsilon$ | $\max\lvert E_\alpha-E_1\rvert$ | $\varepsilon\max\Lambda\lvert1+\ln\Lambda\rvert$ | ratio | packet error, $t\le40$ (phase-aligned) | /$\varepsilon$ | norm drift |
|---|---|---|---|---|---|---|
| 1e-1 | 1.246 | 0.954 | 1.305 | 1.408 | 14.1 | 3e-16 |
| 1e-2 | 9.78e-2 | 9.55e-2 | 1.025 | 3.30e-1 | 33.0 | 3e-16 |
| 1e-3 | 9.57e-3 | 9.55e-3 | 1.002 | 3.27e-2 | 32.7 | 3e-16 |
| 3e-4 | 2.87e-3 | 2.86e-3 | 1.001 | 9.80e-3 | 32.7 | 3e-16 |

The raw packet error is dominated by a global phase drift from the constant term. With that phase removed, the error is linear in $\varepsilon$.

**B1.** The Mittag-Leffler evaluator agrees with the power series to $3.5\times10^{-12}$.

**B2. Time-stepping the memory equation with its full history** (BDF2 convolution quadrature), compared with the exact response:

| $\alpha$ | $\Lambda$ | $h=0.02$ | $h=0.01$ | $h=0.005$ |
|---|---|---|---|---|
| 0.90 | +1 | 1.67e-2 | 7.30e-3 | 3.39e-3 |
| 0.99 | +1 | 1.30e-2 | 5.80e-3 | 2.72e-3 |
| 0.99 | −1 | 9.99e-3 | 4.96e-3 | 2.47e-3 |

The error halves with $h$, which is first order. That is the expected rate for the $t^\alpha$ start-up singularity when no correction weights are used. The stepped equation converges to the closed form.

**B3. Switched-on packet.**

| $\varepsilon$ | phase-aligned error, $t\le40$ | /$\varepsilon$ | norm at $t=40$ | $1/\alpha$ |
|---|---|---|---|---|
| 1e-1 | 1.49 | 14.9 | 1.111111 | 1.111111 |
| 1e-2 | 3.32e-1 | 33.2 | 1.010101 | 1.010101 |
| 1e-3 | 3.28e-2 | 32.8 | 1.001001 | 1.001001 |

**B4. A single mode above the cap** ($\Lambda=-1$, $\lvert E\rvert=1$).

| $\alpha$ | pole ${\rm Re}\,s$ | $\pi\lvert\varepsilon\rvert\lvert E\rvert$ | $\lvert J(10)\rvert$ | $\lvert J(40)\rvert$ | $\lvert J(80)\rvert$ |
|---|---|---|---|---|---|
| 0.990 | −3.21e-2 | 3.14e-2 | 0.733 | 0.280 | 0.078 |
| 0.999 | −3.15e-3 | 3.14e-3 | 0.970 | 0.883 | 0.778 |
| 1.000 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 1.001 | +3.14e-3 | 3.14e-3 | 2.083 | 2.281 | 2.573 |
| 1.010 | +3.08e-2 | 3.14e-2 | 2.901 | 6.885 | 23.53 |

---

## 4. Cone-supported kernel ($\sigma=+1$, $E_0=\varepsilon E_c\tilde r_1$ fixed, $v\le c$)

**Symbol.** A thin shell of radius $v\tau$ at lag $\tau$ has $\hat g_\tau(k)=\sin(kv\tau)/(kv\tau)$. The memory symbol is then

$$
\mathcal M_\alpha(s,k)=\tau_c^{-\alpha}\,\frac{(s-ikv)^{1-\alpha}-(s+ikv)^{1-\alpha}}{2ikv(\alpha-1)},\qquad s=-iE/\hbar .
$$

On $E>\hbar kv$ the channel phase cancels the symbol's phase exactly. The measured $\lvert{\rm Im}\rvert/\lvert\cdot\rvert$ is $\le7\times10^{-15}$ for $\alpha\in\{0.7,\,0.9,\,0.99,\,1\}$. So every mode that is stationary is unitary.

**At $\alpha\to1^-$.** The equation reduces to $\operatorname{artanh}(\hbar kv/E)=\hbar kv/E_0$, so

$$
E(k)=\hbar kv\coth\frac{\hbar kv}{E_0}=E_0+\frac{(\hbar kv)^2}{3E_0}-\frac{(\hbar kv)^4}{45E_0^3}+\dots,
\qquad m=\frac{3E_0}{2v^2}.
$$

The group velocity is $v_g=v(\coth y-y\,{\rm csch}^2y)<v$ for all $y=\hbar kv/E_0$. For a general isotropic profile, the $O(k^2)$ term is the same with $v^2\to v_{\rm eff}^2\equiv\langle r^2\rangle/\tau^2\le c^2$.

| $k$ | root | $kv\coth(kv/E_0)$ | $E_0+k^2/2m$ | relative kinetic deviation | $K/E_0$ | $v_g/v$ |
|---|---|---|---|---|---|---|
| 0.01 | 1.000033 | 1.000033 | 1.000033 | −6.7e-6 | 0.000 | 0.007 |
| 0.10 | 1.003331 | 1.003331 | 1.003333 | −6.7e-4 | 0.003 | 0.067 |
| 0.30 | 1.029822 | 1.029822 | 1.030000 | −6.0e-3 | 0.030 | 0.198 |
| 1.00 | 1.313035 | 1.313035 | 1.333333 | −6.1e-2 | 0.333 | 0.589 |
| 3.00 | 3.014909 | 3.014909 | 4.000000 | −0.33 | 3.000 | 0.975 |

**For $\alpha<1$.**
- The ground energy is $E(0)=E_c(E_0/\alpha E_c)^{1/\alpha}$.
- The effective mass is $m_\alpha=3E(0)/((1+\alpha)v^2)$; the fit agrees to $2\times10^{-5}$ relative.
- There is a UV cap at $\hbar k_{\rm cap}v\approx\tfrac12E_c\big(E_0/(\alpha\varepsilon E_c)\big)^{1/\alpha}$, which recedes as $\varepsilon\to0$.

| $\varepsilon$ | $E(0)$ | $(E_0/\alpha)^{1/\alpha}$ | $m$ fit | $3E_*/((1+\alpha)v^2)$ | $\max\lvert E_\alpha-E_1\rvert$, $k\le1$ | cap $kv$ |
|---|---|---|---|---|---|---|
| 1e-1 | 1.124195 | 1.124195 | 1.77507 | 1.77504 | 1.24e-1 | 7.3 |
| 1e-2 | 1.010204 | 1.010204 | 1.52296 | 1.52292 | 1.02e-2 | 53 |
| 1e-3 | 1.001002 | 1.001002 | 1.50229 | 1.50225 | 1.00e-3 | 504 |
| 1e-4 | 1.000100 | 1.000100 | 1.50026 | 1.50023 | 1.00e-4 | 5010 |

**Physical consequences:**
- **The $k=0$ energy is bounded.** Memory that respects the light cone gives $E_0=\tfrac23mv_{\rm eff}^2\le\tfrac23mc^2$, so it cannot be the rest energy $mc^2$. That is a constraint the framework has to live with, or a prediction.
- **Schrödinger is the slow-particle regime.** The corrections are $\sim K/E_0\sim(u/v_{\rm eff})^2$, where $u$ is the particle speed. Schrödinger holds for $u\ll v_{\rm eff}$.
- **The mass is not free.** It is fixed by the zeroth moment ($E_0$) and by how much of the light cone the memory fills ($v_{\rm eff}$). This sharpens the draft's "no mass scale" statement: the scale is still open, but the ratio $E_0/mv_{\rm eff}^2=2/3$ is fixed.

---

## 5. Caveats

- **The factorized kernel violates causality at short lags.** Its spread does not shrink with the lag. §3 keeps it for continuity with the plan; the physical statements are in §4.
- **The cone dispersion is not Lorentz invariant.** The shell is defined in the foliation frame, so no relativistic claim is made.
- **Proof rigor.** The proofs are at physics rigor; the distributional steps of Theorem 2 still need writing out in full.
- **Not yet run:** the gauge sector (Hermitian $\hat R$ with $U(1)$ holonomy) and position-dependent potentials.
- **The $1/\alpha$ norm depends on the switch-on model.** It comes from the Caputo form, which holds the initial value. A source-driven switch-on could end differently.
- **Time-stepping accuracy.** The BDF2 time-stepper is first order here; correction weights would restore second order.
