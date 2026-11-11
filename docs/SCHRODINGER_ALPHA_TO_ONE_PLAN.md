# Plan: the CAL field equation collapses to Schrödinger as $\alpha\to1^-$

Companion plan to the draft *The $\alpha=1$ Unitary Skeleton and the Emergence of Schrödinger Evolution in the $\delta$-Kernel Limit of the CAL* (`schrodinger_axiomatic.pdf`).

**Goal.** Make the CAL itself, read as a field equation, become $i\hbar\,\partial_t\psi=H\psi$ as $\alpha\to1^-$. This replaces Posit 1 and the identification $H\equiv i\hbar\dot UU^{-1}$ with theorems.

**Kept from the draft:**
- Axioms 1–3;
- Definitions 2.3–2.5 ($\mathcal V$, $\mathcal H$, $f_{\rm map}$);
- Lemma 4.1 (kernel blindness at $\alpha=1$);
- Theorem 5.1 (skeleton);
- Theorem 6.1(ii) (invariance forces $\hat R_{\mathcal H}=r_0\nabla^2+r_1$).

**Notation.** $\varepsilon\equiv1-\alpha$. Modes are written $e^{-iEt/\hbar}$. $E_c\equiv\hbar/\tau_c$. All complex powers take the principal branch.

---

## 1. Definitions added

**D1 (power-law memory, causal Weyl integration).**

$$
W^\alpha f(t)=\frac1{\Gamma(\alpha)}\int_{-\infty}^t(t-s)^{\alpha-1}f(s)\,ds .
$$

$D^\alpha$ is the Weyl derivative, with $D^\alpha W^\alpha=\mathbb 1$ on admissible $f$ (no spectral weight at $E=0$). Symbol: $D^\alpha e^{-iEt/\hbar}=(-iE/\hbar)^\alpha e^{-iEt/\hbar}$.

**D2 (kernel factorization).**
$k(\tau;x,x')=\tau_c^{-\alpha}\,\dfrac{\tau^{\alpha-1}}{\Gamma(\alpha)}\times g(x-x')$, where $g$ is a nonnegative, normalized, isotropic spatial weight, and the transports are taken time-translation invariant.
- On the spinless gauge-neutral sector, invariance gives $\hat R_{\mathcal H}=r_0\nabla^2+r_1$ (draft Thm 6.1(ii)).
- $r_0=\sigma_g^2/6>0$, the second moment of $g$.
- The cone-supported (non-factorized) kernel is treated in §5.

---

## 2. Axioms

- **Axiom 1 (CAL), Axiom 2 (two channels), Axiom 3 (channel angle $\theta=\alpha\pi/2$):** as in the draft, with one change. Make the *kernel and transports* real, not $J$. Solutions are $\propto e^{-iEt/\hbar}$, so $J$ must be $\mathbb C\otimes\mathbb H$-valued as Axiom 2 already says. Lemma 4.3 needs only $\hat R$ real-symmetric.
- **Axiom 3 applied to the memory channel.** The memory channel is rotated by $e^{-i\sigma\theta}$ relative to the present channel. Together with the unit of D2, this gives the factor $(\sigma i\tau_c)^{-\alpha}=\tau_c^{-\alpha}e^{-i\sigma\alpha\pi/2}$, with $\sigma=\pm1$ the orientation.
- **Field equation (FE), replacing Posit 1:**

$$
T=0:\qquad \alpha J=\varepsilon\,(\sigma i\tau_c)^{-\alpha}\,\hat R_{\mathcal H}\,W^\alpha J .
$$

$\partial_tT=0$ gives the same tempered solutions: under Weyl admissibility $T$ is a constant, and that constant is $0$.

---

## 3. Lemmas

**L1 (inversion).**
FE $\iff$ $\alpha\,(\sigma i\tau_c)^\alpha D^\alpha J=\varepsilon\,\hat R_{\mathcal H}J$. *Proof:* apply $D^\alpha$ and use $D^\alpha W^\alpha=\mathbb 1$.

**L2 (phase cancellation, verified numerically).**
$(\sigma i)^\alpha(-iE)^\alpha$ is real and positive iff ${\rm sign}\,E=\sigma$. For the other sign it is $\lvert E\rvert^\alpha e^{\pm i\alpha\pi}$, which is real only at $\alpha=1$. The channel angle exactly cancels the Weyl symbol's phase on one sign of frequency.

**L3 (operator facts; fixes Lemmas 4.3/4.4).**
- $\hat R_{\mathcal H}=r_0\nabla^2+r_1(x)$ is self-adjoint on $H^2(\mathbb R^3)$: $\nabla^2$ is essentially self-adjoint on $C_c^\infty$, and $r_1$ is handled by Kato–Rellich.
- $\lVert\hat R\rVert=\infty$, so draft Lemma 4.4's spectral inclusion is vacuous; use a lower bound instead.
- Commuting the limit with $\Pi_{\mathcal H}$ needs strong-resolvent convergence of $\hat R_w\to\hat R$, not just $\lVert\Pi_{\mathcal H}\rVert=1$.

---

## 4. Theorems

**Theorem 1 (collapse).**
Take $\alpha\to1^-$ with $\varepsilon\hat R_{\mathcal H}$ held fixed. That is a double scaling: $\varepsilon r_0$ and $\varepsilon r_1$ are fixed. Then FE becomes

$$
i\hbar\,\partial_tJ=\sigma\,\varepsilon E_c\,\hat R_{\mathcal H}J
=\Big[-\frac{\hbar^2}{2m}\nabla^2+V\Big]J,
\qquad
\frac{\hbar^2}{2m}=\varepsilon E_c r_0,\quad V=-\varepsilon E_c r_1 .
$$

- **The orientation is fixed by physics.** Positive mass with a nonnegative, lag-independent spread ($r_0>0$) forces $\sigma=-1$. The cone-supported kernel forces $\sigma=+1$ instead (see `SCHRODINGER_ALPHA_TO_ONE_RESULTS.md` §4); the causal kernel decides.
- **The mass formula** is the draft's $\hbar^2/2m=(1-\alpha)r_0$, with $E_c$ restored for units.
- **Without the double scaling** ($r_0$, $r_1$ fixed), $H\to0$: $J$ freezes, which is Thm 5.1. In this route the skeleton is a frozen $J$; it equals the draft's $E_c\,{\rm Id}$ phase up to an unobservable global phase.
- **So Thm 5.1 and Thm 6.1 are two scalings of one limit.** At fixed $r_0$ the past drops out (kernel blindness). At fixed $\varepsilon r_0$ the memory stays live and becomes $H$.

**Theorem 2 (unitarity derived; replaces Posit 1).**
By Fourier analysis in $t$ and the spectral theorem for $\hat R_{\mathcal H}$, using L1 and L2, for every $\alpha<1$:
- **Stationary sector.** On ${\rm Ran}\,\mathbb 1_{\varepsilon\hat R\ge0}$ the solutions are $e^{-itE(\hat R)/\hbar}$ with
  $$E(\hat R)=\sigma E_c\big(\varepsilon\hat R/\alpha\big)^{1/\alpha},$$
  real. By Stone's theorem this is a unitary group with self-adjoint generator $E(\hat R)$.
- **Free case** ($\sigma=-1$, kinetic energy $K=\varepsilon E_cr_0k^2$, cap $E_{\rm cap}=\varepsilon E_cr_1$): the stationary sector is exactly $K\le E_{\rm cap}$, with
  $$E=(K-E_{\rm cap})\big[1+\varepsilon\big(1+\ln\tfrac{E_{\rm cap}-K}{E_c}\big)\big]+O(\varepsilon^2)$$
  (checked against the exact form to $10^{-5}$ at $\varepsilon=10^{-2}$).
- **Above the cap.** *Corrected by the run:* eternal (Weyl) solutions have no component there at all, because the decaying root grows into the past and is not tempered. Switched-on solutions decay there at ${\rm Im}\,E\approx-\pi\varepsilon\lvert E\rvert$ (run B4).
- **The limit.** As $\alpha\to1^-$ the damping goes to $0$, L2's restriction disappears, and the flow becomes the full unitary Schrödinger group of Theorem 1.

**Theorem 3 (why the limit is from below).**
1. **Convergence.** $W^\alpha$ converges on oscillating input only for $\alpha<1$. At $\alpha=1$ its value exists only as the limit from below.
2. **Mass sign.** From above, $\varepsilon<0$, so $\hbar^2/2m=\varepsilon E_cr_0<0$ for any nonnegative kernel: negative mass.
3. **Runaway.** From above, the non-stationary sector *grows*: ${\rm Im}\,E\approx+\pi\lvert\varepsilon\rvert\lvert E\rvert$ (numerically $+0.99$). That is a runaway, the same pathology as the poster's Abraham–Lorentz–Dirac panel, appearing on the wrong side of $\alpha=1$.

---

## 5. Causality estimate (the $J^-(x)$ support)

> **Superseded by the run** (`SCHRODINGER_ALPHA_TO_ONE_RESULTS.md` §4). The estimate below assumed a spread that does not grow with the lag. With a cone-supported kernel the result is different:
> - the exact $\alpha\to1^-$ dispersion is $E=\hbar kv\coth(\hbar kv/E_0)$;
> - $m=3E_0/(2v_{\rm eff}^2)$, so $E_0\le\tfrac23mc^2$;
> - there is no lower bound on $\varepsilon$.
>
> The original estimate is kept below for the record.

The spatial reach of the memory is at most $c$ times the lag. So $r_0\lesssim(c\tau_{\rm eff})^2/6$, and with $\hbar^2/2m=\varepsilon E_cr_0$:

$$
\varepsilon\;\gtrsim\;\frac{3\,t_C}{\tau_{\rm eff}},\qquad t_C=\frac{\hbar}{mc^2}\quad(\text{electron: }1.3\times10^{-21}\,{\rm s}).
$$

- **Consequence.** $\alpha\to1^-$ at fixed $m$ requires $\tau_{\rm eff}/t_C\to\infty$, so the Schrödinger limit is jointly a nonrelativistic limit.
- **Consequence.** $E_c\ll mc^2$, so the skeleton scale is not the rest energy.
- **Deliverable.** Redo the estimate with a cone-supported (non-factorized) kernel. The $k^2$ coefficient then acquires frequency dependence; show it reduces to D2 as $c\to\infty$.

---

## 6. Consistency checks

- **The two $i$'s agree.** Axiom 3's $i$ must be Axiom 2's central $i$ of $\mathbb C\otimes\mathbb H$. On the projected spinor sector, the only complex structures commuting with Lorentz are $\pm i$ (commutant $\cong\mathbb C$, checked numerically), so they agree up to $\sigma$.
- **Equivalence of balance laws.** Under Weyl admissibility, $T=0$ and $\partial_tT=0$ have the same tempered solutions; the dispersion relations are identical.
- **Weyl admissibility at the cap.** At the cap, $E\to0$, so admissibility fails there; the cap is a genuine boundary.
- **Constant shifts.** Eq. (11)'s constant shift is inert only at $\alpha=1$. For $\alpha<1$ the zero of energy is physical, because it sets the cap.

---

## 7. Changes to the draft

| Draft item | Change |
|---|---|
| Posit 1, Thm 6.1(i) | Replace with FE plus Theorems 1–2. As written, $H\equiv i\hbar\dot UU^{-1}$ after positing a unitary $U$ is a definition, so the derivation is circular. |
| Axiom 1 "J real" | Say "kernel and transports real". $J$ is complex by Axiom 2 and by the solutions. |
| Lemma 4.3 | Real plus symmetric gives *symmetric*. Self-adjointness needs L3. |
| Lemma 4.4 | $\lVert\hat R\rVert=\infty$. The limit/projection exchange needs strong-resolvent convergence. |
| Thm 6.1(iii), Cor 6.3 | $m=\hbar^2/(2(1-\alpha)r_0)\to\infty$ at fixed $r_0$. State the double scaling. |
| Thm 5.1 | $H={\rm Id}$ must carry units: $E_c\,{\rm Id}$, with phase $e^{-iE_ct/\hbar}$. |
| Remark 5.2 / Table 7.4 | The "$1^-$ edge" is the double-scaled limit, and "from below" is a theorem (Thm 3), not a convention. |

---

## 8. Order of work

1. D1, D2, FE, and L1–L3.
2. Theorem 2: the Fourier/spectral proof, including the distributional details at the cap.
3. Theorem 1, using Trotter–Kato for strong convergence of the groups.
4. Theorem 3: three short lemmas.
5. §5 with a cone-supported kernel.
6. §6 checks.
7. Gauge coupling: transports with $U(1)$ holonomy make $\hat R$ Hermitian rather than real. Redo L3 with the diamagnetic inequality; Theorems 1–2 only need Hermitian and bounded below on the relevant sector.

---

## References

- S. G. Samko, A. A. Kilbas, O. I. Marichev, *Fractional Integrals and Derivatives* (1993): Weyl/Liouville integrals, admissibility.
- M. Naber, *Time fractional Schrödinger equation*, J. Math. Phys. **45**, 3339 (2004): the $i^\alpha$ factor; here it comes from Axiom 3.
- E. Nelson, *Feynman integrals and the Schrödinger equation*, J. Math. Phys. **5**, 332 (1964): boundary values of rotated semigroups.
- T. Kato, *Perturbation Theory for Linear Operators* (holomorphic semigroups; Kato–Rellich).
- M. Reed, B. Simon, *Methods of Modern Mathematical Physics* I–II (Stone; Trotter–Kato).
- M. H. Stone, Ann. of Math. **33**, 643 (1932).
