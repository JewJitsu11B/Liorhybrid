# Plan (v2): Schrödinger from the bare CAL at $\alpha\to1^-$, as feedback on the accumulated past

**Status:** plan and outline for rewriting `docs/paper/cal_schrodinger_collapse.tex`.
- The tests are done: `scripts/cal_edge_transducer.py` (numpy only, about 9 s).
- v1 of this plan (prefactor, $r_1$, $\tau_c$, double scaling, $T=0$) is in git history before this commit.

**Goal.** Derive Schrödinger evolution at $\alpha\to1^-$ from the CAL as written, adding only the update rule.

---

## 1. What changed from v1, and why

| v1 input | Status | Reason |
|---|---|---|
| Prefactor $e^{-i\sigma\alpha\pi/2}$ on the memory (Def. 2.3, Axiom 3) | **removed** | The phase belongs to the source. Integrating the source's phase channel turns it by $\alpha$ quarter turns ($\alpha=2\theta/\pi$). A factor in front counts that rotation twice. |
| Coupling $r_1$ (Def. 2.4) | **removed** | Not in the CAL. It stood in for $\Pi P\Phi$, which is the identity for a free carrier in flat space. |
| Time unit $\tau_c$, double scaling at fixed $(1-\alpha)/\tau_c$ (Def. 2.5) | **removed** | Time is relational. Only rates compared with other rates mean anything. Time is counted in updates and quoted in radians of the carrier's own rotation. |
| $T=0$ as the law of motion (Axiom 3) | **removed** | $T$ is the new state. It vanishes only when the present balances the past. |
| "The lossy integrand gives the decay" | **relabelled** | Loss means contraction from rank 2 to rank 1, which is many-to-one. The measured damping comes from the fractional order instead. |

---

## 2. Inputs of the rewritten paper

1. **The CAL (C8):** $T=\alpha J-(1-\alpha)\int_{J^-}k\,\Pi P\Phi\,J$.
2. **The free, flat sector:**
   - $\Pi P\Phi=\mathbb 1$;
   - an order-$\alpha$ power-law kernel, in update units, with weights $w_j=(j^\alpha-(j-1)^\alpha)/\Gamma(\alpha+1)$;
   - a spatial profile $\hat g(k)=e^{-\langle r^2\rangle k^2/6}$, or any normalized isotropic profile.
3. **The update rule (feedback loop):** each output state becomes the next current, $J_{n+1}=T_n$. $T$ and $J$ are both rank-1 currents, so this is type-consistent.
4. **Orientation:** the current's past has rotated one way. This plays the role $\sigma$ played, now as a property of the source, not a factor.

**Not used:** a prefactor, $r_1$, $\tau_c$, $T=0$, or any double scaling.

---

## 3. Established results (numbers from `cal_edge_transducer.py`)

**Update rule and characteristic equation.**
$$
J_{n+1}=\alpha J_n-(1-\alpha)\,\hat g(k)\sum_{j\ge1}w_jJ_{n-j},\qquad
z-\alpha+(1-\alpha)\,\hat g\,W(z)=0,\qquad W(z)=\sum_{j\ge1}w_jz^{-j}.
$$

**Exact memory sum (Jonquière form).** Write $z=e^{i\varphi}$. Then
$$
W=(1-e^{-i\varphi})\sum_{k\in\mathbb Z}\big(i(\varphi+2\pi k)\big)^{-\alpha-1}.
$$
- The sum over $k\neq0$ is evaluated with Hurwitz zeta functions.
- Agreement with direct summation is $10^{-14}$; with mpmath, $3\times10^{-16}$.

**R1. Rotation from feedback.**
- Below 1 the memory term enters with sign $-(1-\alpha)<0$: the accumulated past pulls back.
- The rotation per update is $\varphi\simeq\sqrt{(1-\alpha)\hat g(k)}$.
- With flat weights the result is exact: $z^2-(1+\alpha)z+\alpha+(1-\alpha)\hat g=0$. At $k=0$ both roots lie on $|z|=1$, with $\cos\varphi=(1+\alpha)/2$.
- Measured rotation over $\sqrt{1-\alpha}$: 0.9897, 0.9983, 0.99978, 0.99997, 1.00000 for $1-\alpha=10^{-2},\dots,10^{-7}$.
- Direct iteration matches the roots to every digit.

**R2. Schrödinger dispersion and mass.**
- $\big(\varphi(0)-\varphi(k)\big)/\varphi(0)\to\langle r^2\rangle k^2/12$, with measured ratios 1.0000 to 0.990.
- So $\hbar/2m=\langle r^2\rangle/12$ in rest-radian units, set by the kernel width alone.
- The regime is long wavelengths: the $k^4$ floor halves when $\langle r^2\rangle$ halves.

**R3. Unitary in the limit.**
- Uniform decay is $(\pi/4)(1-\alpha)$ per radian. The measured ratios to $(1-\alpha)$ are 0.901, 0.840, 0.808, 0.795, 0.789, 0.787, tending to $\pi/4=0.785$.
- There is also a diffusive admixture. The extra decay of mode $k$ over its kinetic rotation is $\sqrt{1-\alpha}$, independent of $k$: measured 0.0992 at $1-\alpha=10^{-2}$ and 0.00100 at $10^{-6}$.
- Hence the evolution is $\exp[-(i+\sqrt{1-\alpha})(\hbar/2m)k^2t]$. That is the phase channel plus a vanishing real-channel admixture.
- Asymptotics: $\varphi=\sqrt{\varepsilon\hat g}+\tfrac i2\varepsilon(1-\hat g)+\tfrac{i\pi}4\varepsilon^{3/2}\sqrt{\hat g}+O(\varepsilon^{3/2}\ln\varepsilon)$, with $\varepsilon=1-\alpha$.

**R4. Wavepacket test.** The past is the carrier rotating one way over its whole infinite past, added exactly.
- The error against Schrödinger falls onto the $k^4$ floor: at $1-\alpha=10^{-4}$, $(4.8,2.6,1.5)\times10^{-2}$ against a floor of $(4.8,2.5,1.3)\times10^{-2}$, for $\langle r^2\rangle=0.24,0.12,0.06$.
- Leakage into the opposite rotation is about $10^{-8}$.
- A two-cycle history fails: 70–97% leaks. The nearly flat memory needs the whole past.
- So the wave at a point is the particle's accumulated causal past evaluated there, not a spread in space at one instant.

**R5. Closure and Born readout.**

| $1-\alpha$ | one-way past: trace drift per radian | one-way past: readout drift | zero past: trace swing | zero past: readout drift |
|---|---|---|---|---|
| $10^{-2}$ | $2.1\times10^{-2}$ | $5.5\times10^{-3}$ | 3.5 | $3.7\times10^{-2}$ |
| $10^{-3}$ | $2.8\times10^{-3}$ | $1.9\times10^{-3}$ | 2.2 | $3.6\times10^{-2}$ |
| $10^{-4}$ | $5.5\times10^{-4}$ | $6.0\times10^{-4}$ | 2.0 | $3.7\times10^{-2}$ |

- With a one-way past, the closure trace and the readout become conserved as $\alpha\to1^-$.
- With a zero past, both rotation directions are present. The trace swings and the readout drift stays near $3.7\times10^{-2}$.
- So Born needs one orientation. This is the edge analogue of positive frequency at $\alpha=2$.

**R6. Feedback classification.**

| Feedback from the accumulated past | Behaviour |
|---|---|
| $\alpha<1$: negative | rotates (the Schrödinger edge) |
| $\alpha=1$: none, the past decouples | frozen |
| $\alpha>1$: positive | roots $e^{\pm\sqrt{\alpha-1}}$; causal response runs away, a bounded response must precede its source (ALD's two horns) |
| $\alpha>1$ with braking $\kappa$ and a fading past $\lambda$ | settles if $(1-\kappa)\big(\alpha+(\alpha-1)/\lambda\big)<1$ |

- Without fading, braking cannot stop the growth. This is integral windup.
- Checked at $\alpha=1.01$: settles for $\kappa>0.10$ at $\lambda=0.1$, and for $\kappa>0.50$ at $\lambda=0.01$.

**R7. Normalizable memory has no undamped mode (re-proof of old Thm 6.2 under the update rule).**
- Take $\sum_jw_j=1$, $w_j\ge0$, and $|z|\ge1$.
- Then $|\varepsilon W(z)|\le\varepsilon\le|z|-\alpha\le|z-\alpha|$.
- Equality would need $z=1$, where $F(1)=2\varepsilon\neq0$.
- So every root lies strictly inside the unit circle, and every mode decays.
- The power law escapes because $W(z)$ is unbounded at $z\to1$: its total weight $\sum_jw_j$ is infinite.

**R8. Loss means contraction.**
- In the free edge the rank-2 pieces $\Pi P\Phi$ are the identity, which is one-to-one. So nothing is erased, consistent with unitarity in the limit.
- Contraction becomes many-to-one in two places, and Landauer's $k_BT\ln2$ is paid there:
  - interaction, a rank-2 action on a rank-1 state;
  - readout, where the closure $JJ^\dagger$ sends 8 real components to 4.

---

## 4. Outline of the rewritten paper

**Working title.** *Schrödinger evolution from the bare Causal Accumulation Law at $\alpha\to1^-$: feedback on the accumulated past, unitarity in the limit, and carrier entanglement as time dilation.* Title to be confirmed.

### Abstract
- The CAL plus the update rule alone gives a rotation of $\sqrt{1-\alpha}$ per update and Schrödinger's dispersion, with $\hbar/2m=\langle r^2\rangle/12$.
- Unitarity holds in the limit, with damping $(\pi/4)(1-\alpha)$ per radian and a $\sqrt{1-\alpha}$ diffusive admixture.
- The orientation is inherited from the source.
- Born needs one orientation.
- The feedback classification across $\alpha=1$.
- The carrier and readout results are kept.

### 1. Scope and claim
- A ledger of inputs, what is derived, and what is not claimed.
- What changed from the Skeleton and from v1: the five removals in §1 above.

### 2. Definitions
- **D2.1 Causal Weyl integral** (keep). The symbol turns an oscillation by $\alpha$ quarter turns.
- **D2.2 Memory kernel in update units:** $w_j$ and $\hat g$. No $\tau_c$.
- **D2.3 Carrier and sector:** $J\in\mathbb C\otimes\mathbb H$. Free, flat transport means $\Pi P\Phi=\mathbb 1$.
- **D2.4 Relational time:** update count $n$; rest radians $t=\varphi(0)\,n$.

### 3. Axioms
- **Ax1 (CAL)** keep.
- **Ax2 (phase from the source):** the channel angle comes from integrating the source's phase channel, $\alpha=2\theta/\pi$. It is not a factor on the memory.
- **Ax3 (update rule):** $J_{n+1}=T_n$, a feedback loop of repeated impulses.
- **Ax4 (orientation):** the past current has rotated one way.

### 4. Lemmas
- **L4.1 Weyl symbol** (keep the old Lemma 4.1).
- **L4.2 Exact memory sum** (Jonquière/Hurwitz).
- **L4.3 Characteristic equation and z-transform:** the solution is the residues at the roots plus a branch-cut tail from $z=1$.
- **L4.4 Flat weights, closed form:** unit circle at $k=0$.

### 5. Main results
- **Thm 5.1 Rotation from negative feedback** (R1).
- **Thm 5.2 Dispersion and mass** (R2), with the long-wavelength floor. This replaces the old Prop 5.4 width scaling.
- **Thm 5.3 Unitarity in the limit** (R3), with the evolution law $\exp[-(i+\sqrt{1-\alpha})(\hbar/2m)k^2t]$.
- **Prop 5.4 Orientation from the past** (R4): the one-way past and the zero past.
- **Thm 5.5 Feedback across $\alpha=1$** (R6): frozen at 1; above 1, the two ALD horns; braking and fading. This replaces the old Thm 5.8.

### 6. Markov limits and loss
- **Prop 6.1 FTC accumulator** (keep the old Prop 6.5, strengthened). The nearly flat sum over updates is the discrete integral; the state is not enlarged.
- **Thm 6.2 Normalizable memory has no undamped mode** (R7, re-proved under the update rule). The exponential kernel becomes an example.
- **Remark 6.3 Loss is contraction** (R8): absent in the free edge; present in interaction and readout.

### 7. Structure of the edge
- **§7.1 Galilean, no light cone** (keep).
- **§7.2 No interaction** (keep, sharper): the edge is one order-1 leg, and a return leg makes order 2.
- **§7.3 Readout** (keep the whole old §6.5: two factors, $C=1/\gamma$, composition, $\Lambda^2J=0$). Add R5: Born needs one orientation.

### 8. Numerical verification
New tables from `cal_edge_transducer.py`, sections A–E, plus the braking and fading table.

### 9. Remarks on scope
- **Relation to $\alpha=2$:** the balance reading. The update rule is not used there, because $2J$ alone doubles the current each update.
- **The wave is the accumulated past** (R4). The full one-way history is required; two cycles fail.
  - Smearing with real weights gives diffusion, the $e^{-H/\tau}$ channel. Smearing with phase gives waves, the $e^{iS/\hbar}$ channel.
  - The edge law $\exp[-(i+\sqrt{1-\alpha})(\hbar/2m)k^2t]$ is that mix.
  - Parallel: Feynman (1948) defines $\psi(x,t)$ as the sum over the past parts of paths.
- **Forward pointer only (one sentence): the equivalence condition.**
  - At the edge, inertial mass comes from the kernel width and rest energy from the rotation rate, and nothing ties the two.
  - Requiring width $=\sqrt6$ Compton wavelengths is the statement that inertial mass $=$ rest energy$/c^2$. Gravity coupling universally to energy would need it.
  - The full treatment belongs to the $\alpha=2$ / gravity program (§6, item 5). It is not claimed here.

### Appendices
- Notation.
- Numerical methods.
- The old Appendix C (light cone imposed by hand) is dropped and the α = 2 paper cited instead.

---

## 5. Removed from the current paper

- Definition 2.3 (prefactor).
- $r_1$ in Definition 2.4.
- Definition 2.5 (double scaling).
- Lemma 4.2 (inversion with the prefactor) and Lemma 4.3 (phase cancellation).
- Lemma 4.4 parts on $r_1$.
- Theorems 5.1 and 5.3 as stated; Corollary 5.5; Proposition 5.6 (response to a source with the prefactor).
- Proposition 6.1 (memory rate with $\tau_c$).
- All old numerical tables.
- Appendix C.

---

## 6. Order of work

1. **Proofs.**
   - Asymptotic root expansion R3, including the Euler–Maclaurin $-\tfrac12$ term, which cancels the order-$\varepsilon$ decay at $k=0$.
   - The flat-weight closed form.
   - The z-transform residue and branch-cut statement.
   - R7.
2. **Script.**
   - Add the braking and fading table, and a numerical check of R7, to `cal_edge_transducer.py`.
   - Retire the old `cal_schrodinger_collapse.py`, or keep it for the readout section only.
3. **Rewrite the TeX** to the outline above. Compile and check pages.
4. **Update `SCHRODINGER_ALPHA_TO_ONE_RESULTS.md`.**
5. **Cross-paper follow-ups:**
   - α = 2 paper: remove the channel-phase remark; set the chain to 1; make $\tau_c$ relational; reframe $T=0$ as the balance reading; add the runaway-above-2 table.
   - Posters 3 and 4: step 1, step 2 and row 4 still show the prefactor and exact unitarity for $\alpha<1$.
   - Gravity notes, for a later paper; none of this goes in the edge paper:
     - the $\alpha=2$ light-cone kernel is the retarded propagator of linearized gravity in harmonic gauge, as it is for the Maxwell potential;
     - the rank-2 closure dyads, which satisfy the energy conditions, accumulated on that cone give a linearized-gravity-shaped field;
     - the equivalence condition above, with Weinberg (1964) as the matching result from the gravity side.
     - Open: $G$ as a ratio, $Gm^2/\hbar c=(\ell_P/\text{kernel width})^2$; conservation of the rank-2 source, which by Deser (1970) would then force GR; and nonlinearity.
     - Prior art: Hehl–Mashhoon nonlocal gravity.

## 7. Open, needing the author

- The Compton-width identification. Default: a one-sentence forward pointer in §9 as the equivalence condition, not a result.
- The title.
