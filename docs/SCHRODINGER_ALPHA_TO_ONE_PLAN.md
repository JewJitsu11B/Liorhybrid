# Plan: Schrödinger's equation from the Causal Accumulation Law as $\alpha_{\rm frac}\to 1^-$

**Target.** Show that the CAL plus one stated dynamical principle gives

$$
i\hbar\,\partial_t\psi=\Big[\frac{(\mathbf p-q\mathbf A)^2}{2m}+qV\Big]\psi
$$

in a stated scaling window of $\alpha_{\rm frac}\to1^-$. Every ingredient (first-order time derivative, linearity, the $i$, unitarity, $m$, $V$, $\mathbf A$) has to be traced to a named piece of the CAL. Nothing equivalent to the conclusion may be assumed, so there is no $U=e^{i\theta G}$ ansatz.

**Notation.** $\alpha\equiv\alpha_{\rm frac}$, $\varepsilon\equiv1-\alpha$. $I^\alpha$ is the Riemann–Liouville integral, $D^\beta$ the RL derivative, and $\iota\equiv e_1e_2e_3$ the Cl(3) pseudoscalar. In schematic form the CAL is

$$
T=\alpha J-(1-\alpha)\,\mathcal K_\alpha[J],\qquad
\mathcal K_\alpha[J](x)=\int_{J^-(x)}k\;\Pi\;PPPP\;\Phi\;P\;J\;d^4x'.
$$

---

## 0. Three facts that fix the shape of the derivation

**F1. With the poster's convention, $\alpha\to1^-$ does not send $k\to\delta$.**
"$\alpha_{\rm frac}=3$ gives $(t-\tau)^2/2$" fixes $k_\alpha(\tau)=\tau^{\alpha-1}/\Gamma(\alpha)$, the RL kernel; Eq. (9) $\to 3J+2J^-$ checks. As $\alpha\to1^-$ this tends to $\Theta(\tau)$, the plain FTC accumulator. The $\delta$-collapse belongs to the other end: $\tau^{\alpha-1}/\Gamma(\alpha)\approx\alpha\,\tau^{\alpha-1}\to\delta$ as $\alpha\to0^+$. Prop. 1 in `proof_hierarchy2` uses $w\to\delta$, so it has to be re-derived, not just re-labelled.

*Aside.* A $\delta$ does appear at $\alpha\to1^-$ if the kernel is $(t-\tau)^{-\alpha}$. Then $(1-\alpha)\tau^{-\alpha}$ has unit mass on $[0,1]$ and concentrates at $0$. That kernel is the Caputo kernel of order $1-\alpha$, though, and it contradicts the $\alpha=3$ statement. Switch to it only if you drop that statement.

**F2. At $\alpha=1$ exactly, $T=J$.**
$\Pi$, every $P$, $\Phi$ and $k$ sit inside the $(1-\alpha)$ term. So the mass, the potential and the memory corrections are all $O(\varepsilon)$. The gauge field also enters through the covariant $\nabla$ in the dynamical principle (C2 below). The limit is therefore singular: the target lives in a window $\varepsilon\to0$ with $\varepsilon\,\Phi_0$ held fixed ($\Phi_0$ is the coincidence strength of $\Pi\Phi$). This has the same structure as the van Hove / Davies weak-coupling limit, $\lambda\to0$ with $\lambda^2t$ fixed. **The "$1^-$" is essential.** At $\alpha=1$ nothing massive survives, and a massless field has no Schrödinger limit.

**F3. The CAL is a constitutive law, not an equation of motion.**
It defines $T$ from $J$. Getting dynamics takes one more principle, stated as an axiom. The recommended one is $\nabla_\mu T^{\mu\nu a}{}_{\rho\sigma}=0$, the same divergence-free principle the poster already uses for Lovelock.

---

## 1. Phase 0: definitions (one page; every later step cites it)

| Choice | Recommended | Reason |
|---|---|---|
| C1 kernel | $k_\alpha=\tau^{\alpha-1}/\Gamma(\alpha)$, with $\tau$ an invariant (proper-time / interval) argument | Forced by the $\alpha=3$ statement; invariance is needed for gate K4 |
| C2 dynamics | $\nabla_\mu T^{\mu\cdots}=0$, with $\nabla$ built from the same connection whose parallel transport is $P$ | Covariant; matches the Lovelock panel; supplies minimal coupling at $O(\varepsilon^0)$ |
| C3 sector ansatz | the $\mu$ index of $J$ rides on the paravector basis, $J^\mu\leftrightarrow\bar e^{\mu}J$ with $\bar e^\mu=(1,-e_k)$, so $\partial_\mu J^\mu=\bar\partial J$, $\bar\partial\equiv\partial_{ct}-e_k\partial_k$; the $\nu\rho\sigma$ indices ride along as a fixed tensor | Without an explicit sector ansatz there is nothing for C2 to act on |
| C4 carrier algebra | $J=\sum_a Z_ae_a\in M_2(\mathbb C)\cong Cl(3,0)$, $e_0=1$, $e_k=\sigma_k$ | Born: $\tfrac12{\rm tr}\,JJ^\dagger=\sum_a\lvert Z_a\rvert^2$. Lorentz: $\det J=Z_0^2-\mathbf Z\cdot\mathbf Z$ |
| C5 Hilbert projection | $\psi=JP_+$, $P_+=\tfrac12(1+e_3)$ | Makes $f_{\rm map}$ explicit and linear: the minimal left ideal is the 2-component Pauli spinor |
| C6 past boundary | the full $J^-(x)$, from $-\infty$ | What the CAL writes. The FTC-from-$t_0$ version adds transients, handled in 3.3 |
| C7 spatial reach | memory range in $\lvert\mathbf x-\mathbf x'\rvert$ that does not grow with $\tau$, or a covariant kernel treated in momentum space | If the range grows like $c\tau$ in a time-sliced treatment, the $\nabla^2$ term picks up $\tau^2$ and becomes $I^{\alpha+2}$, which stays nonlocal even at $\alpha=1$ |

**The complex unit.** $\iota=e_1e_2e_3$ is central in $Cl(3,0)$ and $\iota^2=-1$. In $M_2(\mathbb C)$ this is just $\sigma_1\sigma_2\sigma_3=i\mathbb 1$. Because the $Z_a$ are complex, $\mathbb C$ is already present. The derivation has to show that the *generator* is $\iota\times$(Hermitian). Showing that complex numbers are around is not enough.

---

## 2. Phase 1: leading order $\varepsilon^0$ (the "1" in $1^-$)

$T=J$ together with C2 and C3 gives $\bar\partial J=0$ (minimally coupled through C2's $\nabla$), and hence $\Box J=0$, since $\partial\bar\partial=\partial_{ct}^2-\nabla^2$. This is the APS form of the Weyl equation.

**Deliverables.**
- The equation is first order in time.
- The equation is linear.
- $\tfrac12{\rm tr}\,JJ^\dagger$ is the time component of a conserved current at this order.
- There is no mass, which is why the derivation cannot stop at $\alpha=1$.

---

## 3. Phase 2: order $\varepsilon$, what the memory term supplies

**2.1 Transports.**
Expand about coincidence: $P(x,x')=1-(x'-x)^\mu A_\mu+O(\Delta x^2)$. The poster's functoriality, $P_{\gamma_2\circ\gamma_1}=P_{\gamma_2}P_{\gamma_1}$, is what makes the $O(\varepsilon)$ term gauge covariant. Check that the connection in $P$ and the one in C2's $\nabla$ are the same.

**2.2 $\Phi$ and $\Pi$ at coincidence.**
Split the coincidence action on the carrier into three pieces:
1. The part that maps $J$ to a conjugate (the Clifford-bar/dagger type that couples the two chiralities). This is the Dirac mass.
2. The part that commutes with the carrier. This is a scalar potential.
3. Any part that is non-Hermitian and not proportional to $\iota$. This is dissipation, and it must vanish or be $O(\varepsilon)$ (gate K1).

**2.3 Time memory and localization.**
As $\alpha\to1$, $k\to\Theta$, so the $O(\varepsilon)$ term is formally nonlocal. Taken from $-\infty$, it is IR-divergent on zero-frequency components. On a carrier with a fast phase, $J=e^{-\iota\omega t}\phi$ with $\phi$ slow, the Laplace symbol $\int_0^\infty k_\alpha e^{-s\tau}d\tau=s^{-\alpha}$ gives

$$
\int_0^\infty k_\alpha(\tau)\,e^{-\iota\omega(t-\tau)}\phi(t-\tau)\,d\tau
=e^{-\iota\omega t}\Big[(-\iota\omega)^{-\alpha}\phi(t)-\alpha(-\iota\omega)^{-\alpha-1}\dot\phi(t)+\dots\Big].
$$

Each term is local in time: the fast phase localizes the history. At $\alpha=1$, $(-\iota\omega)^{-1}=\iota/\omega$.

- **The accumulated past of an oscillating carrier is local and carries $\iota$.** This is a *dynamical* source of the $i$. It must agree with the algebraic one from 2.2 (gate K2).
- The expansion parameter is $\lvert\dot\phi\rvert/(\omega\lvert\phi\rvert)$, i.e. kinetic energy over rest energy. So the Markov reduction and the nonrelativistic reduction run on the same small parameter, plus $\varepsilon$.
- Covariant version: in momentum space the memory term is $\varepsilon\Phi_0\,\tilde k_\alpha(p)\,J(p)$, evaluated on shell. Locality in the NR sense means $\tilde k_\alpha$ is smooth near the mass shell.

**2.4 Gap equation (mass from memory).**
The mass coefficient from 2.2 and 2.3 depends on $\omega$ through $(-\iota\omega)^{-\alpha}$, or covariantly through $\tilde k_\alpha(p^2)$. So $m$ is fixed self-consistently: the determinant of the full symbol (leading $\bar\partial$ plus memory) vanishes. That is the poster's "Lorentz by determinant" closure, made dynamical. The expected form is $m^2\propto\varepsilon\Phi_0\,\tilde k_\alpha(m^2)$; the exact form is a deliverable.

**2.5 Scaling window and error terms.**
Take $\varepsilon\to0$ with $\varepsilon\Phi_0$ fixed. The errors are $O(\varepsilon\ln\varepsilon)$ and $O(\lvert\dot\phi\rvert/\omega\lvert\phi\rvert)$. Davies (1974) is the template for making this rigorous.

---

## 4. Phase 3: why $\alpha=1$ is the Markov point, and unitarity

**3.1 Composition.**
$E_\alpha(\lambda(t+s)^\alpha)=E_\alpha(\lambda t^\alpha)\,E_\alpha(\lambda s^\alpha)$ holds for all $t,s\ge0$ **iff $\alpha=1$** (Peng & Li 2010). Lift this to operators with the spectral theorem.
- **Claim to prove:** the poster's history category (paths compose for every $\alpha$) descends to a one-parameter group on endpoint states only at $\alpha=1$.
- This is the precise form of "the endpoint forgets". It replaces "$w\to\delta$" as the reason the limit is Markovian.

**3.2 Born conservation forces an anti-Hermitian generator.**
$\frac{d}{dt}\int\tfrac12{\rm tr}(\psi\psi^\dagger)\,d^3x=0$ for all $\psi$ $\iff$ $\mathcal G+\mathcal G^\dagger=0$ in the trace inner product. Write $\mathcal G=-\iota H/\hbar$ with $H$ Hermitian. Then $\iota$ has to commute with the observables. Either it is central, or it acts on the right with observables acting on the left, as in Hestenes's $\iota e_3$ form; both are acceptable.

**3.3 Stone, self-adjointness and transients.**
- Stone's theorem gives $U(t)=e^{-\iota Ht/\hbar}$.
- Here $\hbar$ only converts units. What is derivable is $\hbar/m$, or equivalently $\omega_C$.
- Self-adjointness of $H$ (not just symmetry) needs Kato–Rellich for $V$.
- If C6 uses a finite $t_0$, the boundary term oscillates at $\omega$ relative to the envelope, so averaging (Krylov–Bogoliubov) removes it. At $\alpha<1$ it decays as $(t-t_0)^{\alpha-1}$; that is the Mittag-Leffler tail.

**3.4 Order of limits and projection.**
Linearity of $f_{\rm map}$ is not enough to exchange the limit with the projection. You need a uniform-in-$\varepsilon$ bound on $\psi_\varepsilon$ together with a bounded $f_{\rm map}$, i.e. dominated convergence. Redo the exchange under the $\varepsilon\Phi_0$-fixed scaling and under $c\to\infty$. Then either show that $\varepsilon\to0$ and $c\to\infty$ commute, or state which order is used.

---

## 5. Phase 4: NR reduction and identification

Use the standard Dirac → Pauli → Schrödinger reduction in APS form (Baylis; Hestenes) with $J=e^{-\iota\omega_Ct}\phi$, keeping leading order in $\lvert\dot\phi\rvert/\omega_C\lvert\phi\rvert$ and $v/c$.

| Quantity | Source |
|---|---|
| $m=\hbar\omega_C/c^2$ | gap equation (2.4) |
| $\mathbf A$, $V$ | connection in C2 and $P$ |
| extra scalar potential | the commuting part of $\Phi$ (2.2) |
| $-(q\hbar/2m)\,\boldsymbol\sigma\cdot\mathbf B$ | appears automatically (Pauli); scalar Schrödinger is the spin-independent sector |
| the $-\alpha(-\iota\omega)^{-\alpha-1}\dot\phi$ term | renormalizes the coefficient of $\partial_t$ (a $Z$ factor). In the scaling window it is $O(\varepsilon\Phi_0/\omega_C^2)=O(1)$, **not small** |

**K4 (inertial mass = rest mass).**
If the memory-induced mass depends on coordinate frequency, $\mu=\mu(\omega)$, the $Z$ factor makes $m_{\rm kin}\neq m_{\rm rest}$ at $O(1)$. If $k$ depends only on the invariant interval, then $\mu=\mu(p^2)$, the pole $p^2=m^2c^2$ is Lorentz invariant, and $m_{\rm kin}=m_{\rm rest}$ automatically. This is why C1 requires an invariant $\tau$.

---

## 6. Phase 5: what $\varepsilon>0$ predicts, and the bound it must survive

**The toy model.** In the NR sector, take $\partial_t f_{\rm map}(T)=0$ and fold the spatial flux into $\mathcal K$. Since $\partial_t I^\alpha=D^{1-\alpha}$, this gives

$$
\alpha\,\partial_t\psi=(1-\alpha)\,\mathcal K\,D^{1-\alpha}\psi
\quad\Longrightarrow\quad
\partial_t\psi=\mathcal G\,D^{\varepsilon}\psi,\qquad \mathcal G\equiv\tfrac{\varepsilon}{\alpha}\mathcal K\ \text{fixed}.
$$

At $\varepsilon=0$, $D^0=1$ and Phase 3 applies. For $\varepsilon>0$, put an eigenmode of $\mathcal G=-\iota H/\hbar$ through the symbol $(-\iota\omega)^{\varepsilon}=\lvert\omega\rvert^{\varepsilon}e^{-\iota\pi\varepsilon\,{\rm sgn}\,\omega/2}$. Checked numerically: ${\rm Im}\,\Omega=-(\pi/2)\varepsilon\lvert\omega\rvert$ for both signs.

- **Norm leak:** $\Gamma\simeq\frac{\pi}{2}\,\varepsilon\,\lvert E\rvert/\hbar$. Both signs of $E$ decay, so the model is stable but dissipative.
- **Log shift:** $E\to E\,[1+\varepsilon\ln(\lvert E\rvert\tau_*/\hbar)]$, where $\tau_*$ is the kernel's time unit.
- **Power-law transients** of amplitude $\propto\varepsilon\,t^{-\alpha}$ (Mittag-Leffler, since $1/\Gamma(1-\alpha)\approx\varepsilon$).
- **The energy zero is physical for $\alpha<1$.** $\psi\to e^{\iota\theta t}\psi$ is not a symmetry of $D^\varepsilon$. Expect Galilean covariance (Bargmann central charge) to hold only at $\alpha=1$; this needs proof.

**Bound.** By F2 the mass *is* memory, so the rest phase passes through $D^\varepsilon$ and $\Gamma\sim\varepsilon\,mc^2/\hbar$. Inputs:
- invisible nucleon-decay limits (KamLAND, SNO+): $\tau_N\gtrsim10^{29}$ yr $=3.2\times10^{36}$ s;
- nucleon rest frequency: $\omega_N=1.43\times10^{24}\ {\rm s^{-1}}$.

These give

$$
\varepsilon\lesssim\frac{2}{\pi\,\omega_N\tau_N}\approx1.4\times10^{-61}.
$$

So for matter, $\alpha_{\rm frac}=1$ to about 61 decimal places, and $\Phi_0$ has to be correspondingly large to keep $\varepsilon\Phi_0$ fixed. The framework must explain that hierarchy, or confine $\alpha<1$ (for example the $\alpha=3$ regime) to a sector that does not carry rest energy. **Do this estimate first.** It decides whether "$\alpha$ visibly below 1" is alive for matter at all.

---

## 7. Order of work and kill criteria

1. **Phase 0 table.** Decide C1–C7 in writing.
2. **G1, feasibility (one page).** The nucleon bound above. If $\alpha_{\rm frac}$ must be visibly below 1 for matter, stop here.
3. **Phase 1.** Weyl equation at $\varepsilon^0$.
4. **K1.** Compute $\Pi\Phi$ at coincidence in the $e_a$ basis. If there is no chirality flip and no $\iota$-proportional anti-Hermitian part, the limit is diffusion: Nelson's $D=\hbar/2m$ without the $i$, which is exactly what `CAUSAL_FIELD_HAMILTONIAN_REVIEW.md` found for the code operator. Stop and find the complex structure.
5. **Phase 2,** with two gates:
   - **K2:** the $\iota$ from 2.3 (dynamical) equals the $\iota$ from 2.2 (algebraic).
   - **K3:** the sign of the memory-induced $\mu^2$. $\iota^2=-1$ and $\iota^\dagger=-\iota$ both enter, and $\mu^2<0$ means a tachyon, which kills the route.
6. **Phase 3.** Semigroup iff $\alpha=1$; Born conservation forces an anti-Hermitian generator; Stone; limit interchange.
7. **Phase 4,** with gate **K4:** $m_{\rm kin}=m_{\rm rest}$, which requires the invariant kernel.
8. **Phase 5.** Predictions.
9. **Numerics** (§9).

---

## 8. What changes in `proof_hierarchy2` §5

- Replace "Markovian $=w\to\delta$" with F1/F2 plus 3.1 (semigroup $\iff\alpha=1$).
- Remove the $U=e^{i\theta G}$ ansatz. It is the conclusion, and the steps after it are the definition of a generator.
- "Convex combination of $I$ and a self-adjoint $R$ … exponentiation yields unitary": $e^{A}$ with $A$ self-adjoint is positive, not unitary. Unitarity needs $e^{-\iota A}$, and supplying that $\iota$ is the actual content (2.3, 3.2).
- "$R^\dagger=R$ because $J$ is real": a real symmetric generator gives diffusion, not Schrödinger.
- Eq. (21), interchange of limit and projection: linearity is insufficient. It needs a uniform bound (3.4) and must be redone under the scaling.

---

## 9. Numerics in this repo

- **Toy:** $\partial_t\psi=\mathcal G D^\varepsilon\psi$ with $\mathcal G=-\iota H/\hbar$, on a 1D harmonic oscillator, using an L1 or a correct Grünwald–Letnikov scheme. Check three things:
  - convergence to Crank–Nicolson Schrödinger as $\varepsilon\to0$, at rate $\varepsilon\ln\varepsilon$;
  - the leak rate against $\frac{\pi}{2}\varepsilon E/\hbar$;
  - the semigroup defect $\lVert U(t+s)-U(t)U(s)\rVert\propto\varepsilon$.
- **Don't use `kernels/fractional_memory.py::fractional_kernel_weights` as it stands.**
  - It samples $\tau^{\alpha-1}$ at $\tau=10^{-8}\Delta t$ for $j=0$ and then renormalizes to unit sum. For $\alpha\le0.5$ the present step takes almost all the weight (at $\alpha=0.5$ with 100 steps, $w_0=0.998$; at $\alpha=0.3$, $w_0\approx1.000$). At $\alpha=1$ it is a flat average.
  - The docstring's GL weights $(-1)^k\binom{\alpha}{k}$ are not what the code computes.
  - The unit-sum normalization also discards the $1/\Gamma(\alpha)$ and $(1-\alpha)\leftrightarrow\Gamma(1-\alpha)$ bookkeeping that F1 and F2 depend on.

---

## References

- M. Naber, *Time fractional Schrödinger equation*, J. Math. Phys. **45**, 3339 (2004).
- N. Laskin, *Time fractional quantum mechanics*, Chaos Solitons Fractals **102**, 16 (2017).
- B. N. N. Achar, B. T. Yale, J. W. Hanneken, *Time fractional Schrödinger equation revisited*, Adv. Math. Phys. **2013**, 290216.
- A. Iomin, *Fractional-time quantum dynamics*, Phys. Rev. E **80**, 022103 (2009).
- J. Peng, K. Li, *A note on property of the Mittag-Leffler function*, J. Math. Anal. Appl. **370**, 635 (2010).
- E. B. Davies, *Markovian master equations*, Commun. Math. Phys. **39**, 91 (1974).
- M. Reed, B. Simon, *Methods of Modern Mathematical Physics* I–II (Stone; Kato–Rellich).
- D. Hestenes, *Real spinor fields*, J. Math. Phys. **8**, 798 (1967).
- W. E. Baylis, *Electrodynamics: A Modern Geometric Approach* (Birkhäuser, 1999).
- E. C. G. Stueckelberg, *Quantum theory in real Hilbert space*, Helv. Phys. Acta **33**, 727 (1960).
- V. Bargmann, *On unitary ray representations of continuous groups*, Ann. Math. **59**, 1 (1954).
- E. Nelson, *Derivation of the Schrödinger equation from Newtonian mechanics*, Phys. Rev. **150**, 1079 (1966).
- KamLAND, PRL **96**, 101802 (2006); SNO+, PRD **105**, 112012 (2022): invisible nucleon decay.
