# Plan: Schrödinger's equation from the Causal Accumulation Law as $\alpha_{\rm frac}\to 1^-$

**Target.** Show that the CAL, with a power-law memory under causal Weyl integration, gives

$$
i\hbar\,\partial_t\psi=\Big[\frac{(\mathbf p-q\mathbf A)^2}{2m}+qV\Big]\psi
$$

in the limit $\alpha_{\rm frac}\to1^-$, together with one stated equation of motion for the carrier. Every ingredient (first-order time evolution, linearity, the $i$, unitarity, $m$, $V$, $\mathbf A$) is traced to a named source. Nothing equivalent to the conclusion is assumed, so there is no $U=e^{i\theta G}$ ansatz.

**Notation.** $\alpha\equiv\alpha_{\rm frac}$, $\varepsilon\equiv1-\alpha\to0^+$, $\iota\equiv e_1e_2e_3$.

---

## 0. Kernel, and why the limit is taken from below

**Choice: power law with causal Weyl integration.**

$$
W^\alpha J(t)=\frac1{\Gamma(\alpha)}\int_{-\infty}^{t}(t-\tau)^{\alpha-1}J(\tau)\,d\tau,
\qquad
T_\alpha=\alpha J-(1-\alpha)\,\kappa\,\tau_*^{-\alpha}\,\widetilde{\mathcal K}\,W^\alpha J .
$$

- **The fiber factor.** $\widetilde{\mathcal K}$ collects $\Pi\,PPPP\,\Phi\,P$. Here it is taken to be time-translation invariant, with dimensionless strength $\kappa$.
- **The time scale.** A power law has no scale of its own, so $\tau_*$ has to be supplied for the units to close.
- **Which side.** Use the left-sided (Liouville) form, which integrates over the past. Some texts, e.g. Miller–Ross, reserve "Weyl" for $\int_t^\infty$; that integrates over the future, the wrong side of time, and is excluded.
- **Fourier multiplier.** $W^\alpha e^{\iota\omega t}=(\iota\omega)^{-\alpha}e^{\iota\omega t}$ with $(\iota\omega)^{-\alpha}=\lvert\omega\rvert^{-\alpha}e^{-\iota\pi\alpha\,{\rm sgn}\,\omega/2}$. This was verified by direct numerical integration for $\alpha=0.7,\,0.95$ and both signs of $\omega$. The $\iota$ here must be the complex structure of Lemma 2, since that is the one the carrier's time-Fourier analysis uses.

**Why $1^-$ and not $1$.**
- **For $\alpha<1$:** the kernel $(t-\tau)^{\alpha-1}$ decays, so $W^\alpha$ of an oscillating carrier converges (Dirichlet test).
- **At $\alpha=1$:** the kernel is $\Theta$, which does not decay, so $\int_{-\infty}^te^{\iota\omega\tau}d\tau$ does not converge. Its value $(\iota\omega)^{-1}$ exists only as $\lim_{\alpha\to1^-}(\iota\omega)^{-\alpha}$.
- **For $\alpha>1$:** the kernel grows and the integral diverges.
- **Consequence.** On the infinite past, the $\alpha=1$ theory is *defined* as the limit from below, and from below only. On that side the memory weight $1-\alpha$ also stays positive.

**Lemma 1 (the past drops out as $\alpha\to1^-$).**
The memory multiplier is $m_\varepsilon(\omega)=\varepsilon\kappa(\iota\omega\tau_*)^{-(1-\varepsilon)}$.
- For each $\omega\neq0$, $m_\varepsilon(\omega)\to0$.
- Uniformly, $\lvert m_\varepsilon(\omega)\rvert\le\varepsilon\kappa\max\!\big(1,(\lvert\omega\rvert\tau_*)^{-1}\big)$.

Hence

$$
\lVert T_\alpha-J\rVert\le\varepsilon\Big[(1+\kappa)\lVert J\rVert+\kappa\,\tau_*^{-1}\lVert W^1J\rVert\Big]=O(\varepsilon)
$$

whenever $J$ has a square-integrable antiderivative from $-\infty$, i.e. $\int\lvert\hat J\rvert^2/\omega^2\,d\omega<\infty$. That is Weyl's own admissibility condition: no weight at zero frequency.

What the bound implies:

- **The limit is non-uniform in frequency.** At fixed $\varepsilon$, modes with $\lvert\omega\rvert\tau_*\lesssim\varepsilon\kappa$ keep an $O(1)$ memory. Check: at $\varepsilon=10^{-3}$ and $\omega\tau_*=10^{-6}$ the multiplier is about $10^3$. The infrared remembers longest.
- **Matter has a built-in gap.** The rest phase puts the spectrum at $\lvert\omega\rvert\approx\omega_C=mc^2/\hbar$. The effective small parameter is therefore
  $$\varepsilon_{\rm eff}=\frac{\varepsilon\kappa}{\omega_C\tau_*},$$
  not $\varepsilon$. If the memory scale is much shorter than a Compton period ($\tau_*\ll1/\omega_C$), then $\varepsilon_{\rm eff}\gg\varepsilon$.
- **Massless and soft modes have no gap.** For them the limit from below is not uniform.

**Corollaries.**
- $f_{\rm map}$ is bounded and linear, so $f_{\rm map}(T_\alpha)\to f_{\rm map}(J)$. Swapping the limit with the projection is immediate; this replaces Eq. (21) of `proof_hierarchy2`.
- In the limit the state is the present carrier alone. That is Markovian by construction (the poster's "forgetful limit"), with no $\delta$-collapse needed; this replaces Prop. 1.

**Consequence that drives the rest.** In the limit, $\Pi$, $P$, $\Phi$ and $k$ all leave together with the past. So nothing in the Schrödinger equation ($m$, $V$, $\mathbf A$ or $i$) can come from the memory term. The draft's operator form $\hat T_\alpha=\alpha I-(1-\alpha)R$ tends to $I$, so $H=f_{\rm map}[\hat T]$ would be a constant. $H$ has to come from the carrier's own equation of motion (§3).

---

## 1. What survives the limit: the carrier

$J=\sum_{a=0}^3Z_ae_a\in M_2(\mathbb C)\cong Cl(3,0)$, with $e_0=1$ and $e_k=\sigma_k$. That is 8 real components, the same content as a Dirac spinor (Baylis's APS eigenspinor, or Hestenes's even STA spinor). Lorentz transformations act on the left: $J\to LJ$, $L\in SL(2,\mathbb C)$.

**Identities (verified numerically):**
- $J\bar J=\det J\cdot\mathbb 1$;
- $\tfrac12{\rm tr}\,JJ^\dagger=\sum_a\lvert Z_a\rvert^2$;
- $\det(JJ^\dagger)=\lvert\det J\rvert^2$;
- $\sigma_1\sigma_2\sigma_3=i\mathbb 1$.

**How they read physically:**
- $j\equiv JJ^\dagger$ is the current paravector; it transforms as $j\to LjL^\dagger$.
- **Born by trace:** $\tfrac12{\rm tr}\,j=j^0$, the probability density.
- **Lorentz by determinant:** $\det j=j_\mu j^\mu$, an invariant.
- $\det J=\rho\,e^{\iota\beta}$ packs the scalar and pseudoscalar bilinears together: $\rho\cos\beta=\bar\psi\psi$, $\rho\sin\beta=\bar\psi i\gamma_5\psi$. Then $\lvert\det J\rvert^2=j\cdot j$ is the Fierz identity.

**Deliverable:** prove this dictionary.

---

## 2. The $i$ is forced

**Lemma 2.** Project to $\psi=JP_+$, with $P_+=\tfrac12(1+e_3)$; this minimal left ideal has 4 real dimensions.
- The real span of $SL(2,\mathbb C)$ is all of $M_2(\mathbb C)$, 8-dimensional (checked).
- So the real-linear maps on the ideal that commute with the Lorentz action are right multiplication by $P_+M_2(\mathbb C)P_+\cong\mathbb C$, a 2-dimensional commutant (checked).
- The only Lorentz-commuting complex structures are therefore $\pm\iota$. The sign is the time-orientation convention.

The same $\iota$ is the one in the Weyl symbol $(\iota\omega)^{-\alpha}$ (§0). **Deliverable:** write out the proof (half a page).

---

## 3. Equation of motion for the limiting carrier: the one real input

**K1: $\nabla_\mu T^{\mu\cdots}=0$ alone gives a massless field.**
In the limit it becomes $\bar\partial J=0$. On a plane wave $A\,e^{-\iota p\cdot x}$ this reads $\bar pA=0$, and $\det\bar p=p^2$ (checked).
- If $p^2\neq0$, then $A=0$.
- If $\det A\neq0$, then $p=0$.

So every nontrivial mode is null with $\det A=0$: massless, and with no Schrödinger limit.

**Recommended: a uniqueness argument.**
Require the equation to be:
- first order, linear and local;
- covariant under left $SL(2,\mathbb C)$;
- parity invariant;
- invariant under the $U(1)$ generated by $\iota$. Its Noether current is $JJ^\dagger$, so Born conservation is built in.

Then the classification of first-order Lorentz-covariant equations (Gel'fand–Minlos–Shapiro) should leave Dirac's equation with one real constant $m\ge0$. In STA form it is $\nabla\psi\,\iota\sigma_3\hbar=m\psi\gamma_0$; Baylis gives the APS form. **Deliverable:** the proof, or the theorem with its hypotheses checked.

**Status of $m$.** $m$ is a closure constant, with the same status as $\Lambda$ in the Lovelock panel. It multiplies ${\rm Re}\det J=\bar\psi\psi$.

**Gauge fields.** $\mathbf A$ and $V$ enter through the covariant derivative in this equation.

---

## 4. Unitarity

1. The Born current is conserved, so $\int j^0\,d^3x$ is constant.
2. Evolution on $L^2$ of the ideal is therefore unitary.
3. Stone's theorem gives $U=e^{-\iota Ht/\hbar}$, with $H$ the self-adjoint Dirac Hamiltonian (Kato–Rellich, or the Dirac-specific results in Thaller).

---

## 5. Nonrelativistic limit

This is standard (large/small components, or Foldy–Wouthuysen). With $JP_+=e^{-\iota mc^2t/\hbar}(\varphi,\chi)$:

$$
i\hbar\,\partial_t\varphi=\Big[\frac{(\mathbf p-q\mathbf A)^2}{2m}+qV-\frac{q\hbar}{2m}\boldsymbol\sigma\cdot\mathbf B\Big]\varphi .
$$

The scalar Schrödinger equation is the spin-independent sector.

**Order of limits.** Take $\alpha\to1^-$ first, then $c\to\infty$. Lemma 1's rate is $\varepsilon\kappa/(\omega_C\tau_*)$, and $\omega_C\propto c^2$, so the memory bound improves as $c\to\infty$ and the limits commute on the massive sector. Check this explicitly.

---

## 6. The approach from below: what $\varepsilon>0$ leaves

Expand the Weyl multiplier about $\alpha=1^-$ (verified numerically: the $O(\varepsilon)$ remainder is $\sim\varepsilon^2$ and the $O(\varepsilon^2)$ remainder is $\sim\varepsilon^3$):

$$
\hat T_\alpha(\omega)=\Big[1-\varepsilon-\frac{\varepsilon\kappa}{\iota\omega\tau_*}
-\frac{\varepsilon^2\kappa}{\iota\omega\tau_*}\Big(\ln(\lvert\omega\rvert\tau_*)+\tfrac{\iota\pi}{2}{\rm sgn}\,\omega\Big)+O(\varepsilon^3)\Big]\hat J(\omega).
$$

- **$O(\varepsilon)$:** $-\varepsilon\kappa/(\iota\omega\tau_*)=\iota\,\varepsilon\kappa/(\omega\tau_*)$. This is multiplication by $\iota$, a reactive (phase-type) shift, if $\kappa$ is Hermitian and commutes with $\iota$. The power-law shape does not appear at this order: it is the order-1 Weyl integral, the same for any kernel family with this limit.
- **$O(\varepsilon^2)$: the power-law signature.**
  - a log running $\propto\varepsilon^2\kappa\ln(\lvert\omega\rvert\tau_*)/(\omega\tau_*)$;
  - the first non-$\iota$ term, $-\varepsilon^2\kappa\pi/(2\lvert\omega\rvert\tau_*)$, which has the same sign for both signs of $\omega$ (checked).
- **Decision D6: how $T$ enters the dynamics for $\alpha<1$.** If the equation of motion is imposed on $T$ ($\mathcal D T_\alpha=0$), and the memory multiplier commutes with $\mathcal D$ and is invertible, then $\mathcal DJ=0$ and the memory is invisible. The $\alpha<1$ physics then lives only in $[\mathcal D,\widetilde{\mathcal K}]\neq0$ (the index structure of $\Pi$ and $\Phi$, the $x$-dependence of $P$), or in observables built from $T$ rather than $J$.
- **No generic bound on $\varepsilon$.** For matter, everything scales with $\varepsilon_{\rm eff}=\varepsilon\kappa/(\omega_C\tau_*)$. A bound needs $\kappa$, $\tau_*$ and D6.

---

## 7. Order of work

1. **Lemma 1** as a one-sided limit with the Weyl condition, including the non-uniformity statement.
2. **The bilinear dictionary** (§1) and **Lemma 2.**
3. **The equation of motion.** K1 first; then the uniqueness proof leading to Dirac.
4. **Born → unitary → Stone.**
5. **NR reduction** and the order of limits.
6. **D6**, then the $O(\varepsilon)$ and $O(\varepsilon^2)$ terms on the Dirac carrier.

---

## 8. What changes in `proof_hierarchy2` §5

- **Prop. 1.** Replace the $\delta$-collapse with Lemma 1. The past drops out as $\alpha\to1^-$ because of the weight $(1-\alpha)$ and the Weyl condition, not because the kernel localizes.
- **Eq. (21).** It follows immediately from Lemma 1 and a bounded $f_{\rm map}$.
- **$H=f_{\rm map}[\alpha I-(1-\alpha)R]$.** It tends to $f_{\rm map}[I]$, a constant. $H$ has to come from the carrier's equation of motion.
- **The $U=e^{i\theta G}$ ansatz.** Remove it: it assumes the conclusion.
- **"$e^{\text{self-adjoint}}$ is unitary".** It isn't: unitarity needs $e^{-\iota A}$, and Lemma 2 supplies the $\iota$.

---

## 9. Numerics (§6 only)

- **Implementation.** The causal Weyl integral is a Fourier multiplier, so implement it by FFT with $(\iota\omega)^{-\alpha}$ on zero-mean input. That avoids the truncation error of a finite history.
- **Repo code.** `kernels/fractional_memory.py::fractional_kernel_weights` does not compute the GL weights its docstring states. It samples $\tau^{\alpha-1}$ at $10^{-8}\Delta t$ and renormalizes to unit sum; at $\alpha=0.5$ over 100 steps, the present step gets 99.8% of the weight. Don't use it for this.

---

## References

- S. G. Samko, A. A. Kilbas, O. I. Marichev, *Fractional Integrals and Derivatives* (Gordon & Breach, 1993): Liouville/Weyl integrals, admissibility.
- K. S. Miller, B. Ross, *An Introduction to the Fractional Calculus and Fractional Differential Equations* (Wiley, 1993): right-sided "Weyl" convention.
- W. E. Baylis, *Electrodynamics: A Modern Geometric Approach* (Birkhäuser, 1999).
- D. Hestenes, *Real spinor fields*, J. Math. Phys. **8**, 798 (1967).
- I. M. Gel'fand, R. A. Minlos, Z. Ya. Shapiro, *Representations of the Rotation and Lorentz Groups and their Applications* (Pergamon, 1963).
- B. Thaller, *The Dirac Equation* (Springer, 1992).
- L. L. Foldy, S. A. Wouthuysen, Phys. Rev. **78**, 29 (1950).
- M. Reed, B. Simon, *Methods of Modern Mathematical Physics* I–II.
- M. Naber, J. Math. Phys. **45**, 3339 (2004).
