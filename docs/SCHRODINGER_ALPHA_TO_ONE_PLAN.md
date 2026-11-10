# Plan: Schrödinger's equation from the Causal Accumulation Law as $\alpha_{\rm frac}\to 1^-$

**Target.** Show that the CAL in the limit $\alpha_{\rm frac}\to1^-$, together with one stated equation of motion for the carrier, gives

$$
i\hbar\,\partial_t\psi=\Big[\frac{(\mathbf p-q\mathbf A)^2}{2m}+qV\Big]\psi .
$$

Each ingredient has to be traced to a named source: first-order time evolution, linearity, the $i$, unitarity, $m$, $V$ and $\mathbf A$. Nothing equivalent to the conclusion may be assumed, so there is no $U=e^{i\theta G}$ ansatz.

**Notation.** $\alpha\equiv\alpha_{\rm frac}$, $\varepsilon\equiv1-\alpha$, $\iota\equiv e_1e_2e_3$. In schematic form the CAL is

$$
T_\alpha=\alpha J-(1-\alpha)\,\mathcal K_\alpha[J],\qquad
\mathcal K_\alpha[J](x)=\int_{J^-(x)}k\;\Pi\;PPPP\;\Phi\;P\;J\;d^4x'.
$$

---

## 0. The limit: the past drops out, whatever the kernel

**Lemma 1.**
Suppose $M\equiv\sup_{\alpha\in[\alpha_0,1)}\lVert\mathcal K_\alpha[J]\rVert<\infty$. Then

$$
\lVert T_\alpha-J\rVert\le\varepsilon\,(\lVert J\rVert+M)\;\longrightarrow\;0 .
$$

The bound does not depend on $k$, $\Pi$, $P$ or $\Phi$.

**Hypotheses** (state them; they are the only places the kernel enters):
1. The kernel's total weight on the past stays bounded as $\alpha\to1$. Any kernel whose normalization is fixed independently of $\alpha$ qualifies. Excluded are weights that grow like $1/(1-\alpha)$, such as $(1-\alpha)\tau^{-\alpha}$; that is a $\delta$ in disguise, not a different memory.
2. $J$ is bounded on $J^-(x)$, or the past starts at a finite $t_0$. Over an infinite past, the accumulated weight of a $J$ with nonzero mean can diverge.

**Corollaries.**
- $f_{\rm map}$ is bounded and linear, so $f_{\rm map}(T_\alpha)\to f_{\rm map}(J)$. Exchanging the limit with the projection is immediate; this replaces the argument behind Eq. (21) of `proof_hierarchy2`.
- At $\alpha=1$ the state is the present carrier alone. The evolution is Markovian by construction (the poster's "forgetful limit"), with no $\delta$-collapse needed; this replaces Prop. 1.

**The consequence that drives the rest of the plan.**
$\Pi$, $P$, $\Phi$ and $k$ all leave together with the past. So nothing in the Schrödinger equation (not $m$, $V$, $\mathbf A$, or $i$) can come from the memory term.

In the draft's operator form, $\hat T_\alpha=\alpha I-(1-\alpha)R$, the same limit gives $\hat T\to I$. Then $H=f_{\rm map}[\hat T]$ would be a constant, and the only dynamics would be a global phase. So $H$ has to come from the equation of motion of $J$ itself (§3).

---

## 1. What is left: the carrier

$J=\sum_{a=0}^3Z_ae_a\in M_2(\mathbb C)\cong Cl(3,0)$, with $e_0=1$ and $e_k=\sigma_k$. That is 8 real components, the same content as a Dirac spinor (Baylis's APS eigenspinor, or Hestenes's even STA spinor).

Lorentz transformations act on the left: $J\to LJ$ with $L\in SL(2,\mathbb C)$.

**Identities (verified numerically):**
- $J\bar J=\det J\cdot\mathbb 1$, where $\bar J$ is the Clifford conjugate;
- $\tfrac12{\rm tr}\,JJ^\dagger=\sum_a\lvert Z_a\rvert^2$;
- $\det(JJ^\dagger)=\lvert\det J\rvert^2$;
- $\sigma_1\sigma_2\sigma_3=i\mathbb 1$.

**How these read physically:**
- $j\equiv JJ^\dagger$ is the current paravector. It transforms as $j\to LjL^\dagger$.
- **Born by trace:** $\tfrac12{\rm tr}\,j=j^0$, the probability density.
- **Lorentz by determinant:** $\det j=\lvert\det J\rvert^2=j_\mu j^\mu$, an invariant.
- $\det J=\rho\,e^{\iota\beta}$ packs the scalar and pseudoscalar bilinears together: $\rho\cos\beta=\bar\psi\psi$ and $\rho\sin\beta=\bar\psi i\gamma_5\psi$. The relation $\lvert\det J\rvert^2=j\cdot j$ is the Fierz identity $j_\mu j^\mu=(\bar\psi\psi)^2+(\bar\psi i\gamma_5\psi)^2$.

The poster's closure line is therefore exactly the Dirac bilinear structure. Proving that dictionary is the first deliverable.

---

## 2. The $i$ is forced, not inserted

**Lemma 2.**
- **Setup.** Project to the Hilbert sector $\psi=JP_+$ with $P_+=\tfrac12(1+e_3)$, the minimal left ideal; it is a 2-component spinor with 4 real dimensions.
- **Spanning step.** The real span of $SL(2,\mathbb C)$ is all of $M_2(\mathbb C)$, which is 8-dimensional (checked).
- **Commutant.** The real-linear maps on the ideal that commute with the Lorentz action are therefore right multiplication by $P_+M_2(\mathbb C)P_+\cong\mathbb C$. This commutant is 2-dimensional (checked).
- **Conclusion.** The only complex structures that commute with Lorentz are $\pm\iota$, which is $\pm i$ on the column spinor. The sign is the time-orientation convention.

So the $i$ in Schrödinger's equation is the unique Lorentz-commuting complex structure on the projected carrier. **Deliverable:** write out the proof, which is half a page.

---

## 3. Equation of motion for $J$: the one real input

The CAL defines $T$ from $J$; it is not an equation of motion. At $\alpha=1$ it only says $T=J$, so the dynamics has to be stated.

**K1: $\nabla_\mu T^{\mu\cdots}=0$ alone gives a massless field.**
At $\alpha=1$ it becomes $\bar\partial J=0$. On a plane wave $J=A\,e^{-\iota p\cdot x}$ this reads $\bar pA=0$, and $\det\bar p=p^2$ (checked).
- If $p^2\neq0$, then $A=0$.
- If $\det A\neq0$, then $p=0$.

So every nontrivial mode is null and has $\det A=0$: the field is massless and has no Schrödinger limit. If "Lorentz by determinant" is meant to be a nonzero invariant, the divergence condition alone cannot be the dynamics.

**Recommended route: a uniqueness argument.**
Require the equation for $J$ to be:
- first order;
- linear and local;
- covariant under left $SL(2,\mathbb C)$;
- parity invariant;
- invariant under the $U(1)$ generated by $\iota$. Its Noether current is $j=JJ^\dagger$, so Born conservation is built in.

With these hypotheses, the classification of first-order Lorentz-covariant equations (Gel'fand–Minlos–Shapiro) should leave exactly Dirac's equation, with one real constant $m\ge0$. In STA form this is $\nabla\psi\,\iota\sigma_3\hbar=m\psi\gamma_0$; Baylis gives the APS form. Parity removes a chiral mass phase, and the $U(1)$ removes Majorana-type terms. **Deliverable:** the proof, or the cited theorem with its hypotheses checked one by one.

**Status of $m$.**
$m$ is a closure constant, with the same status as $\Lambda$ in the Lovelock panel. In the action it multiplies ${\rm Re}\det J=\bar\psi\psi$. It is not derived from the CAL at $\alpha=1$.

**Gauge fields.**
$\mathbf A$ and $V$ enter through the covariant derivative in this equation, i.e. the connection whose parallel transport is $P$. $P$ itself left with the memory, so the coupling has to sit in the equation of motion.

---

## 4. Unitarity

1. The Born current is conserved (Noether, $U(1)_\iota$), so $\int j^0\,d^3x$ is constant.
2. Evolution on $L^2$ of the ideal is therefore unitary.
3. Stone's theorem gives $U(t)=e^{-\iota Ht/\hbar}$ with $H$ the self-adjoint Dirac Hamiltonian. Self-adjointness with potentials needs Kato–Rellich or the Dirac-specific results in Thaller.
4. $\hbar$ converts units.

---

## 5. Nonrelativistic limit

This is standard: eliminate the small components, or use Foldy–Wouthuysen. Write

$$
JP_+=e^{-\iota mc^2t/\hbar}(\varphi,\chi),\qquad \chi=O(v/c)\,\varphi .
$$

The result is

$$
i\hbar\,\partial_t\varphi=\Big[\frac{(\mathbf p-q\mathbf A)^2}{2m}+qV-\frac{q\hbar}{2m}\boldsymbol\sigma\cdot\mathbf B\Big]\varphi .
$$

The scalar Schrödinger equation is the spin-independent sector.

**Order of limits.** Take $\alpha\to1^-$ first (Lemma 1), then $c\to\infty$. The two commute if the bound $M$ in Lemma 1 is uniform in $c$; check this.

---

## 6. Where the kernel comes back: $\alpha<1$

$$
T_\alpha=J-\varepsilon\big(J+\mathcal K_1[J]\big)+O(\varepsilon^2).
$$

The first correction depends on the kernel. It is the leading memory effect, and the only place where $\Pi$, $P$, $\Phi$ and $k$ enter the quantum regime. Every prediction (norm leak, energy shifts, non-Markovian tails, failure of the semigroup property per Peng–Li) is $\varepsilon$ times something built from $\mathcal K_1$.

**No generic bound on $\varepsilon$ follows.** The earlier $\varepsilon\lesssim10^{-61}$ estimate assumed the mass came from the memory, which Lemma 1 rules out. A bound exists only once $\mathcal K_1$ has been computed on the Dirac carrier and its coupling to the rest-energy phase is known.

---

## 7. Order of work

1. **Lemma 1**, with its two hypotheses stated.
2. **The bilinear dictionary** (§1) and **Lemma 2** (the $i$).
3. **The equation of motion.** K1 first: if $\nabla\cdot T=0$ alone is kept, stop, because the result is massless. Otherwise prove the uniqueness that leads to Dirac.
4. **Born → unitary → Stone.**
5. **NR reduction.**
6. **$\alpha<1$:** compute $\mathcal K_1$ on the Dirac carrier; predictions and bounds follow from it.

---

## 8. What changes in `proof_hierarchy2` §5

- **Prop. 1.** Replace the $\delta$-collapse with Lemma 1. The past drops out because of the weight $(1-\alpha)$; the kernel is irrelevant.
- **Eq. (21).** It follows immediately from Lemma 1 and a bounded $f_{\rm map}$.
- **$H=f_{\rm map}[\alpha I-(1-\alpha)R]$.** At $\alpha=1$ this is $f_{\rm map}[I]$, a constant. $H$ has to come from $J$'s equation of motion (§3).
- **The $U=e^{i\theta G}$ ansatz.** Remove it: it assumes the conclusion.
- **"$e^{\text{self-adjoint}}$ is unitary".** It isn't: $e^{A}$ with $A$ self-adjoint is positive. Unitarity needs $e^{-\iota A}$, and Lemma 2 supplies the $\iota$.

---

## 9. Numerics (only relevant for §6)

- `kernels/fractional_memory.py::fractional_kernel_weights` does not compute the GL weights its docstring states.
  - It samples $\tau^{\alpha-1}$ at $10^{-8}\Delta t$ and renormalizes to unit sum. At $\alpha=0.5$ over 100 steps, the present step gets 99.8% of the weight.
  - Fix it before using it to estimate $\mathcal K_1$ effects. It is irrelevant to the $\alpha=1$ derivation.

---

## References

- W. E. Baylis, *Electrodynamics: A Modern Geometric Approach* (Birkhäuser, 1999): APS eigenspinor, Dirac equation.
- D. Hestenes, *Real spinor fields*, J. Math. Phys. **8**, 798 (1967).
- I. M. Gel'fand, R. A. Minlos, Z. Ya. Shapiro, *Representations of the Rotation and Lorentz Groups and their Applications* (Pergamon, 1963): first-order relativistic equations.
- B. Thaller, *The Dirac Equation* (Springer, 1992): self-adjointness, NR limit.
- L. L. Foldy, S. A. Wouthuysen, Phys. Rev. **78**, 29 (1950).
- M. Reed, B. Simon, *Methods of Modern Mathematical Physics* I–II (Stone; Kato–Rellich).
- J. Peng, K. Li, J. Math. Anal. Appl. **370**, 635 (2010): Mittag-Leffler semigroup property only at $\alpha=1$ (for §6).
- M. Naber, J. Math. Phys. **45**, 3339 (2004): time-fractional Schrödinger (for §6).
