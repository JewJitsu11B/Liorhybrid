# Results: the Feynman path sum at closure order $\alpha=2$ with the light-cone kernel width

**The paper** (*An Axiomatic Derivation of the Feynman Path Sum at the Light-Cone Kernel Width and Closure Order α = 2 of the Causal Accumulation Law*): `docs/paper/cal_feynman_alpha2.pdf` (source `docs/paper/cal_feynman_alpha2.tex`). It has the full statements and proofs; this file summarizes them.

**Reproduce:** `python3 scripts/cal_feynman_alpha2.py` (numpy only, about 7 seconds). Units: $\hbar=c=\tau_c=1$ and $\mu=1$, unless stated.

**Inputs:**
- the CAL at $\alpha=2$, written as $T=2J+r_1K_2J=S$, where $S$ is a source with zero past;
- a light-cone kernel: at lag $\tau$ the profile is the light front of radius $c\tau$. Its Fourier transform is $\operatorname{sinc}(c|k|\tau)$ in $d=1,2,3$;
- no channel phase on the memory. With the phase, the mass is tachyonic;
- for the readout only: one sign of frequency.

## Results

1. **Lorentz invariance selects $\alpha=2$.**
   - The order-$\alpha$ symbol is $(\tau_cR)^{-\alpha}\sin((\alpha-1)\varphi)/((\alpha-1)\sin\varphi)$.
   - It depends on $s^2+c^2k^2$ alone only at $\alpha=2$ (and trivially at $\alpha=0$). The leading defect is $\alpha(2-\alpha)\varphi^2/6$.
   - At $\alpha=2$ the kernel is the retarded Green's function of $\square$. In 3+1 it is Kirchhoff's shell. In 1+1 it is one order-one accumulation along each null leg.
2. **Klein–Gordon.** After the source is off, $(\square+\mu^2)J=0$, with $\mu^2=r_1/(2\tau_c^2)$ and $mc^2=\hbar\sqrt{r_1/2}/\tau_c$. No double scaling is needed.
3. **Path sum.**
   - The Neumann series $\tfrac12\sum(-\mu^2)^nG^{*n}*S$ converges absolutely, with bound $\tfrac12\cosh\mu t$. It resums to the retarded propagator $G_\mu$.
   - It is a sum over piecewise-null histories with $-\mu^2$ per corner.
   - In 1+1, the $n$-th term is the $(2n+1)$-corner amplitude of Feynman's checkerboard divided by $2ic\mu$, because $(-\mu^2)^n=(i\mu)^{2n}$.
   - In 3+1, by descent, the histories without corners lie on the cone. The corners alone build the $J_1$ tail inside it.
4. **Pair composition.** The past enters only through $(J,\partial_tJ)$. This is not the faithful limit, because no records are kept.
5. **Unitarity and readout.**
   - Each sign of frequency is unitary, with generator $\sqrt{m^2c^4-\hbar^2c^2\nabla^2}$.
   - On positive frequency, $\int JJ^\dagger$ is conserved. So the trace readout is normalized in every fixed frame.
   - Mixed frequencies break this.
   - The Klein–Gordon charge is the frame-independent conserved quantity. It differs from the trace at order $v^2/c^2$.
6. **Feynman's kernel.** Strip the rest phase from the positive-frequency part of the path sum and let $c\to\infty$. It converges to $K_F$ with error $O(1/c^2)$. Its group law is Feynman's time slicing.

## Numbers

| check | result |
|---|---|
| symbol, closed vs numeric | agree to 6 digits; Lorentz defect 0 only at α = 0, 2 |
| KG roots, k = 0…4 | equal to √(k²+μ²) to 10 digits |
| 1+1 path sum vs ½J₀ | residual 3.35e-5 → 8.38e-6 as h halves (grid error) |
| checkerboard vs J₀ | 3.70e-2, 1.81e-2, 8.95e-3, 4.45e-3 (first order in ε) |
| 3+1 descent tail vs −μJ₁/(4π√w) | ≤ 2.5e-6 |
| pair composition | 1.1e-14 |
| NR limit, err·c² | 2.42, 3.32, 3.59, 3.65, 3.67 → predicted 3.672 |
| ‖J‖² spread: positive / mixed frequency | 2.5e-16 / 1.6e-1 |

## Open

- what selects one sign of frequency;
- interaction between carriers at α = 2;
- the spin (Dirac) form in 3+1;
- the relation between the α = 2 mass and the edge mass;
- a frame-independent position readout (Newton–Wigner).
