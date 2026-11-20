# Canonicalization checks

Machine checks of the algebra and representation theory in "Field Theoretic Self-Attention:
A Canonicalization of Born, Lorentz, and Killing", the companion to "On Information Physics"
(the Causal Accumulation Law). The paper is `FieldTheoretic_SelfAttention_Born_Lorentz_Killing.pdf`
in this folder, with its LaTeX source. Each step in the paper carries a "Checked" line naming
the Lean theorems and sympy checks below. Two independent methods:

- `lean/` - Lean 4 (v4.35.0) with Mathlib v4.35.0 and PhysLean. 207 theorems, no `sorry`,
  no `native_decide`, no custom axioms. `CalChecks/Crosscheck.lean` prints the axioms of
  every theorem; all depend only on `propext`, `Classical.choice`, `Quot.sound`.
- `python/` - sympy exact symbolic checks, 67 of 67 passing.

## Run

```
cd lean
lake exe cache get
lake build CalChecks
```

```
python python/check_onesheet_sympy.py
```

## Lean files

| File | Content |
|---|---|
| `Basic.lean` | Carrier, quaternion relations, Born quadratic, Hermitian slice and Minkowski determinant, purity, Lagrange identity, CHSH, uniqueness of alpha = 2 |
| `Extras.lean` | Bell correlation E = -a.b, Killing form of gl(2), Cl(3,0) map to the Pauli matrices |
| `RepTheory.lean` | Characters and weights for Clebsch-Gordan, SO(4) by left and right quaternion multiplication, real Killing signature, fractional integral eigenfunction |
| `Complete.lean` | Carrier bijection, cone stabilizer, POVM rows, purity from channels, SU(2) to SO(3), Tsirelson bound |
| `Structure.lean` | Kernel {+1, -1} of the spinor map, Mathlib `killingForm` over C and R, positive cone as one congruence orbit |
| `Clifford.lean` | Cl(3,0) isomorphic to M2(C) as real algebras, dimension 8 |
| `Final.lean` | 1 + 3 split under conjugation, boost dependence of purity, the Born rule as a kinematic theorem (`born_kinematic`) |
| `SO4.lean` | (Sp(1) x Sp(1)) / {+1, -1} isomorphic to SO(4) as groups: every rotation of the four channels is x -> p x q for unit quaternions (via Cartan-Dieudonne), kernel exactly (1,1) and (-1,-1) |
| `Crosscheck.lean` | Agreement with PhysLean's Pauli matrices and SL(2,C) results; axiom audit |

## Kinematics

The Born, Lorentz and Killing results are kinematic: they are proved from the carrier
`J` in M2(C) and its closure `J J†` alone, with no equation of motion. For Born this is
`born_kinematic`: for any nonzero carrier and any measurement (`E_m` positive semidefinite,
summing to 1), the weights `tr(E_m J J†) / tr(J J†)` are non-negative, sum to one, and are
unchanged under `J -> J U` with `U` unitary. Additivity is `born_additive`.

## Not formalized

- Zeeman's theorem.
- Clebsch-Gordan as a decomposition of representations (characters and weights only).
- The dynamics: the CAL integral itself and the transport and holonomy rows, which the paper defers.
- The forward lemma the paper discloses as unwritten (that the alpha = 2 integration carries
  the source current to 1 - R_n). `uniq_alpha_dotσ` proves the compatibility result only.
- The Bayesian reading of `J J†` and the category-theory remark, which the paper marks as readings.

## Wording fixed in the paper

These three points were found by the checks and are corrected in the current edition.

1. The uniqueness lemma needs 1 and R_n linearly independent; R^dagger = R and R^2 = 1
   alone allow R = +1 or -1. It does hold for R_n = n.sigma (`uniq_alpha_dotσ`).
2. The Bell paragraph's local line 1 - 2 theta/pi has the opposite sign to E = -cos theta.
3. I^alpha e^{s tau} = s^{-alpha} e^{s tau} needs lower limit -infinity and Re s > 0.
