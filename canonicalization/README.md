# Canonicalization checks

Machine checks of the algebra and representation theory in "Field Theoretic Self-Attention:
A Canonicalization of Born, Lorentz, and Killing", the companion to "On Information Physics"
(the Causal Accumulation Law). The paper comes in two editions that say the same thing block for
block: the matrix edition in M2(C), `FieldTheoretic_SelfAttention_Born_Lorentz_Killing.pdf`, and
the Clifford edition in Cl(3,0), `FieldTheoretic_SelfAttention_Born_Lorentz_Killing_Clifford.pdf`,
both in this folder with their LaTeX sources. Each step carries a short bracketed tag naming
the principal Lean theorems and sympy checks below. Two independent methods:

- `lean/` - Lean 4 (v4.35.0) with Mathlib v4.35.0 and PhysLean. 322 theorems, no `sorry`,
  no `native_decide`, no custom axioms. `CalChecks/Crosscheck.lean` prints the axioms of
  every theorem; all depend only on `propext`, `Classical.choice`, `Quot.sound`.
- `python/` - sympy exact symbolic checks, 67 of 67 passing.

Of the first 294, 207 are in the nine original files, whose axioms `Crosscheck.lean` prints. The
other 87 (`ThreeWay.lean` 12, `CliffordNative.lean` 72, `CliffordPhyslib.lean` 3) are under the
same conditions; each of those files ends with its own `#print axioms` lines.

## Run

```
cd lean
lake exe cache get
lake build CalChecks
```

```
python python/check_onesheet_sympy.py
python python/check_clifford_dictionary.py
python python/check_clifford_native.py
python python/check_naturality.py
python python/check_area_killing.py
python python/check_schild.py
python python/check_threeway.py
python python/check_radius.py
```

## Python files

| File | Content |
|---|---|
| `check_onesheet_sympy.py` | The 67 checks of the matrix edition, in M2(C) |
| `check_clifford_dictionary.py` | D1-D18: each Clifford operation against its matrix counterpart, in the Pauli representation |
| `check_clifford_native.py` | The same labels again in Cl(3,0) built from the generator relations, with no matrices until a final block `REP` that ties the product to the Pauli representation; 104 of 104 |
| `check_naturality.py` | N1-N3: no linear equivariant map from the carrier to the Hermitian slice; `J J†` is the only quadratic one invariant under right U(2), up to scale |
| `check_area_killing.py` | W1-W4: the area element of a plane of vectors is a Killing norm |
| `check_schild.py` | S1-S8: the squared area element against its root. The square is Schild's integrand and is invariant only under unit-Jacobian changes of sheet coordinates; its solutions have constant area density and solve the Nambu-Goto equations; a rotating string as a worked case. Classical, flat space |
| `check_threeway.py` | T1-T3: `det X = (tr X)^2/4 + (1/8) kappa(iX, iX)` on the Hermitian slice, in matrix and Clifford form; runs three scripts in `threeway/` |
| `check_radius.py` | T4-T7: the relation solved for the radius, `|x|^2 = (tr X)^2/4 - det X = -(1/8) kappa(iX, iX)`; rank one, where the Born and Killing readouts coincide; conjugation keeps all three readouts; congruence keeps only the determinant, a boost trading trace for Killing form, while SU(2) keeps all three. Each in matrix form and in Cl(3,0) without matrices. Sympy only |
| `threeway/` | Eight independent tests of that relation: exact, 50-digit numeric, mutation, pure-Python rational, step-by-step and symmetry arguments |

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
| `ThreeWay.lean` | `det X = (tr X)^2/4 + (1/8) kappa(iX, iX)` with Mathlib's `killingForm`: for all complex 2x2 over C, and for Hermitian X with the real form on su(2) |
| `CliffordNative.lean` | The dictionary as theorems, with Mathlib's `reverse` and `involute` on `CliffordAlgebra`: reversion is dagger, Clifford conjugate is adjugate, `q * conj q` is the determinant, scalar part is half the real trace. Then Clifford-form statements: closure, norm, sign flip under the pseudoscalar, real Killing form, three-readout relation |
| `CliffordPhyslib.lean` | PhysLean's own Cl(3,0) (`PauliMatrix.ofCliffordAlgebra`): same quadratic form, same map to the Pauli matrices |

## Conventions the Clifford checks pin down

- The constant 1/8 in the three-readout relation is for the Killing form over the centre
  (the complex Lie algebra) and for the bivector algebra. With Mathlib's real Killing form on
  all of Cl(3,0) it is 1/16 (`threeway_Cl3_real`, against `threeway_spin3`).
- `N q = q * conj q` is central, in the span of 1 and the pseudoscalar, and is a real scalar
  only on paravectors and bivectors (`N_eq`, `N_pre`, `N_biv_val`).
- The Bell correlation `E(a,b) = -a.b` holds in the tensor product over the centre and fails
  in the plain real tensor product of two copies of Cl(3,0) (sympy, notes under C1-C3 and D17).

## Kinematics

The Born, Lorentz and Killing results are kinematic: they are proved from the carrier
`J` in M2(C) and its closure `J J†` alone, with no equation of motion. For Born this is
`born_kinematic`: for any nonzero carrier and any measurement (`E_m` positive semidefinite,
summing to 1), the weights `tr(E_m J J†) / tr(J J†)` are non-negative, sum to one, and are
unchanged under `J -> J U` with `U` unitary. Additivity is `born_additive`.

## Not formalized

- The complex Killing form stated in Clifford form (sympy only). Several Clifford-form
  theorems are on coordinate families, and most of their proofs transport through the
  isomorphism; only the statements are matrix-free.
- Naturality N1-N3 and area W1-W4, S1-S8 (sympy only).
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

## The Schrodinger edge (alpha -> 1 from below)

Independent checks of "Schrodinger evolution from the bare Causal Accumulation Law at alpha -> 1^-"
(`docs/paper/cal_schrodinger_collapse.pdf`) and of its five-page canonical sheet
(`docs/paper/cal_schrodinger_canonicalization.pdf`). The paper's own script is
`scripts/cal_edge_transducer.py`; none of the checks below reuse it, except `N_followup.py`,
which loads it to compare numbers.

| File | Content |
|---|---|
| `lean/CalChecks/SchrodingerEdge.lean` | 28 theorems: a normalizable memory has no root on or outside the unit circle; flat-weight roots below, at and above alpha = 1; the weights tend to 1; the homogeneous recurrence above the edge; concurrence equals 1/gamma and sech of the rapidity; the self-normalization lemma |
| `python/edge/E_exact.py`, `E_rotation.py` | Exact algebra in sympy: 70 checks labelled by the paper's statement, including the order-by-order expansion behind the rotation theorem and its next-order terms |
| `python/edge/A_*.py` | mpmath at 40 digits or more: the Weyl integral, the Jonquiere memory sum, exact roots down to 1 - alpha = 1e-40, the tables, root counts by the argument principle |
| `python/edge/N_own.py`, `N_followup.py` | Direct iteration with independent code, then the paper's script run for comparison |
| `python/edge/G_check.py` | G1-G4: the corrected impulse response above the edge, the SL(2,R) class of the flat-weight recurrence, the unitarity exponents |

With `SchrodingerEdge.lean` the library has 322 theorems.

`E_exact.py` reports one failure by design: it tests the impulse response as the paper first
stated it (`exp(-s|n-n0|)/(2 sinh s)`), which has the wrong sign and answers a source held on.
The paper and the sheet now carry the corrected form, which `G_check.py` verifies.

What the checks changed in the paper: the impulse response above the edge; the wavepacket table
now shows the norm lost and the distance without rescaling; "70 to 97 percent of the norm" is a
ratio of amplitudes; the brake thresholds are for the simulated weights (0.097, 0.513); the
long-wavelength exponent keeps its -(pi/4)(1 - alpha) term; two rows of the normalizable-kernel
table depended on a 400-lag cut; the frozen carrier at alpha = 1 is stated for convergent sums.

Not proved: the rotation theorem (formal expansion, reproduced symbolically and matched by exact
roots to 34 digits); Jonquiere's formula (cited, verified numerically); the two-carrier
"no interaction" remark, which no check covers.
