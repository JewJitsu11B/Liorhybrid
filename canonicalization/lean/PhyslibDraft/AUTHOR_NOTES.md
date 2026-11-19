# Author notes: SO(4) double cover, draft for PhysLean

These notes are for you, the author. PhysLean's AI policy makes the human author responsible for
every line and requires that the PR description and all replies to reviewers are written by a
human. The notes give you what you need to vouch for the draft: what each statement says, why it
is true, and where your paper uses it.

## Status

- Files: `QuaternionIsometry.lean` (148 lines) and `SO4.lean` (448 lines), in this folder.
- Both build against PhysLean master `8c87bdf` (2026-11-19) and Mathlib v4.35.0.
- Every main result depends only on `propext`, `Classical.choice`, `Quot.sound`.
- No `sorry`, no `axiom`, no `theorem` keyword (their rule: `lemma` unless listed as exempt).
- No line over 100 characters, no trailing whitespace, module docs in their i-iv format.
- Not done: PhysLean's own linters (`lint_all`, `module_doc_lint`, `theorem_lint`,
  `forMathlib_lint`, `lint-style.sh`). They only run inside a checkout of their repository.
- Not done: a search of open PhysLean or Mathlib pull requests for the same result. The installed
  copies of both libraries do not contain it.

## Where this comes from in the paper

"Field Theoretic Self-Attention: A Canonicalization of Born, Lorentz, and Killing" uses the
result in three places.

1. Born rule section, the row for (1/2) tr(J J†). The group-class note says that on the channel
   index a, (SU(2) x SU(2))/Z2 = SO(4) acts by real rotations, keeping both sum |Z_a|^2 and
   det J = sum Z_a^2.
2. Channels and multiplication table. The units satisfy e_j e_k = -delta_jk + epsilon_jkl e_l,
   which is Hamilton's convention i j = k. `leftMulMatrix` and `rightMulMatrix` are the matrices
   of left and right multiplication in exactly that basis 1, e_1, e_2, e_3.
3. Purity from the channels. The note says v is invariant under real SO(4) rotations of the
   channel index.

The paper states the isomorphism and uses the invariance. It does not prove the isomorphism. The
proof in the draft (through reflections) is standard mathematics and is not taken from the paper.
In the canonicalization checks the same result is `CalChecks/SO4.lean`; the invariance of the two
quantities is `sumSq_left`, `sumSq_right`, `normSq_left`, `normSq_right` in `RepTheory.lean`.

## What each piece says and why it is true

### QuaternionIsometry.lean (namespace `Quaternion`)

| Declaration | Plain statement | Why |
|---|---|---|
| `mul_star_mul_self` | u conj(x) u = 2<u,x> u - \|u\|^2 x | Componentwise polynomial identity. Derivation by hand: u conj(x) + x conj(u) = 2<u,x>, multiply on the right by u. |
| `IsUnitTwoSided` | A map is x -> p x q or x -> p conj(x) q with \|p\| = \|q\| = 1 | Definition. |
| `isUnitTwoSided_id` | The identity is of this form | p = q = 1. |
| `IsUnitTwoSided.comp` | The form is closed under composition | Four cases. conj reverses products, so two conj-type maps compose to a plain one. |
| `reflection_orthogonal_span_singleton` | For v != 0, reflection in the hyperplane orthogonal to v is x -> -u conj(x) u with u = v/\|v\| | The reflection is x - 2<u,x>u. Substitute the identity above with \|u\| = 1. |
| `isUnitTwoSided_reflection` | Every hyperplane reflection is of the form | Previous lemma, with p = -u, q = u. For v = 0 the reflection is the identity. |
| `isUnitTwoSided_linearIsometryEquiv` | Every linear isometry of H is of the form | Mathlib's Cartan-Dieudonne theorem (`LinearIsometryEquiv.reflections_generate_dim`) writes the isometry as a product of reflections; induct on the list using `comp`. |

### SO4.lean (namespace `SO4`)

| Declaration | Plain statement | Why |
|---|---|---|
| `coords`, `ofCoords` | Quaternion to its four real coordinates and back | Definitions; mutually inverse. |
| `leftMulMatrix p`, `rightMulMatrix q`, `starMatrix` | 4x4 real matrices of x -> p x, x -> x q, x -> conj(x) | Definitions; `*_mulVec` lemmas check they act as claimed. |
| `leftMulMatrix_mul`, `rightMulMatrix_mul` | L(ab) = L(a) L(b), R(ab) = R(b) R(a) | Entry by entry. Right multiplication reverses order. |
| `leftMulMatrix_mul_rightMulMatrix_comm` | L(p) and R(q) commute | Associativity: p (x q) = (p x) q. This is the "left SU(2)" and "right SU(2)" commuting. |
| `*_transpose_mul_self`, `det_*` | L^T L = \|p\|^2, det L = \|p\|^4, same for R; det of conj is -1 | Entry by entry; determinant by cofactor expansion. |
| `isometryOfOrthogonal` | An orthogonal 4x4 matrix gives a linear isometry of H | Orthogonal matrices preserve the dot product (PhysLean's `dotProduct_mulVec_orthogonal`), and \|x\|^2 is the dot product of coordinates. |
| `exists_leftMulMatrix_mul_rightMulMatrix` | Every R with R^T R = 1 and det R = 1 equals L(p) R(q) for unit p, q | The isometry is p x q or p conj(x) q. The second has determinant -1, so it is excluded. |
| `leftMulMatrix_mul_rightMulMatrix_eq_one` | L(p) R(q) = 1 with \|p\| = 1 forces p = q = 1 or p = q = -1 | Multiply by L(p)^T to get R(q) = L(p)^T. Comparing seven entries gives q = conj(p) and the imaginary parts of p equal to their negatives, so zero. Then p.re^2 = 1. |
| `quaternionCover` | The group homomorphism (p, q) -> matrix of x -> p x conj(q) | conj on the right factor is what makes it a homomorphism, because R reverses order. |
| `quaternionCover_surjective` | It is onto SO(4) | From the existence lemma, taking the second component to be conj(q). |
| `quaternionCover_eq_one_iff` | Its kernel is {(1,1), (-1,-1)} | From the kernel lemma. |
| `quaternionCoverEquiv` | (Sp(1) x Sp(1)) / kernel is isomorphic to SO(4) | First isomorphism theorem. |

## Questions a reviewer is likely to ask

- Why matrices and not linear maps? The target `Matrix.specialOrthogonalGroup (Fin 4) R` is a
  matrix group in Mathlib, and the determinant argument needs explicit matrices.
- Why `unitary H` for Sp(1)? It is Mathlib's existing group of elements with star a * a = 1.
  The file does not prove Sp(1) = SU(2); the module doc mentions the identification in words only.
- Is `QuaternionIsometry.lean` better suited to Mathlib? Possibly. It imports only Mathlib, which
  is what PhysLean's `ForMathlib` directory is for. A reviewer may ask for it to go upstream.
- Why are many proofs `fin_cases ... <;> simp ... <;> ring`? They are finite polynomial identities
  between explicit 4x4 matrices. A reviewer may prefer a more structural proof; that is a fair
  request and would be real work.
- Why is the kernel lemma stated with only \|p\| = 1 and not \|q\| = 1? It is not needed: q is
  determined by p.

## What you have to do yourself

1. Read both files until you can explain each lemma without help. The tables above are the map.
2. Confirm the name in the copyright header. I used the name from this machine's user folder.
3. Verify the reference. I cited Conway and Smith, "On Quaternions and Octonions" (2003), from
   memory, as the standard source for "every rotation of R^4 is x -> p x q". I have not opened
   the book. PhysLean's policy says the human must check that the work exists, that it states
   this, and any page numbers. Replace or remove the reference if you cannot confirm it.
4. Decide whether to cite your own paper. It is where the result is used, not where it is proved.
5. Put the files in a fork of PhysLean:
   - `QuaternionIsometry.lean` to `Physlib/Mathematics/ForMathlib/QuaternionIsometry.lean`
   - `SO4.lean` to `Physlib/Mathematics/Groups/SO4/Basic.lean`
   - change `public import PhyslibDraft.QuaternionIsometry` to the new module path
   - add both to `Physlib.lean`, keeping the list sorted
6. Run their checks in that checkout: `lake build`, `lake exe lint_all`,
   `lake exe module_doc_lint`, `lake exe theorem_lint`, `lake exe forMathlib_lint`, and
   `./scripts/lint/required/lint-style.sh` after committing.
7. Split it. Their guideline is under 200 lines per PR. A workable split is three PRs:
   - the quaternion isometry file;
   - `SO4.lean` sections A to D (matrices, surjectivity, kernel);
   - `SO4.lean` section E (the homomorphism and the isomorphism).
8. Write the PR description yourself and answer reviewers yourself. Their agent rules forbid an
   AI from doing either. Their policy has no explicit disclosure rule, but commits produced with
   AI help are expected to carry a co-author trailer.
9. Consider posting on their Zulip channel first; they ask for that before larger additions.
