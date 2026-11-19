/-
Copyright (c) 2026 Sam Leizerman. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Sam Leizerman
-/
module

public import Mathlib.LinearAlgebra.UnitaryGroup
public import Mathlib.GroupTheory.QuotientGroup.Basic
public import Mathlib.Analysis.Normed.Module.FiniteDimension
public import Mathlib.Tactic.FinCases
public import Mathlib.Tactic.Linarith
public import Physlib.Mathematics.ForMathlib.OrthogonalMatrix
public import PhyslibDraft.QuaternionIsometry
/-!

# The double cover of SO(4) by pairs of unit quaternions

The group `SO(4)` is the quotient of `Sp(1) × Sp(1)` by `{(1, 1), (-1, -1)}`, where `Sp(1)` is
the group of unit quaternions.

## i. Overview

A pair `(p, q)` of unit quaternions acts on `ℍ ≅ ℝ⁴` by `x ↦ p * x * star q`. This action is by
rotations, every rotation of `ℝ⁴` arises in this way, and two pairs give the same rotation
exactly when they differ by an overall sign. Since `Sp(1) ≅ SU(2)`, this is the statement
`SO(4) ≅ (SU(2) × SU(2)) / ℤ₂`: the four-dimensional Euclidean rotation group splits into a left
and a right factor. It is the Euclidean counterpart of the cover of the restricted Lorentz group
by `SL(2, ℂ)`.

In this file `Sp(1)` is `unitary ℍ` and `SO(4)` is `Matrix.specialOrthogonalGroup (Fin 4) ℝ`.

## ii. Key results

- `SO4.leftMulMatrix`, `SO4.rightMulMatrix` : the `4 × 4` real matrices of left and right
  multiplication by a quaternion, in the basis `1, i, j, k`.
- `SO4.exists_leftMulMatrix_mul_rightMulMatrix` : every special orthogonal `4 × 4` real matrix
  is `leftMulMatrix p * rightMulMatrix q` for unit quaternions `p` and `q`.
- `SO4.leftMulMatrix_mul_rightMulMatrix_eq_one` : such a product is the identity only for
  `p = q = 1` or `p = q = -1`.
- `SO4.quaternionCover` : the group homomorphism `Sp(1) × Sp(1) →* SO(4)`.
- `SO4.quaternionCover_surjective`, `SO4.quaternionCover_eq_one_iff` : it is surjective with
  kernel `{(1, 1), (-1, -1)}`.
- `SO4.quaternionCoverEquiv` : the resulting isomorphism from the quotient to `SO(4)`.

## iii. Table of contents

- A. Coordinates on the quaternions
- B. The matrices of left and right multiplication
  - B.1. Definitions
  - B.2. Action on coordinates
  - B.3. Multiplicativity
  - B.4. Orthogonality and determinant
- C. Surjectivity
  - C.1. The isometry of an orthogonal matrix
  - C.2. Every rotation is a two-sided multiplication
- D. The kernel
- E. The covering homomorphism
  - E.1. Unit quaternions
  - E.2. The homomorphism
  - E.3. Surjectivity, kernel and the isomorphism

## iv. References

- J. H. Conway and D. A. Smith, *On Quaternions and Octonions*, A K Peters, 2003.

-/

@[expose] public section

namespace SO4

open Matrix Quaternion
open scoped Quaternion

/-!

## A. Coordinates on the quaternions

-/

/-- The coordinates of a quaternion in the basis `1, i, j, k`. -/
def coords (x : ℍ) : Fin 4 → ℝ := ![x.re, x.imI, x.imJ, x.imK]

/-- The quaternion with the given coordinates in the basis `1, i, j, k`. -/
def ofCoords (v : Fin 4 → ℝ) : ℍ := ⟨v 0, v 1, v 2, v 3⟩

@[simp]
lemma coords_ofCoords (v : Fin 4 → ℝ) : coords (ofCoords v) = v := by
  ext i; fin_cases i <;> simp [coords, ofCoords]

@[simp]
lemma ofCoords_coords (x : ℍ) : ofCoords (coords x) = x := by
  ext <;> simp [coords, ofCoords]

lemma coords_add (x y : ℍ) : coords (x + y) = coords x + coords y := by
  ext i; fin_cases i <;> simp [coords]

lemma coords_smul (r : ℝ) (x : ℍ) : coords (r • x) = r • coords x := by
  ext i; fin_cases i <;> simp [coords]

lemma ofCoords_add (v w : Fin 4 → ℝ) : ofCoords (v + w) = ofCoords v + ofCoords w := by
  ext <;> simp [ofCoords]

lemma ofCoords_smul (r : ℝ) (v : Fin 4 → ℝ) : ofCoords (r • v) = r • ofCoords v := by
  ext <;> simp [ofCoords]

/-- The squared norm of a quaternion is the dot product of its coordinates with themselves. -/
lemma normSq_eq_coords_dotProduct (x : ℍ) : normSq x = coords x ⬝ᵥ coords x := by
  simp [normSq_def', coords, dotProduct, Fin.sum_univ_four]; ring

/-!

## B. The matrices of left and right multiplication

-/

/-!

### B.1. Definitions

-/

/-- The matrix of left multiplication `x ↦ p * x` in the basis `1, i, j, k`. -/
def leftMulMatrix (p : ℍ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![p.re, -p.imI, -p.imJ, -p.imK;
     p.imI, p.re, -p.imK, p.imJ;
     p.imJ, p.imK, p.re, -p.imI;
     p.imK, -p.imJ, p.imI, p.re]

/-- The matrix of right multiplication `x ↦ x * q` in the basis `1, i, j, k`. -/
def rightMulMatrix (q : ℍ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![q.re, -q.imI, -q.imJ, -q.imK;
     q.imI, q.re, q.imK, -q.imJ;
     q.imJ, -q.imK, q.re, q.imI;
     q.imK, q.imJ, -q.imI, q.re]

/-- The matrix of quaternion conjugation `x ↦ star x` in the basis `1, i, j, k`. -/
def starMatrix : Matrix (Fin 4) (Fin 4) ℝ := diagonal ![1, -1, -1, -1]

/-!

### B.2. Action on coordinates

-/

lemma leftMulMatrix_mulVec (p x : ℍ) : leftMulMatrix p *ᵥ coords x = coords (p * x) := by
  ext i
  fin_cases i <;> simp [leftMulMatrix, coords, mulVec, dotProduct, Fin.sum_univ_four] <;> ring

lemma rightMulMatrix_mulVec (q x : ℍ) : rightMulMatrix q *ᵥ coords x = coords (x * q) := by
  ext i
  fin_cases i <;> simp [rightMulMatrix, coords, mulVec, dotProduct, Fin.sum_univ_four] <;> ring

lemma starMatrix_mulVec (x : ℍ) : starMatrix *ᵥ coords x = coords (star x) := by
  ext i
  fin_cases i <;> simp [starMatrix, coords, mulVec_diagonal]

/-!

### B.3. Multiplicativity

-/

lemma leftMulMatrix_mul (a b : ℍ) :
    leftMulMatrix (a * b) = leftMulMatrix a * leftMulMatrix b := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [leftMulMatrix, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

/-- Right multiplication reverses the order of products. -/
lemma rightMulMatrix_mul (a b : ℍ) :
    rightMulMatrix (a * b) = rightMulMatrix b * rightMulMatrix a := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rightMulMatrix, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

/-- Left and right multiplication commute, by associativity of the quaternions. -/
lemma leftMulMatrix_mul_rightMulMatrix_comm (p q : ℍ) :
    leftMulMatrix p * rightMulMatrix q = rightMulMatrix q * leftMulMatrix p := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [leftMulMatrix, rightMulMatrix, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

@[simp]
lemma leftMulMatrix_one : leftMulMatrix 1 = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [leftMulMatrix]

@[simp]
lemma rightMulMatrix_one : rightMulMatrix 1 = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rightMulMatrix]

/-!

### B.4. Orthogonality and determinant

-/

lemma leftMulMatrix_transpose_mul_self (p : ℍ) :
    (leftMulMatrix p)ᵀ * leftMulMatrix p = normSq p • (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [leftMulMatrix, normSq_def', Matrix.mul_apply, Fin.sum_univ_four] <;> ring

lemma rightMulMatrix_transpose_mul_self (q : ℍ) :
    (rightMulMatrix q)ᵀ * rightMulMatrix q = normSq q • (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rightMulMatrix, normSq_def', Matrix.mul_apply, Fin.sum_univ_four] <;> ring

/-- The determinant of left multiplication is `‖p‖⁴`, in particular positive: left
multiplication by a unit quaternion preserves orientation. -/
lemma det_leftMulMatrix (p : ℍ) : (leftMulMatrix p).det = normSq p ^ 2 := by
  simp [leftMulMatrix, normSq_def', det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  ring

lemma det_rightMulMatrix (q : ℍ) : (rightMulMatrix q).det = normSq q ^ 2 := by
  simp [rightMulMatrix, normSq_def', det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  ring

/-- Quaternion conjugation reverses orientation. -/
lemma det_starMatrix : starMatrix.det = -1 := by
  simp [starMatrix, det_diagonal, Fin.prod_univ_four]

lemma normSq_eq_one_of_norm {p : ℍ} (hp : ‖p‖ = 1) : normSq p = 1 := by
  rw [normSq_eq_norm_mul_self, hp, mul_one]

/-!

## C. Surjectivity

-/

/-!

### C.1. The isometry of an orthogonal matrix

-/

/-- The linear isometry of `ℍ` given by an orthogonal `4 × 4` matrix acting on coordinates. -/
noncomputable def isometryOfOrthogonal (R : Matrix (Fin 4) (Fin 4) ℝ) (hR : Rᵀ * R = 1) :
    ℍ ≃ₗᵢ[ℝ] ℍ :=
  LinearIsometry.toLinearIsometryEquiv
    { toFun := fun x => ofCoords (R *ᵥ coords x)
      map_add' := fun x y => by rw [coords_add, mulVec_add, ofCoords_add]
      map_smul' := fun r x => by rw [coords_smul, mulVec_smul, ofCoords_smul]; rfl
      norm_map' := fun x => by
        have hsq : ‖ofCoords (R *ᵥ coords x)‖ * ‖ofCoords (R *ᵥ coords x)‖ = ‖x‖ * ‖x‖ := by
          rw [← normSq_eq_norm_mul_self, ← normSq_eq_norm_mul_self, normSq_eq_coords_dotProduct,
            coords_ofCoords, dotProduct_mulVec_orthogonal hR, normSq_eq_coords_dotProduct]
        exact (mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)).mp hsq }
    rfl

lemma coords_isometryOfOrthogonal (R : Matrix (Fin 4) (Fin 4) ℝ) (hR : Rᵀ * R = 1) (x : ℍ) :
    coords (isometryOfOrthogonal R hR x) = R *ᵥ coords x :=
  coords_ofCoords _

/-- Two matrices that act in the same way on the coordinates of every quaternion are equal. -/
lemma eq_of_mulVec_coords {A B : Matrix (Fin 4) (Fin 4) ℝ}
    (h : ∀ x : ℍ, A *ᵥ coords x = B *ᵥ coords x) : A = B := by
  apply mulVec_injective
  funext w
  simpa using h (ofCoords w)

/-!

### C.2. Every rotation is a two-sided multiplication

-/

/-- Every special orthogonal `4 × 4` real matrix is the matrix of `x ↦ p * x * q` for unit
quaternions `p` and `q`. The other possibility for an isometry, `x ↦ p * star x * q`, has
determinant `-1`. -/
lemma exists_leftMulMatrix_mul_rightMulMatrix (R : Matrix (Fin 4) (Fin 4) ℝ)
    (hR : Rᵀ * R = 1) (hdet : R.det = 1) :
    ∃ p q : ℍ, ‖p‖ = 1 ∧ ‖q‖ = 1 ∧ R = leftMulMatrix p * rightMulMatrix q := by
  obtain ⟨p, q, hp, hq, h⟩ := isUnitTwoSided_linearIsometryEquiv (isometryOfOrthogonal R hR)
  rcases h with h | h
  · refine ⟨p, q, hp, hq, eq_of_mulVec_coords fun x => ?_⟩
    rw [← coords_isometryOfOrthogonal R hR, h, ← mulVec_mulVec, rightMulMatrix_mulVec,
      leftMulMatrix_mulVec, mul_assoc]
  · exfalso
    have hRm : R = leftMulMatrix p * rightMulMatrix q * starMatrix :=
      eq_of_mulVec_coords fun x => by
        rw [← coords_isometryOfOrthogonal R hR, h, ← mulVec_mulVec, ← mulVec_mulVec,
          starMatrix_mulVec, rightMulMatrix_mulVec, leftMulMatrix_mulVec, mul_assoc]
    rw [hRm, det_mul, det_mul, det_leftMulMatrix, det_rightMulMatrix,
      normSq_eq_one_of_norm hp, normSq_eq_one_of_norm hq, det_starMatrix] at hdet
    norm_num at hdet

/-!

## D. The kernel

-/

/-- If `x ↦ p * x * q` is the identity and `p` is a unit quaternion, then `p = q = 1` or
`p = q = -1`. Multiplying by the transpose of `leftMulMatrix p` gives
`rightMulMatrix q = (leftMulMatrix p)ᵀ`, and comparing entries forces the imaginary parts of
`p` and `q` to vanish. -/
lemma leftMulMatrix_mul_rightMulMatrix_eq_one {p q : ℍ} (hp : ‖p‖ = 1)
    (h : leftMulMatrix p * rightMulMatrix q = 1) : (p = 1 ∧ q = 1) ∨ (p = -1 ∧ q = -1) := by
  have h2 : rightMulMatrix q = (leftMulMatrix p)ᵀ := by
    have h3 := congrArg (fun M => (leftMulMatrix p)ᵀ * M) h
    simp only [← Matrix.mul_assoc, leftMulMatrix_transpose_mul_self, normSq_eq_one_of_norm hp,
      one_smul, Matrix.one_mul, Matrix.mul_one] at h3
    exact h3
  have a00 := congrFun (congrFun h2 0) 0
  have a10 := congrFun (congrFun h2 1) 0
  have a20 := congrFun (congrFun h2 2) 0
  have a30 := congrFun (congrFun h2 3) 0
  have a12 := congrFun (congrFun h2 1) 2
  have a23 := congrFun (congrFun h2 2) 3
  have a31 := congrFun (congrFun h2 3) 1
  simp [rightMulMatrix, leftMulMatrix, transpose_apply] at a00 a10 a20 a30 a12 a23 a31
  have hn := normSq_eq_one_of_norm hp
  rw [normSq_def'] at hn
  have p1 : p.imI = 0 := by linarith
  have p2 : p.imJ = 0 := by linarith
  have p3 : p.imK = 0 := by linarith
  have hp0 : (p.re - 1) * (p.re + 1) = 0 := by
    simp only [p1, p2, p3] at hn; linarith
  rcases mul_eq_zero.mp hp0 with h0 | h0
  · left
    constructor <;> ext <;> simp <;> linarith
  · right
    constructor <;> ext <;> simp <;> linarith

/-!

## E. The covering homomorphism

-/

/-!

### E.1. Unit quaternions

-/

lemma eq_one_of_mul_self_eq_one {t : ℝ} (h0 : 0 ≤ t) (h : t * t = 1) : t = 1 := by
  have h1 : (t - 1) * (t + 1) = 0 := by linarith
  rcases mul_eq_zero.mp h1 with h2 | h2 <;> linarith

/-- An element of `unitary ℍ` has norm one. -/
lemma norm_coe_unitary (p : unitary ℍ) : ‖(p : ℍ)‖ = 1 := by
  have h := (Unitary.mem_iff.mp p.2).2
  rw [self_mul_star] at h
  have h2 : normSq (p : ℍ) = 1 := by simpa using congrArg (fun z : ℍ => z.re) h
  rw [normSq_eq_norm_mul_self] at h2
  exact eq_one_of_mul_self_eq_one (norm_nonneg _) h2

/-- A quaternion of norm one is an element of `unitary ℍ`. -/
lemma mem_unitary_of_norm_eq_one {a : ℍ} (h : ‖a‖ = 1) : a ∈ unitary ℍ := by
  rw [Unitary.mem_iff, star_mul_self, self_mul_star, normSq_eq_one_of_norm h]
  simp

/-!

### E.2. The homomorphism

-/

/-- The matrix of `x ↦ p * x * q` is special orthogonal for unit quaternions `p` and `q`. -/
lemma leftMulMatrix_mul_rightMulMatrix_mem {p q : ℍ} (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) :
    leftMulMatrix p * rightMulMatrix q ∈ specialOrthogonalGroup (Fin 4) ℝ := by
  rw [mem_specialOrthogonalGroup_iff, mem_orthogonalGroup_iff']
  constructor
  · rw [transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc (leftMulMatrix p)ᵀ,
      leftMulMatrix_transpose_mul_self, normSq_eq_one_of_norm hp, one_smul, Matrix.one_mul,
      rightMulMatrix_transpose_mul_self, normSq_eq_one_of_norm hq, one_smul]
  · rw [det_mul, det_leftMulMatrix, det_rightMulMatrix, normSq_eq_one_of_norm hp,
      normSq_eq_one_of_norm hq]
    norm_num

/-- The group homomorphism `Sp(1) × Sp(1) →* SO(4)` sending `(p, q)` to the rotation
`x ↦ p * x * star q`. The `star` on the right factor makes the map a homomorphism. -/
def quaternionCover : unitary ℍ × unitary ℍ →* specialOrthogonalGroup (Fin 4) ℝ where
  toFun x := ⟨leftMulMatrix (x.1 : ℍ) * rightMulMatrix (star (x.2 : ℍ)),
    leftMulMatrix_mul_rightMulMatrix_mem (norm_coe_unitary x.1)
      (by rw [Quaternion.norm_star]; exact norm_coe_unitary x.2)⟩
  map_one' := by
    apply Subtype.ext
    simp
  map_mul' x y := by
    apply Subtype.ext
    change leftMulMatrix ((x.1 : ℍ) * (y.1 : ℍ)) * rightMulMatrix (star ((x.2 : ℍ) * (y.2 : ℍ)))
      = (leftMulMatrix (x.1 : ℍ) * rightMulMatrix (star (x.2 : ℍ)))
        * (leftMulMatrix (y.1 : ℍ) * rightMulMatrix (star (y.2 : ℍ)))
    rw [star_mul, leftMulMatrix_mul, rightMulMatrix_mul, Matrix.mul_assoc, Matrix.mul_assoc,
      ← Matrix.mul_assoc (leftMulMatrix (y.1 : ℍ)), leftMulMatrix_mul_rightMulMatrix_comm,
      Matrix.mul_assoc]

lemma coe_quaternionCover (x : unitary ℍ × unitary ℍ) :
    (quaternionCover x : Matrix (Fin 4) (Fin 4) ℝ)
      = leftMulMatrix (x.1 : ℍ) * rightMulMatrix (star (x.2 : ℍ)) :=
  rfl

/-!

### E.3. Surjectivity, kernel and the isomorphism

-/

/-- Every element of `SO(4)` comes from a pair of unit quaternions. -/
lemma quaternionCover_surjective : Function.Surjective quaternionCover := by
  intro A
  have hA := mem_specialOrthogonalGroup_iff.mp A.2
  have hR : (A : Matrix (Fin 4) (Fin 4) ℝ)ᵀ * (A : Matrix (Fin 4) (Fin 4) ℝ) = 1 := by
    have h := hA.1
    rw [mem_orthogonalGroup_iff'] at h
    exact h
  obtain ⟨p, q, hp, hq, h⟩ := exists_leftMulMatrix_mul_rightMulMatrix A.1 hR hA.2
  refine ⟨(⟨p, mem_unitary_of_norm_eq_one hp⟩,
    ⟨star q, mem_unitary_of_norm_eq_one (by rw [Quaternion.norm_star]; exact hq)⟩), ?_⟩
  apply Subtype.ext
  rw [coe_quaternionCover]
  simp only [star_star]
  exact h.symm

/-- A pair of unit quaternions gives the identity rotation exactly when it is `(1, 1)` or
`(-1, -1)`. -/
lemma quaternionCover_eq_one_iff (x : unitary ℍ × unitary ℍ) :
    quaternionCover x = 1 ↔
      ((x.1 : ℍ) = 1 ∧ (x.2 : ℍ) = 1) ∨ ((x.1 : ℍ) = -1 ∧ (x.2 : ℍ) = -1) := by
  constructor
  · intro h
    have h1 : leftMulMatrix (x.1 : ℍ) * rightMulMatrix (star (x.2 : ℍ)) = 1 := by
      rw [← coe_quaternionCover, h]; rfl
    rcases leftMulMatrix_mul_rightMulMatrix_eq_one (norm_coe_unitary x.1) h1 with
      ⟨ha, hb⟩ | ⟨ha, hb⟩
    · exact Or.inl ⟨ha, by simpa using congrArg star hb⟩
    · exact Or.inr ⟨ha, by simpa using congrArg star hb⟩
  · rintro (⟨ha, hb⟩ | ⟨ha, hb⟩) <;>
    · apply Subtype.ext
      rw [coe_quaternionCover, ha, hb]
      ext i j
      fin_cases i <;> fin_cases j <;>
        simp [leftMulMatrix, rightMulMatrix, Matrix.mul_apply, Fin.sum_univ_four]

/-- The isomorphism `(Sp(1) × Sp(1)) / {(1, 1), (-1, -1)} ≃* SO(4)`. -/
noncomputable def quaternionCoverEquiv :
    (unitary ℍ × unitary ℍ) ⧸ quaternionCover.ker ≃* specialOrthogonalGroup (Fin 4) ℝ :=
  QuotientGroup.quotientKerEquivOfSurjective quaternionCover quaternionCover_surjective

end SO4
