/-
Agent C2: the three-way identity  det X = (tr X)^2/4 + (1/8) kappa(i X0, i X0)
for Hermitian 2x2 complex X, X0 = X - (tr X / 2) 1.

Two versions, differing only in which Killing form is used.

* `threeway_gl2`  : kappa = Mathlib's `killingForm ℂ M2` (gl(2,C) as a complex Lie algebra),
  evaluated through the project's theorem `killingForm_gl2`.  No Hermitian hypothesis needed.
* `threeway_su2`  : kappa = Mathlib's `killingForm ℝ su2`, where `su2` is defined here as the
  real Lie subalgebra of skew-Hermitian traceless matrices (3-dimensional).  Its value is
  computed from `tr (ad ∘ ad)` in a basis (`killing_coords`), not assumed.

Standalone file: not imported anywhere.
-/
import Mathlib
import CalChecks.Structure

open Matrix Complex
open scoped ComplexConjugate

namespace CalChecks.C2

open CalChecks

attribute [local instance] LieRing.ofAssociativeRing LieAlgebra.ofAssociativeAlgebra

/-! ## Version 1: Killing form of gl(2,C) over C (project definition) -/

/-- Three-way identity with Mathlib's complex Killing form on `gl(2,ℂ)`.
Holds for every complex 2x2 matrix. -/
theorem threeway_gl2 (X : M2) :
    X.det = X.trace ^ 2 / 4
      + (1 / 8) * killingForm ℂ M2 (I • (X - (X.trace / 2) • (1 : M2)))
          (I • (X - (X.trace / 2) • (1 : M2))) := by
  rw [killingForm_gl2]
  simp [det_fin_two, trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two]
  linear_combination (-((X 0 0 - X 1 1) ^ 2 / 4 + X 0 1 * X 1 0)) * I_sq

/-- The Hermitian case, as stated in the claim. -/
theorem threeway_gl2_hermitian (X : M2) (_hX : X.IsHermitian) :
    X.det = X.trace ^ 2 / 4
      + (1 / 8) * killingForm ℂ M2 (I • (X - (X.trace / 2) • (1 : M2)))
          (I • (X - (X.trace / 2) • (1 : M2))) :=
  threeway_gl2 X

/-! ## Version 2: the genuine real Lie algebra su(2) -/

/-- `su(2)`: skew-Hermitian traceless 2x2 complex matrices, as a real Lie subalgebra. -/
def su2 : LieSubalgebra ℝ M2 :=
  { carrier := {A | Aᴴ = -A ∧ A.trace = 0}
    add_mem' := by
      rintro A B ⟨hA, hA'⟩ ⟨hB, hB'⟩
      refine ⟨?_, ?_⟩
      · rw [conjTranspose_add, hA, hB]; abel
      · rw [trace_add, hA', hB', add_zero]
    zero_mem' := by simp
    smul_mem' := by
      rintro c A ⟨hA, hA'⟩
      refine ⟨?_, ?_⟩
      · rw [conjTranspose_smul, hA, star_trivial, smul_neg]
      · rw [trace_smul, hA', smul_zero]
    lie_mem' := by
      rintro A B ⟨hA, hA'⟩ ⟨hB, hB'⟩
      refine ⟨?_, ?_⟩
      · rw [Ring.lie_def, conjTranspose_sub, conjTranspose_mul, conjTranspose_mul, hA, hB]
        simp
      · rw [Ring.lie_def, trace_sub, trace_mul_comm, sub_self] }

theorem mem_su2 {A : M2} : A ∈ su2 ↔ Aᴴ = -A ∧ A.trace = 0 := Iff.rfl

/-- The element `y₁ e₁ + y₂ e₂ + y₃ e₃`, `e_j = -i σ_j`, written out. -/
def mat (y : Fin 3 → ℝ) : M2 :=
  !![-(y 2 : ℂ) * I, -(y 0 : ℂ) * I - (y 1 : ℂ); -(y 0 : ℂ) * I + (y 1 : ℂ), (y 2 : ℂ) * I]

theorem mat_mem (y : Fin 3 → ℝ) : mat y ∈ su2 := by
  refine mem_su2.mpr ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [mat, conjTranspose_apply] <;> try ring
  · simp [mat, trace_fin_two]

/-- Real coordinates on `su(2)`. -/
noncomputable def coords : su2 ≃ₗ[ℝ] (Fin 3 → ℝ) where
  toFun A := ![-((A : M2) 1 0).im, ((A : M2) 1 0).re, -((A : M2) 0 0).im]
  map_add' A B := by
    ext i
    fin_cases i <;> simp <;> try ring
  map_smul' c A := by
    ext i
    fin_cases i <;> simp <;> try ring
  invFun y := ⟨mat y, mat_mem y⟩
  left_inv := by
    rintro ⟨A, hA⟩
    obtain ⟨hA1, hA2⟩ := mem_su2.mp hA
    have h00 : conj (A 0 0) = -A 0 0 := by
      simpa [conjTranspose_apply] using congrFun (congrFun hA1 0) 0
    have h01 : conj (A 1 0) = -A 0 1 := by
      simpa [conjTranspose_apply] using congrFun (congrFun hA1 0) 1
    have h11 : A 1 1 = -A 0 0 := by
      rw [trace_fin_two] at hA2; linear_combination hA2
    have r00 : (A 0 0).re = 0 := by
      have := congrArg Complex.re h00
      simp at this
      linarith
    have r01 : (A 0 1).re = -(A 1 0).re := by
      have := congrArg Complex.re h01
      simp at this
      linarith
    have i01 : (A 0 1).im = (A 1 0).im := by
      have := congrArg Complex.im h01
      simp at this
      linarith
    apply Subtype.ext
    ext i j
    fin_cases i <;> fin_cases j <;> apply Complex.ext <;> simp [mat, h11, r00, r01, i01]
  right_inv y := by
    ext i
    fin_cases i <;> simp [mat]

theorem coords_apply (A : su2) :
    coords A = ![-((A : M2) 1 0).im, ((A : M2) 1 0).re, -((A : M2) 0 0).im] := rfl

theorem coords_symm_coe (y : Fin 3 → ℝ) : ((coords.symm y : su2) : M2) = mat y := rfl

/-- The basis `e₁, e₂, e₃` of `su(2)`. -/
noncomputable def bas : Module.Basis (Fin 3) ℝ su2 := Module.Basis.ofEquivFun coords

/-- Killing form of `su(2)` from its definition `tr(ad ∘ ad)`, in the basis `bas`:
`κ(y, z) = -8 y·z`. -/
theorem killing_coords (y z : Fin 3 → ℝ) :
    killingForm ℝ su2 (coords.symm y) (coords.symm z)
      = -8 * (y 0 * z 0 + y 1 * z 1 + y 2 * z 2) := by
  rw [killingForm_apply_apply, LinearMap.trace_eq_matrix_trace ℝ bas]
  simp [Matrix.trace, Fin.sum_univ_three, LinearMap.toMatrix_apply, bas, coords_apply,
    coords_symm_coe, mat, Ring.lie_def, Matrix.mul_apply, Fin.sum_univ_two, Pi.single_apply]
  ring

/-- Consequence (derived, not assumed): `κ(A,B) = 4 Re tr(AB)` on `su(2)`. -/
theorem killing_su2 (A B : su2) :
    killingForm ℝ su2 A B = 4 * (((A : M2) * (B : M2)).trace).re := by
  obtain ⟨y, rfl⟩ := coords.symm.surjective A
  obtain ⟨z, rfl⟩ := coords.symm.surjective B
  rw [killing_coords]
  simp [coords_symm_coe, mat, trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two]
  ring

theorem killing_su2_mk (M : M2) (h : M ∈ su2) :
    killingForm ℝ su2 ⟨M, h⟩ ⟨M, h⟩ = 4 * ((M * M).trace).re :=
  killing_su2 ⟨M, h⟩ ⟨M, h⟩

theorem trace_sq_real (M : M2) (h : M ∈ su2) : (((M * M).trace).re : ℂ) = (M * M).trace := by
  have hM := (mem_su2.mp h).1
  have hs : star ((M * M).trace) = (M * M).trace := by
    rw [← trace_conjTranspose, conjTranspose_mul, hM, neg_mul_neg]
  exact Complex.conj_eq_iff_re.mp hs

/-- For Hermitian `X`, `i X₀` lies in `su(2)`. -/
theorem I_smul_traceless_mem (X : M2) (hX : X.IsHermitian) :
    I • (X - (X.trace / 2) • (1 : M2)) ∈ su2 := by
  have ht : star X.trace = X.trace := by rw [← trace_conjTranspose, hX.eq]
  have h2 : star (X.trace / 2) = X.trace / 2 := by
    rw [star_div₀, ht]; simp
  refine mem_su2.mpr ⟨?_, ?_⟩
  · rw [conjTranspose_smul, conjTranspose_sub, conjTranspose_smul, conjTranspose_one, hX.eq, h2]
    simp
  · rw [trace_smul, trace_sub, trace_smul, trace_one, Fintype.card_fin]
    simp only [smul_eq_mul]
    push_cast
    ring

/-- **Three-way identity, part 1 of the claim.**  For Hermitian `X`, with `κ` the Killing
form of the real Lie algebra `su(2)` (Mathlib's `killingForm`, i.e. `tr(ad ∘ ad)`):
`det X = (tr X)²/4 + (1/8) κ(i X₀, i X₀)`. -/
theorem threeway_su2 (X : M2) (hX : X.IsHermitian) :
    X.det = X.trace ^ 2 / 4
      + (1 / 8 : ℂ) * ((killingForm ℝ su2
          ⟨I • (X - (X.trace / 2) • (1 : M2)), I_smul_traceless_mem X hX⟩
          ⟨I • (X - (X.trace / 2) • (1 : M2)), I_smul_traceless_mem X hX⟩ : ℝ) : ℂ) := by
  have h := I_smul_traceless_mem X hX
  have key := threeway_gl2 X
  rw [killingForm_gl2] at key
  have htr0 : (I • (X - (X.trace / 2) • (1 : M2))).trace = 0 := (mem_su2.mp h).2
  rw [killing_su2_mk, key, htr0]
  push_cast
  rw [trace_sq_real _ h]
  ring

end CalChecks.C2

#print axioms CalChecks.C2.threeway_gl2
#print axioms CalChecks.C2.killing_coords
#print axioms CalChecks.C2.killing_su2
#print axioms CalChecks.C2.threeway_su2
