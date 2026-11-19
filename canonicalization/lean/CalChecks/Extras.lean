/-
Second batch: Bell correlations, the Killing form of gl(2), and the Clifford algebra
Cl(3,0) realised on the Pauli matrices (via Mathlib's `CliffordAlgebra`).
-/
import Mathlib
import CalChecks.Basic

open Matrix Complex
open scoped ComplexConjugate Kronecker

namespace CalChecks

/-! ## Bell / CHSH: the joint closure and E(a,b) = -a·b -/

/-- `a·σ` for a real 3-vector. -/
def dotσ (a : Fin 3 → ℝ) : M2 := (a 0 : ℂ) • σx + (a 1 : ℂ) • σy + (a 2 : ℂ) • σz

/-- Joint closure `ϱ = ¼ (1⊗1 + Σ_j e_j ⊗ e_j)`. -/
noncomputable def jointClosure : Matrix (Fin 2 × Fin 2) (Fin 2 × Fin 2) ℂ :=
  (1 / 4 : ℂ) • ((1 : M2) ⊗ₖ (1 : M2) + e1 ⊗ₖ e1 + e2 ⊗ₖ e2 + e3 ⊗ₖ e3)

theorem dotσ_eq (a : Fin 3 → ℝ) :
    dotσ a = !![(a 2 : ℂ), a 0 - I * a 1; a 0 + I * a 1, -(a 2 : ℂ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [dotσ, σx, σy, σz] <;> ring

theorem jointClosure_trace : jointClosure.trace = 1 := by
  simp [jointClosure, Matrix.trace, Fintype.sum_prod_type, Fin.sum_univ_two,
    e1, e2, e3, σx, σy, σz]
  norm_num

/-- `E(a,b) = tr(ϱ · a·σ ⊗ b·σ) = -a·b`. -/
theorem bell_correlation (a b : Fin 3 → ℝ) :
    (jointClosure * (dotσ a ⊗ₖ dotσ b)).trace
      = ((-(a 0 * b 0 + a 1 * b 1 + a 2 * b 2) : ℝ) : ℂ) := by
  rw [dotσ_eq, dotσ_eq]
  simp [jointClosure, Matrix.trace, Matrix.mul_apply, Fintype.sum_prod_type, Fin.sum_univ_two,
    e1, e2, e3, σx, σy, σz]
  ring_nf
  simp [Complex.I_sq]
  ring

/-- Each marginal is a fair coin. -/
theorem bell_marginal_left (a : Fin 3 → ℝ) :
    (jointClosure * (dotσ a ⊗ₖ (1 : M2))).trace = 0 := by
  rw [dotσ_eq]
  simp [jointClosure, Matrix.trace, Matrix.mul_apply, Fintype.sum_prod_type, Fin.sum_univ_two,
    e1, e2, e3, σx, σy, σz]

/-! ## Killing form of gl(2) -/

/-- `ad_A` on `M₂(ℂ)`. -/
noncomputable def adM (A : M2) : M2 →ₗ[ℂ] M2 := LinearMap.mulLeft ℂ A - LinearMap.mulRight ℂ A

theorem adM_apply (A X : M2) : adM A X = A * X - X * A := by simp [adM]

theorem stdBasis_repr (M : M2) (i j : Fin 2) :
    (Matrix.stdBasis ℂ (Fin 2) (Fin 2)).repr M (i, j) = M i j := by
  simp [Matrix.stdBasis]

/-- `κ(A,B) = tr(ad_A ad_B) = 4 tr(AB) - 2 tr A tr B`. -/
theorem killing_gl2 (A B : M2) :
    LinearMap.trace ℂ M2 (adM A ∘ₗ adM B) = 4 * (A * B).trace - 2 * A.trace * B.trace := by
  rw [LinearMap.trace_eq_matrix_trace ℂ (Matrix.stdBasis ℂ (Fin 2) (Fin 2))]
  simp [Matrix.trace, LinearMap.toMatrix_apply, Fintype.sum_prod_type, Fin.sum_univ_two,
    adM_apply, Matrix.mul_apply, Matrix.stdBasis_eq_single, Matrix.single_apply, stdBasis_repr]
  ring

/-- The centre drops out of the Killing form. -/
theorem killing_centre (c : ℂ) (B : M2) :
    LinearMap.trace ℂ M2 (adM (c • (1 : M2)) ∘ₗ adM B) = 0 := by
  rw [killing_gl2]
  simp [trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two]
  ring

/-- On `su(2)`: `det A = -(1/8) κ(A,A)`. -/
theorem det_su2_killing (y1 y2 y3 : ℝ) :
    (gen 0 y1 y2 y3).det
      = -(1 / 8) * LinearMap.trace ℂ M2 (adM (gen 0 y1 y2 y3) ∘ₗ adM (gen 0 y1 y2 y3)) := by
  have htr : (gen 0 y1 y2 y3).trace = 0 := by
    simp [gen, e1, e2, e3, σx, σy, σz, trace_fin_two]
  rw [killing_gl2, htr, det_su2]
  ring

/-! ## Cl(3,0) on the Pauli matrices, through Mathlib's `CliffordAlgebra` -/

/-- Euclidean quadratic form on ℝ³. -/
noncomputable def Q3 : QuadraticForm ℝ (Fin 3 → ℝ) :=
  QuadraticMap.weightedSumSquares ℝ (fun _ : Fin 3 => (1 : ℝ))

/-- `v ↦ v·σ` as an ℝ-linear map. -/
noncomputable def pauliVec : (Fin 3 → ℝ) →ₗ[ℝ] M2 where
  toFun v := dotσ v
  map_add' v w := by
    simp only [dotσ, Pi.add_apply, Complex.ofReal_add, add_smul]; abel
  map_smul' r v := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [dotσ, σx, σy, σz, Matrix.smul_apply, Complex.real_smul] <;> ring

/-- Clifford relation: `(v·σ)² = |v|²`. -/
theorem pauliVec_sq (v : Fin 3 → ℝ) :
    pauliVec v * pauliVec v = algebraMap ℝ M2 (Q3 v) := by
  show dotσ v * dotσ v = _
  rw [dotσ_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Q3, QuadraticMap.weightedSumSquares_apply, Fin.sum_univ_three, Matrix.mul_apply,
      Fin.sum_univ_two, Matrix.algebraMap_matrix_apply] <;>
    ring_nf <;> simp [Complex.I_sq] <;> ring

/-- The algebra map `Cl(3,0) → M₂(ℂ)` sending the generators to the Pauli matrices. -/
noncomputable def cliffordToM2 : CliffordAlgebra Q3 →ₐ[ℝ] M2 :=
  CliffordAlgebra.lift Q3 ⟨pauliVec, pauliVec_sq⟩

theorem cliffordToM2_ι (v : Fin 3 → ℝ) :
    cliffordToM2 (CliffordAlgebra.ι Q3 v) = dotσ v :=
  CliffordAlgebra.lift_ι_apply _ _ v

/-- Paravector norm: `(t + x·σ)(t - x·σ) = t² - |x|²` (Minkowski form). -/
theorem paravector_norm (t x y z : ℝ) :
    herm t x y z * herm t (-x) (-y) (-z) = ((t ^ 2 - x ^ 2 - y ^ 2 - z ^ 2 : ℝ) : ℂ) • (1 : M2) := by
  rw [herm_eq, herm_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;> simp [Complex.I_sq] <;> ring

end CalChecks
