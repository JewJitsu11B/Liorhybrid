/-
Fifth batch: structural results.
* kernel of `SL(2,ℂ) → SO⁺(1,3)` is `{±1}`
* Mathlib's own `killingForm` on `gl(2,ℂ)`, over ℂ and over ℝ (`= 2 Re κ_ℂ`)
* the positive cone is one congruence orbit, with stabiliser `U(2)`
-/
import Mathlib
import CalChecks.Complete

open Matrix Complex
open scoped ComplexConjugate ComplexOrder MatrixOrder

namespace CalChecks

attribute [local instance] LieRing.ofAssociativeRing LieAlgebra.ofAssociativeAlgebra

/-! ## Kernel of the spinor map -/

/-- If `G X G† = X` for every Hermitian `X` and `det G = 1`, then `G = ±1`. -/
theorem spinor_kernel (G : M2) (hdet : G.det = 1)
    (h : ∀ t x y z : ℝ, G * herm t x y z * Gᴴ = herm t x y z) : G = 1 ∨ G = -1 := by
  have h1 : G * Gᴴ = 1 := by
    have := h 1 0 0 0
    simpa [herm] using this
  have h1' : Gᴴ * G = 1 := mul_eq_one_comm.mp h1
  have comm : ∀ t x y z : ℝ, G * herm t x y z = herm t x y z * G := by
    intro t x y z
    have := congrArg (fun M => M * G) (h t x y z)
    simp only [Matrix.mul_assoc, h1', Matrix.mul_one] at this
    exact this
  have hz := comm 0 0 0 1
  have hx := comm 0 1 0 0
  rw [herm_eq] at hz hx
  have z01 := congrFun (congrFun hz 0) 1
  have z10 := congrFun (congrFun hz 1) 0
  have x00 := congrFun (congrFun hx 0) 0
  have x01 := congrFun (congrFun hx 0) 1
  simp [Matrix.mul_apply, Fin.sum_univ_two] at z01 z10 x00 x01
  have g01 : G 0 1 = 0 := by
    have : (2 : ℂ) * G 0 1 = 0 := by linear_combination -z01
    simpa using this
  have g10 : G 1 0 = 0 := by
    have : (2 : ℂ) * G 1 0 = 0 := by linear_combination z10
    simpa using this
  have g11 : G 1 1 = G 0 0 := x01.symm
  have hd : G 0 0 * G 0 0 = 1 := by
    rw [det_fin_two, g01, g10, g11] at hdet
    simpa using hdet
  have hcases : G 0 0 = 1 ∨ G 0 0 = -1 := mul_self_eq_one_iff.mp hd
  rcases hcases with h0 | h0
  · left
    ext i j
    fin_cases i <;> fin_cases j <;> simp [g01, g10, g11, h0]
  · right
    ext i j
    fin_cases i <;> fin_cases j <;> simp [g01, g10, g11, h0]

/-- Conversely `±1` act trivially. -/
theorem spinor_kernel_converse (X : M2) : (-1 : M2) * X * (-1 : M2)ᴴ = X := by
  simp

/-! ## Mathlib's Killing form on `gl(2,ℂ)` -/

theorem ad_eq_adM (A : M2) : LieAlgebra.ad ℂ M2 A = adM A := by
  ext X : 1
  simp [adM, Ring.lie_def]

/-- Mathlib's `killingForm` on `gl(2,ℂ)` is `4 tr(AB) - 2 tr A tr B`. -/
theorem killingForm_gl2 (A B : M2) :
    killingForm ℂ M2 A B = 4 * (A * B).trace - 2 * A.trace * B.trace := by
  rw [killingForm_apply_apply, ad_eq_adM, ad_eq_adM, killing_gl2]

/-- Real trace of a complex-linear endomorphism of `M₂(ℂ)` is twice the real part. -/
theorem trace_real (f : M2 →ₗ[ℂ] M2) :
    LinearMap.trace ℝ M2 (f.restrictScalars ℝ) = 2 * (LinearMap.trace ℂ M2 f).re := by
  rw [LinearMap.trace_eq_matrix_trace ℝ
      (Complex.basisOneI.smulTower (Matrix.stdBasis ℂ (Fin 2) (Fin 2))),
    LinearMap.trace_eq_matrix_trace ℂ (Matrix.stdBasis ℂ (Fin 2) (Fin 2))]
  simp [Matrix.trace, LinearMap.toMatrix_apply, Fintype.sum_prod_type, Fin.sum_univ_two,
    Module.Basis.smulTower_repr_mk, Module.Basis.smulTower_apply, Complex.coe_basisOneI,
    Complex.coe_basisOneI_repr]
  ring

/-- The Killing form of `gl(2,ℂ)` regarded as a real Lie algebra is `2 Re κ_ℂ`. -/
theorem killingForm_real (A B : M2) :
    killingForm ℝ M2 A B = 2 * (killingForm ℂ M2 A B).re := by
  rw [killingForm_apply_apply, killingForm_apply_apply, ← trace_real]
  congr 1

/-- Signature on the rotation block, now for the genuine real Killing form. -/
theorem killingForm_real_rotation (y1 y2 y3 : ℝ) :
    killingForm ℝ M2 (gen 0 y1 y2 y3) (gen 0 y1 y2 y3) = -16 * (y1 ^ 2 + y2 ^ 2 + y3 ^ 2) := by
  rw [killingForm_real, killingForm_apply_apply, ad_eq_adM, killingR_rotation]

theorem killingForm_real_boost (x y z : ℝ) :
    killingForm ℝ M2 (herm 0 x y z) (herm 0 x y z) = 16 * (x ^ 2 + y ^ 2 + z ^ 2) := by
  rw [killingForm_real, killingForm_apply_apply, ad_eq_adM, killingR_boost]

theorem killingForm_real_orthogonal (y1 y2 y3 x y z : ℝ) :
    killingForm ℝ M2 (gen 0 y1 y2 y3) (herm 0 x y z) = 0 := by
  rw [killingForm_real, killingForm_apply_apply, ad_eq_adM, ad_eq_adM, killingR_orthogonal]

/-! ## The positive cone is a single congruence orbit -/

/-- Every positive definite matrix is `G·1·G†` with `G` invertible: the interior of the
cone is the orbit `GL(2,ℂ)/U(2)` of the identity. -/
theorem cone_transitive (P : M2) (hP : P.PosDef) :
    ∃ G : M2, G.det ≠ 0 ∧ P = G * 1 * Gᴴ := by
  have h0 : (0 : M2) ≤ P := Matrix.nonneg_iff_posSemidef.mpr hP.posSemidef
  have hS : CFC.sqrt P * CFC.sqrt P = P := CFC.sqrt_mul_sqrt_self P h0
  have hSh : (CFC.sqrt P)ᴴ = CFC.sqrt P := (IsSelfAdjoint.of_nonneg (CFC.sqrt_nonneg P))
  refine ⟨CFC.sqrt P, ?_, ?_⟩
  · have hdet : P.det ≠ 0 := hP.det_pos.ne'
    rw [← hS, det_mul] at hdet
    exact left_ne_zero_of_mul hdet
  · rw [Matrix.mul_one, hSh, hS]

/-- Conversely every `G G†` with `G` invertible is positive definite. -/
theorem cone_orbit_posDef (G : M2) (hG : G.det ≠ 0) : (G * 1 * Gᴴ).PosDef := by
  rw [Matrix.mul_one]
  have hps : (G * Gᴴ).PosSemidef := Matrix.posSemidef_self_mul_conjTranspose G
  rw [hps.posDef_iff_det_ne_zero, det_mul, det_conjTranspose]
  exact mul_ne_zero hG (by simpa using hG)

end CalChecks
