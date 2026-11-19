/-
Seventh batch: last loose items.
-/
import Mathlib
import CalChecks.Clifford

open Matrix Complex
open scoped ComplexConjugate

namespace CalChecks

/-- `tr X = 2t`: the trace is the time component. -/
theorem trace_herm (t x y z : ℝ) : (herm t x y z).trace = ((2 * t : ℝ) : ℂ) := by
  rw [herm_eq]; simp [trace_fin_two]; ring

/-- Under `SU(2) ≅ Sp(1)` conjugation the channel index splits `1 ⊕ 3`:
`e₀` is a singlet (scaled by `|p|²`) and `(e₁,e₂,e₃)` rotate by `Rot p`. -/
theorem conj_splits_one_three (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    carrier (fun k => (p k : ℂ)) * carrier Z * (carrier (fun k => (p k : ℂ)))ᴴ
      = carrier ![((p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 : ℝ) : ℂ) * Z 0,
          (Rot p 0 0 : ℂ) * Z 1 + (Rot p 0 1 : ℂ) * Z 2 + (Rot p 0 2 : ℂ) * Z 3,
          (Rot p 1 0 : ℂ) * Z 1 + (Rot p 1 1 : ℂ) * Z 2 + (Rot p 1 2 : ℂ) * Z 3,
          (Rot p 2 0 : ℂ) * Z 1 + (Rot p 2 1 : ℂ) * Z 2 + (Rot p 2 2 : ℂ) * Z 3] := by
  rw [carrier_eq, carrier_eq, carrier_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Rot, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply] <;>
    ring_nf <;> simp [Complex.I_sq] <;> cring

/-- Purity is boost dependent: a boost moves the maximally mixed closure off purity 1/2. -/
theorem purity_boost_dependent :
    ∃ G : M2, G.det = 1 ∧
      ((G * 1 * Gᴴ) * (G * 1 * Gᴴ)).trace / (G * 1 * Gᴴ).trace ^ 2
        ≠ ((1 : M2) * 1).trace / (1 : M2).trace ^ 2 := by
  refine ⟨!![2, 0; 0, 1 / 2], by simp [det_fin_two_of], ?_⟩
  have hX : (!![2, 0; 0, 1 / 2] : M2) * 1 * (!![2, 0; 0, 1 / 2] : M2)ᴴ = !![4, 0; 0, 1 / 4] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply, map_ofNat] <;> norm_num
  rw [hX]
  simp [trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two]
  norm_num

end CalChecks
