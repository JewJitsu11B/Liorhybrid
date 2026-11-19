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

/-! ## The Born rule as a kinematic theorem

Everything below uses only the carrier `J` and its closure `J J†`: no equation of motion. -/

section BornKinematic
open scoped ComplexOrder

/-- The closure vanishes only for the zero carrier. -/
theorem closure_trace_ne_zero (J : M2) (hJ : J ≠ 0) : (J * Jᴴ).trace ≠ 0 :=
  fun h => hJ (Matrix.trace_mul_conjTranspose_self_eq_zero_iff.mp h)

/-- Non-negativity: `tr(E J J†) ≥ 0` for every effect `E ⪰ 0`. -/
theorem born_weight_nonneg (E J : M2) (hE : E.PosSemidef) : 0 ≤ (E * (J * Jᴴ)).trace := by
  have h : (E * (J * Jᴴ)).trace = (Jᴴ * E * J).trace := by
    rw [← Matrix.mul_assoc, Matrix.trace_mul_comm, ← Matrix.mul_assoc]
  rw [h]
  exact (hE.conjTranspose_mul_mul_same J).trace_nonneg

theorem nonneg_div_nonneg {a t : ℂ} (ha : 0 ≤ a) (ht : 0 ≤ t) : 0 ≤ a / t := by
  obtain ⟨har, hai⟩ := Complex.nonneg_iff.mp ha
  obtain ⟨htr, hti⟩ := Complex.nonneg_iff.mp ht
  have ha' : a = (a.re : ℂ) := Complex.ext rfl (by simpa using hai.symm)
  have ht' : t = (t.re : ℂ) := Complex.ext rfl (by simpa using hti.symm)
  rw [ha', ht', ← Complex.ofReal_div]
  exact Complex.zero_le_real.mpr (div_nonneg har htr)

/-- The Born rule from kinematics alone. For any nonzero carrier `J` and any measurement
`{E_m}` (`E_m ⪰ 0`, `Σ E_m = 1`), the weights `p_m = tr(E_m J J†) / tr(J J†)` are a
probability distribution, and they do not change under `J ↦ J U` with `U` unitary. -/
theorem born_kinematic {ι : Type*} [Fintype ι] (E : ι → M2) (J : M2) (hJ : J ≠ 0)
    (hE : ∀ m, (E m).PosSemidef) (hsum : ∑ n, E n = 1) :
    (∀ m, 0 ≤ (E m * (J * Jᴴ)).trace / (J * Jᴴ).trace) ∧
    (∑ m, (E m * (J * Jᴴ)).trace / (J * Jᴴ).trace = 1) ∧
    (∀ U : M2, U * Uᴴ = 1 → ∀ m,
      (E m * ((J * U) * (J * U)ᴴ)).trace / ((J * U) * (J * U)ᴴ).trace
        = (E m * (J * Jᴴ)).trace / (J * Jᴴ).trace) := by
  refine ⟨fun m => ?_, born_weights_sum E _ hsum (closure_trace_ne_zero J hJ), ?_⟩
  · exact nonneg_div_nonneg (born_weight_nonneg (E m) J (hE m))
      (born_posSemidef J).trace_nonneg
  · intro U hU m
    rw [born_right_unitary J U hU]

end BornKinematic

end CalChecks
