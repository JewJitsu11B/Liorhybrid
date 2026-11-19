/-
Sixth batch: `Cl(3,0) ≅ M₂(ℂ)` as real algebras.

Mathlib has no dimension formula for Clifford algebras, so the bound `dim Cl(3,0) ≤ 8`
is proved here directly: the eight ordered words in the generators span the algebra.
Together with surjectivity of the Pauli representation this gives the isomorphism.
-/
import Mathlib
import CalChecks.Structure

open Matrix Complex

namespace CalChecks

noncomputable section

abbrev Cl3 := CliffordAlgebra Q3

/-- The three generators of `Cl(3,0)`. -/
def γ (i : Fin 3) : Cl3 := CliffordAlgebra.ι Q3 (Pi.single i 1)

theorem Q3_single (i : Fin 3) : Q3 (Pi.single i 1) = 1 := by
  fin_cases i <;>
    simp [Q3, QuadraticMap.weightedSumSquares_apply, Fin.sum_univ_three, Pi.single_apply]

theorem γ_sq (i : Fin 3) : γ i * γ i = 1 := by
  rw [γ, CliffordAlgebra.ι_sq_scalar, Q3_single, map_one]

theorem Q3_polar (i j : Fin 3) (h : i ≠ j) :
    QuadraticMap.polar Q3 (Pi.single j 1) (Pi.single i 1) = 0 := by
  fin_cases i <;> fin_cases j <;>
    first
    | exact absurd rfl h
    | (simp [QuadraticMap.polar, Q3, QuadraticMap.weightedSumSquares_apply, Fin.sum_univ_three,
        Pi.single_apply])

theorem γ_anti (i j : Fin 3) (h : i ≠ j) : γ j * γ i = -(γ i * γ j) := by
  have key := CliffordAlgebra.ι_mul_ι_add_swap (Q := Q3) (Pi.single j 1) (Pi.single i 1)
  rw [Q3_polar i j h, map_zero] at key
  exact eq_neg_of_add_eq_zero_left key

theorem s10 : γ 1 * γ 0 = -(γ 0 * γ 1) := γ_anti 0 1 (by decide)
theorem s20 : γ 2 * γ 0 = -(γ 0 * γ 2) := γ_anti 0 2 (by decide)
theorem s21 : γ 2 * γ 1 = -(γ 1 * γ 2) := γ_anti 1 2 (by decide)

theorem γ_sq' (i : Fin 3) (y : Cl3) : γ i * (γ i * y) = y := by
  rw [← mul_assoc, γ_sq, one_mul]
theorem s10' (y : Cl3) : γ 1 * (γ 0 * y) = -(γ 0 * (γ 1 * y)) := by
  rw [← mul_assoc, s10, neg_mul, mul_assoc]
theorem s20' (y : Cl3) : γ 2 * (γ 0 * y) = -(γ 0 * (γ 2 * y)) := by
  rw [← mul_assoc, s20, neg_mul, mul_assoc]
theorem s21' (y : Cl3) : γ 2 * (γ 1 * y) = -(γ 1 * (γ 2 * y)) := by
  rw [← mul_assoc, s21, neg_mul, mul_assoc]

/-- The eight ordered words. -/
def w : Fin 8 → Cl3 :=
  ![1, γ 0, γ 1, γ 2, γ 0 * γ 1, γ 0 * γ 2, γ 1 * γ 2, γ 0 * (γ 1 * γ 2)]

abbrev S : Submodule ℝ Cl3 := Submodule.span ℝ (Set.range w)

theorem w_mem (k : Fin 8) : w k ∈ S := Submodule.subset_span ⟨k, rfl⟩

/-- Left multiplication by a generator permutes the words up to sign. -/
theorem γ_mul_w_mem (i : Fin 3) (k : Fin 8) : γ i * w k ∈ S := by
  have m0 : (1 : Cl3) ∈ S := by simpa [w] using w_mem 0
  have m1 : γ 0 ∈ S := by simpa [w] using w_mem 1
  have m2 : γ 1 ∈ S := by simpa [w] using w_mem 2
  have m3 : γ 2 ∈ S := by simpa [w] using w_mem 3
  have m4 : γ 0 * γ 1 ∈ S := by simpa [w] using w_mem 4
  have m5 : γ 0 * γ 2 ∈ S := by simpa [w] using w_mem 5
  have m6 : γ 1 * γ 2 ∈ S := by simpa [w] using w_mem 6
  have m7 : γ 0 * (γ 1 * γ 2) ∈ S := by simpa [w] using w_mem 7
  fin_cases i <;> fin_cases k <;>
    simp [w, γ_sq, γ_sq', s10, s20, s21, s10', s20', s21', m0, m1, m2, m3, m4, m5, m6, m7]

theorem γ_mul_mem (i : Fin 3) {a : Cl3} (ha : a ∈ S) : γ i * a ∈ S := by
  induction ha using Submodule.span_induction with
  | mem x hx =>
    obtain ⟨k, rfl⟩ := hx
    exact γ_mul_w_mem i k
  | zero => simp
  | add x y _ _ hx hy => rw [mul_add]; exact S.add_mem hx hy
  | smul r x _ hx => rw [mul_smul_comm]; exact S.smul_mem r hx

theorem ι_eq_sum (m : Fin 3 → ℝ) :
    CliffordAlgebra.ι Q3 m = m 0 • γ 0 + m 1 • γ 1 + m 2 • γ 2 := by
  have hm : m = m 0 • Pi.single 0 1 + m 1 • Pi.single 1 1 + m 2 • Pi.single 2 1 := by
    ext j; fin_cases j <;> simp
  conv_lhs => rw [hm]
  simp [γ, map_add, map_smul]

theorem mem_S (a : Cl3) : a ∈ S := by
  induction a using CliffordAlgebra.left_induction with
  | algebraMap r =>
    rw [Algebra.algebraMap_eq_smul_one]
    exact S.smul_mem r (by simpa [w] using w_mem 0)
  | add x y hx hy => exact S.add_mem hx hy
  | ι_mul x m hx =>
    rw [ι_eq_sum, add_mul, add_mul, smul_mul_assoc, smul_mul_assoc, smul_mul_assoc]
    exact S.add_mem (S.add_mem (S.smul_mem _ (γ_mul_mem 0 hx)) (S.smul_mem _ (γ_mul_mem 1 hx)))
      (S.smul_mem _ (γ_mul_mem 2 hx))

theorem S_eq_top : S = ⊤ := eq_top_iff.mpr fun a _ => mem_S a

instance : Module.Finite ℝ Cl3 :=
  Module.finite_def.mpr (by rw [← S_eq_top]; exact Submodule.fg_span (Set.finite_range w))

theorem finrank_Cl3_le : Module.finrank ℝ Cl3 ≤ 8 := by
  have h := finrank_range_le_card (R := ℝ) w
  rw [Set.finrank] at h
  have h' : Module.finrank ℝ (⊤ : Submodule ℝ Cl3) ≤ Fintype.card (Fin 8) := by
    rw [← S_eq_top]; exact h
  rw [finrank_top] at h'
  simpa using h'

theorem finrank_M2 : Module.finrank ℝ M2 = 8 := by
  rw [← Module.finrank_mul_finrank ℝ ℂ M2, Complex.finrank_real_complex, Module.finrank_matrix]
  simp

/-! ### The Pauli representation is onto -/

theorem F_γ0 : cliffordToM2 (γ 0) = σx := by
  rw [γ, cliffordToM2_ι]
  ext i j; fin_cases i <;> fin_cases j <;> simp [dotσ, σx, σy, σz]

theorem F_γ1 : cliffordToM2 (γ 1) = σy := by
  rw [γ, cliffordToM2_ι]
  ext i j; fin_cases i <;> fin_cases j <;> simp [dotσ, σx, σy, σz]

theorem F_γ2 : cliffordToM2 (γ 2) = σz := by
  rw [γ, cliffordToM2_ι]
  ext i j; fin_cases i <;> fin_cases j <;> simp [dotσ, σx, σy, σz]

/-- The pseudoscalar `γ₀γ₁γ₂` maps to `i·1`: this is the `h` of the one-sheet. -/
theorem F_vol : cliffordToM2 (γ 0 * (γ 1 * γ 2)) = I • (1 : M2) := by
  rw [map_mul, map_mul, F_γ0, F_γ1, F_γ2]
  ext i j; fin_cases i <;> fin_cases j <;> simp [σx, σy, σz]

/-- Paravector `t + xγ₀ + yγ₁ + zγ₂`. -/
def pre (t x y z : ℝ) : Cl3 := algebraMap ℝ Cl3 t + x • γ 0 + y • γ 1 + z • γ 2

theorem F_pre (t x y z : ℝ) : cliffordToM2 (pre t x y z) = herm t x y z := by
  simp only [pre, map_add, map_smul, AlgHom.commutes, F_γ0, F_γ1, F_γ2]
  rw [herm_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.algebraMap_matrix_apply, σx, σy, σz] <;> cring

theorem cliffordToM2_surjective : Function.Surjective cliffordToM2 := by
  intro M
  obtain ⟨t, x, y, z, h1⟩ := exists_herm (H M) (H_isHermitian M)
  obtain ⟨t', x', y', z', h2⟩ := exists_herm (H ((-I) • M)) (H_isHermitian _)
  refine ⟨pre t x y z + γ 0 * (γ 1 * γ 2) * pre t' x' y' z', ?_⟩
  rw [map_add, map_mul, F_pre, F_vol, F_pre, h1, h2, Matrix.smul_mul, Matrix.one_mul,
    ← A_eq_I_smul_H, H_add_A]

theorem cliffordToM2_injective : Function.Injective cliffordToM2 := by
  have hsurj : LinearMap.range cliffordToM2.toLinearMap = ⊤ :=
    LinearMap.range_eq_top.mpr cliffordToM2_surjective
  have hrn := LinearMap.finrank_range_add_finrank_ker cliffordToM2.toLinearMap
  rw [hsurj, finrank_top, finrank_M2] at hrn
  have hle : Module.finrank ℝ (CliffordAlgebra Q3) ≤ 8 := finrank_Cl3_le
  have hk : Module.finrank ℝ (LinearMap.ker cliffordToM2.toLinearMap) = 0 := by omega
  have hbot : LinearMap.ker cliffordToM2.toLinearMap = ⊥ := Submodule.finrank_eq_zero.mp hk
  exact LinearMap.ker_eq_bot.mp hbot

/-- `Cl(3,0) ≅ M₂(ℂ)` as real algebras. -/
def cliffordEquiv : Cl3 ≃ₐ[ℝ] M2 :=
  AlgEquiv.ofBijective cliffordToM2 ⟨cliffordToM2_injective, cliffordToM2_surjective⟩

theorem cliffordEquiv_ι (v : Fin 3 → ℝ) : cliffordEquiv (CliffordAlgebra.ι Q3 v) = dotσ v :=
  cliffordToM2_ι v

/-- `dim_ℝ Cl(3,0) = 8`. -/
theorem finrank_Cl3 : Module.finrank ℝ Cl3 = 8 := by
  rw [← finrank_M2]
  exact LinearEquiv.finrank_eq cliffordEquiv.toLinearEquiv

end

end CalChecks
