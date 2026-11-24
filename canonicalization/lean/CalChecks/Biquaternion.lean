/-
Biquaternions, complex 2x2 matrices and Cl(3,0) are one algebra.

  𝔹 = ℍ[ℂ]  (quaternions with complex coefficients)
      ≃ₐ[ℂ]  M₂(ℂ)        by  1, i, j, k ↦ 1, e₁, e₂, e₃  (e_j = -i σ_j, the channel units)
      ≃ₐ[ℝ]  Cl(3,0)      through `cliffordEquiv` of `Clifford.lean`

and, in between, the dictionary of involutions, norm, trace, units and grades.
The first map is the carrier itself: `biquatToM2 q = carrier ![q.re, q.imI, q.imJ, q.imK]`.
-/
import CalChecks.CliffordNative

open Matrix Complex Quaternion
open scoped ComplexConjugate

namespace CalChecks

/-- Biquaternions: quaternions with complex coefficients. -/
abbrev Biquat := ℍ[ℂ]

/-! ## 1. 𝔹 → M₂(ℂ) -/

/-- The channel units `e₁, e₂, e₃` are a quaternion basis of `M₂(ℂ)` over `ℂ`. -/
def unitBasis : QuaternionAlgebra.Basis M2 (-1 : ℂ) 0 (-1) where
  i := e1
  j := e2
  k := e3
  i_mul_i := by rw [e1_sq]; simp
  j_mul_j := by rw [e2_sq]; simp
  i_mul_j := e1_e2
  j_mul_i := by rw [e2_e1]; simp

/-- The algebra map `𝔹 → M₂(ℂ)` fixed by `i, j, k ↦ e₁, e₂, e₃`. -/
def biquatToM2 : Biquat →ₐ[ℂ] M2 := unitBasis.liftHom

/-- It is the carrier: the four complex coefficients are the channel amplitudes. -/
theorem biquatToM2_apply (q : Biquat) :
    biquatToM2 q = carrier ![q.re, q.imI, q.imJ, q.imK] := by
  show unitBasis.lift q = _
  simp [QuaternionAlgebra.Basis.lift, unitBasis, carrier, Algebra.algebraMap_eq_smul_one]

theorem biquatToM2_injective : Function.Injective biquatToM2 := by
  intro p q h
  rw [biquatToM2_apply, biquatToM2_apply] at h
  have hc := carrier_injective h
  have h0 := congrFun hc 0
  have h1 := congrFun hc 1
  have h2 := congrFun hc 2
  have h3 := congrFun hc 3
  simp at h0 h1 h2 h3
  exact Quaternion.ext p q h0 h1 h2 h3

theorem biquatToM2_surjective : Function.Surjective biquatToM2 := by
  intro M
  refine ⟨⟨coordsOf M 0, coordsOf M 1, coordsOf M 2, coordsOf M 3⟩, ?_⟩
  rw [biquatToM2_apply]
  have hv : (![coordsOf M 0, coordsOf M 1, coordsOf M 2, coordsOf M 3] : Fin 4 → ℂ) = coordsOf M := by
    ext a; fin_cases a <;> rfl
  show carrier ![coordsOf M 0, coordsOf M 1, coordsOf M 2, coordsOf M 3] = M
  rw [hv, carrier_coordsOf]

/-- **𝔹 ≅ M₂(ℂ)** as complex algebras. -/
noncomputable def biquatEquivM2 : Biquat ≃ₐ[ℂ] M2 :=
  AlgEquiv.ofBijective biquatToM2 ⟨biquatToM2_injective, biquatToM2_surjective⟩

theorem biquatEquivM2_apply (q : Biquat) : biquatEquivM2 q = biquatToM2 q := rfl

/-! ## 2. Units -/

theorem biquatToM2_i : biquatToM2 (⟨0, 1, 0, 0⟩ : Biquat) = e1 := by
  rw [biquatToM2_apply]; simp [carrier]
theorem biquatToM2_j : biquatToM2 (⟨0, 0, 1, 0⟩ : Biquat) = e2 := by
  rw [biquatToM2_apply]; simp [carrier]
theorem biquatToM2_k : biquatToM2 (⟨0, 0, 0, 1⟩ : Biquat) = e3 := by
  rw [biquatToM2_apply]; simp [carrier]
/-- A complex scalar goes to that multiple of the identity; in particular `i ↦ i·1`. -/
theorem biquatToM2_coe (z : ℂ) : biquatToM2 (z : Biquat) = z • (1 : M2) := by
  rw [biquatToM2_apply]; simp [carrier]

/-! ## 3. Involutions, norm, trace on the matrix side -/

/-- Complex conjugation of the four coefficients. -/
def coeffConj (q : Biquat) : Biquat := ⟨conj q.re, conj q.imI, conj q.imJ, conj q.imK⟩

/-- Quaternion conjugate ↦ adjugate. -/
theorem biquatToM2_star (q : Biquat) : biquatToM2 (star q) = adjugate (biquatToM2 q) := by
  rw [biquatToM2_apply, biquatToM2_apply, carrier_eq, carrier_eq, adjugate_fin_two_of]
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> ring

/-- Hermitian conjugate of a biquaternion (both conjugations) ↦ conjugate transpose. -/
theorem biquatToM2_star_coeffConj (q : Biquat) :
    biquatToM2 (star (coeffConj q)) = (biquatToM2 q)ᴴ := by
  rw [biquatToM2_apply, biquatToM2_apply, carrier_eq, carrier_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [coeffConj, conjTranspose_apply] <;> ring

/-- Complex conjugation of the coefficients ↦ `M ↦ (adjugate M)ᴴ`. -/
theorem biquatToM2_coeffConj (q : Biquat) :
    biquatToM2 (coeffConj q) = (adjugate (biquatToM2 q))ᴴ := by
  rw [biquatToM2_apply, biquatToM2_apply, carrier_eq, carrier_eq, adjugate_fin_two_of]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [coeffConj, conjTranspose_apply] <;> ring

/-- The quaternion norm (complex valued) is the determinant. -/
theorem det_biquatToM2 (q : Biquat) : (biquatToM2 q).det = Quaternion.normSq q := by
  rw [biquatToM2_apply, det_carrier, Quaternion.normSq_def']
  simp

/-- `q q* ↦ (det) 1`. -/
theorem biquatToM2_mul_star (q : Biquat) :
    biquatToM2 (q * star q) = (biquatToM2 q).det • (1 : M2) := by
  rw [map_mul, biquatToM2_star, mul_adjugate]

/-- The scalar coefficient is half the trace. -/
theorem trace_biquatToM2 (q : Biquat) : (biquatToM2 q).trace = 2 * q.re := by
  rw [biquatToM2_apply, carrier_eq, trace_fin_two_of]
  simp; ring

/-! ## 4. 𝔹 ≅ Cl(3,0) -/

/-- **𝔹 ≅ Cl(3,0)** as real algebras: `𝔹 ≅ M₂(ℂ) ≅ Cl(3,0)`. -/
noncomputable def biquatEquivCl3 : Biquat ≃ₐ[ℝ] Cl3 :=
  (biquatEquivM2.restrictScalars ℝ).trans cliffordEquiv.symm

/-- The triangle commutes: through `Cl(3,0)` or directly, the matrix is the same. -/
theorem cliffordEquiv_biquatEquivCl3 (q : Biquat) :
    cliffordEquiv (biquatEquivCl3 q) = biquatToM2 q := by
  simp [biquatEquivCl3, biquatEquivM2_apply]

/-- Quaternion conjugate ↦ Clifford conjugate. -/
theorem biquatEquivCl3_star (q : Biquat) :
    biquatEquivCl3 (star q) = cconj (biquatEquivCl3 q) := by
  apply cliffordEquiv.injective
  rw [dict_cconj, cliffordEquiv_biquatEquivCl3, cliffordEquiv_biquatEquivCl3, biquatToM2_star]

/-- Hermitian conjugate ↦ reversion. -/
theorem biquatEquivCl3_star_coeffConj (q : Biquat) :
    biquatEquivCl3 (star (coeffConj q)) = CliffordAlgebra.reverse (biquatEquivCl3 q) := by
  apply cliffordEquiv.injective
  rw [dict_reverse, cliffordEquiv_biquatEquivCl3, cliffordEquiv_biquatEquivCl3,
    biquatToM2_star_coeffConj]

/-- Complex conjugation of the coefficients ↦ grade involution. -/
theorem biquatEquivCl3_coeffConj (q : Biquat) :
    biquatEquivCl3 (coeffConj q) = CliffordAlgebra.involute (biquatEquivCl3 q) := by
  apply cliffordEquiv.injective
  rw [dict_involute, cliffordEquiv_biquatEquivCl3, cliffordEquiv_biquatEquivCl3,
    biquatToM2_coeffConj]

/-- The quaternion norm is the Clifford norm `N q = q * conj q`. -/
theorem biquatEquivCl3_N (q : Biquat) :
    biquatEquivCl3 (q * star q) = N (biquatEquivCl3 q) := by
  rw [N, map_mul, biquatEquivCl3_star]

/-- `i, j, k` are the unit bivectors. -/
theorem biquatEquivCl3_i : biquatEquivCl3 (⟨0, 1, 0, 0⟩ : Biquat) = γ 2 * γ 1 := by
  apply cliffordEquiv.injective
  rw [cliffordEquiv_biquatEquivCl3, biquatToM2_i, cliffordEquiv_apply, map_mul, F_γ2, F_γ1]
  ext i j; fin_cases i <;> fin_cases j <;> simp [e1, σx, σy, σz]
theorem biquatEquivCl3_j : biquatEquivCl3 (⟨0, 0, 1, 0⟩ : Biquat) = γ 0 * γ 2 := by
  apply cliffordEquiv.injective
  rw [cliffordEquiv_biquatEquivCl3, biquatToM2_j, cliffordEquiv_apply, map_mul, F_γ0, F_γ2]
  ext i j; fin_cases i <;> fin_cases j <;> simp [e2, σx, σy, σz]
theorem biquatEquivCl3_k : biquatEquivCl3 (⟨0, 0, 0, 1⟩ : Biquat) = γ 1 * γ 0 := by
  apply cliffordEquiv.injective
  rw [cliffordEquiv_biquatEquivCl3, biquatToM2_k, cliffordEquiv_apply, map_mul, F_γ1, F_γ0]
  ext i j; fin_cases i <;> fin_cases j <;> simp [e3, σx, σy, σz]

/-- The complex unit of the coefficients is the pseudoscalar. -/
theorem biquatEquivCl3_I : biquatEquivCl3 ((I : ℂ) : Biquat) = Ips := by
  apply cliffordEquiv.injective
  rw [cliffordEquiv_biquatEquivCl3, biquatToM2_coe, dict_Ips]

/-- **Grades.** With coefficients `Z_a = a_a + i b_a`:
`Re Z₀` is grade 0, `Im Z_j` grade 1, `Re Z_j` grade 2 (with a sign), `Im Z₀` grade 3. -/
theorem biquatEquivCl3_grades (a0 a1 a2 a3 b0 b1 b2 b3 : ℝ) :
    biquatEquivCl3 (⟨a0 + b0 * I, a1 + b1 * I, a2 + b2 * I, a3 + b3 * I⟩ : Biquat)
      = algebraMap ℝ Cl3 a0 + pre 0 b1 b2 b3 - biv a1 a2 a3 + b0 • Ips := by
  apply cliffordEquiv.injective
  rw [cliffordEquiv_biquatEquivCl3, biquatToM2_apply, carrier_eq]
  simp only [cliffordEquiv_apply, map_add, map_sub, map_smul, F_pre, toM2_biv, toM2_Ips,
    AlgHom.commutes]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [herm, σx, σy, σz, Algebra.algebraMap_eq_smul_one] <;> ring_nf <;> simp [Complex.I_sq] <;> ring

/-! ## 5. The tensor form: ℂ ⊗_ℝ ℍ ≅ 𝔹 -/

section Tensor
open TensorProduct

/-- The real quaternion units inside the biquaternions. -/
def realBasis : QuaternionAlgebra.Basis Biquat (-1 : ℝ) 0 (-1) where
  i := ⟨0, 1, 0, 0⟩
  j := ⟨0, 0, 1, 0⟩
  k := ⟨0, 0, 0, 1⟩
  i_mul_i := by ext <;> simp
  j_mul_j := by ext <;> simp
  i_mul_j := by ext <;> simp
  j_mul_i := by ext <;> simp

/-- The real quaternions sit in the biquaternions, coefficient by coefficient. -/
noncomputable def quatToBiquat : ℍ[ℝ] →ₐ[ℝ] Biquat := realBasis.liftHom

theorem quatToBiquat_apply (h : ℍ[ℝ]) :
    quatToBiquat h = (⟨(h.re : ℂ), (h.imI : ℂ), (h.imJ : ℂ), (h.imK : ℂ)⟩ : Biquat) := by
  show realBasis.lift h = _
  ext <;> simp [QuaternionAlgebra.Basis.lift, realBasis, Algebra.algebraMap_eq_smul_one]

/-- `z ⊗ h ↦ z h`. -/
noncomputable def tensorToBiquat : ℂ ⊗[ℝ] ℍ[ℝ] →ₐ[ℂ] Biquat :=
  Algebra.TensorProduct.lift (Algebra.ofId ℂ Biquat) quatToBiquat
    (fun z h => Algebra.commutes z (quatToBiquat h))

theorem tensorToBiquat_tmul (z : ℂ) (h : ℍ[ℝ]) :
    tensorToBiquat (z ⊗ₜ[ℝ] h) = (z : Biquat) * quatToBiquat h := by
  simp [tensorToBiquat]
  rfl

theorem tensorToBiquat_surjective : Function.Surjective tensorToBiquat := by
  intro q
  refine ⟨q.re ⊗ₜ[ℝ] (1 : ℍ[ℝ]) + q.imI ⊗ₜ[ℝ] (⟨0, 1, 0, 0⟩ : ℍ[ℝ])
    + q.imJ ⊗ₜ[ℝ] (⟨0, 0, 1, 0⟩ : ℍ[ℝ]) + q.imK ⊗ₜ[ℝ] (⟨0, 0, 0, 1⟩ : ℍ[ℝ]), ?_⟩
  simp only [map_add, tensorToBiquat_tmul, quatToBiquat_apply]
  ext <;> simp

theorem tensorToBiquat_injective : Function.Injective tensorToBiquat := by
  have hfr : Module.finrank ℂ (ℂ ⊗[ℝ] ℍ[ℝ]) = Module.finrank ℂ Biquat := by
    rw [Module.finrank_baseChange, Quaternion.finrank_eq_four, Quaternion.finrank_eq_four]
  exact (LinearMap.injective_iff_surjective_of_finrank_eq_finrank hfr
    (f := tensorToBiquat.toLinearMap)).mpr tensorToBiquat_surjective

/-- **ℂ ⊗_ℝ ℍ ≅ 𝔹** as complex algebras. -/
noncomputable def tensorEquivBiquat : ℂ ⊗[ℝ] ℍ[ℝ] ≃ₐ[ℂ] Biquat :=
  AlgEquiv.ofBijective tensorToBiquat ⟨tensorToBiquat_injective, tensorToBiquat_surjective⟩

/-- **The whole chain:** `ℂ ⊗_ℝ ℍ ≅ 𝔹 ≅ M₂(ℂ) ≅ Cl(3,0)` as real algebras. -/
noncomputable def tensorEquivCl3 : ℂ ⊗[ℝ] ℍ[ℝ] ≃ₐ[ℝ] Cl3 :=
  (tensorEquivBiquat.restrictScalars ℝ).trans biquatEquivCl3

/-- All four have real dimension 8. -/
theorem finrank_Biquat_real : Module.finrank ℝ Biquat = 8 := by
  rw [← finrank_Cl3]
  exact LinearEquiv.finrank_eq biquatEquivCl3.toLinearEquiv

end Tensor

end CalChecks

section Audit
open CalChecks
#print axioms biquatEquivM2
#print axioms biquatEquivCl3
#print axioms biquatEquivCl3_star
#print axioms biquatEquivCl3_star_coeffConj
#print axioms biquatEquivCl3_coeffConj
#print axioms biquatEquivCl3_grades
#print axioms det_biquatToM2
#print axioms tensorEquivBiquat
#print axioms tensorEquivCl3
end Audit
