/-
Eighth batch: `(Sp(1) × Sp(1)) / {±1} ≅ SO(4)`.

Every rotation of the four real channel coordinates is `x ↦ p x q` for unit quaternions
`p`, `q`, and the only pairs acting trivially are `±(1, 1)`.

Surjectivity uses the Cartan–Dieudonné theorem from Mathlib (every linear isometry is a
product of reflections) together with the fact that the reflection of `ℍ` in the hyperplane
orthogonal to a unit quaternion `u` is `x ↦ -u x̄ u`.
-/
import Mathlib
import CalChecks.Final

open Matrix Quaternion
open scoped Quaternion RealInnerProductSpace

namespace CalChecks

noncomputable section

/-! ## Isometries of the quaternions -/

/-- `u x̄ u = 2⟨u,x⟩ u - |u|² x`. -/
theorem quat_sandwich (u x : ℍ) :
    u * star x * u = (2 * inner ℝ u x) • u - normSq u • x := by
  ext <;> simp [Quaternion.inner_def, Quaternion.normSq_def'] <;> ring

/-- A map of the form `x ↦ p x q` or `x ↦ p x̄ q` with `|p| = |q| = 1`. -/
def IsLR (φ : ℍ → ℍ) : Prop :=
  ∃ p q : ℍ, ‖p‖ = 1 ∧ ‖q‖ = 1 ∧ ((∀ x, φ x = p * x * q) ∨ (∀ x, φ x = p * star x * q))

theorem IsLR.comp {f g : ℍ → ℍ} (hf : IsLR f) (hg : IsLR g) : IsLR (f ∘ g) := by
  obtain ⟨p1, q1, hp1, hq1, h1⟩ := hf
  obtain ⟨p2, q2, hp2, hq2, h2⟩ := hg
  rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
  · exact ⟨p1 * p2, q2 * q1, by simp [hp1, hp2], by simp [hq1, hq2],
      Or.inl fun x => by simp [h1, h2, mul_assoc]⟩
  · exact ⟨p1 * p2, q2 * q1, by simp [hp1, hp2], by simp [hq1, hq2],
      Or.inr fun x => by simp [h1, h2, mul_assoc]⟩
  · exact ⟨p1 * star q2, star p2 * q1, by simp [hp1, hq2], by simp [hq1, hp2],
      Or.inr fun x => by simp [h1, h2, mul_assoc]⟩
  · exact ⟨p1 * star q2, star p2 * q1, by simp [hp1, hq2], by simp [hq1, hp2],
      Or.inl fun x => by simp [h1, h2, mul_assoc]⟩

/-- A reflection of `ℍ` in a hyperplane is of the form `x ↦ p x̄ q` (or the identity). -/
theorem reflection_isLR (v : ℍ) : IsLR ((ℝ ∙ v)ᗮ.reflection) := by
  by_cases hv : v = 0
  · subst hv
    refine ⟨1, 1, by simp, by simp, Or.inl fun x => ?_⟩
    rw [Submodule.reflection_orthogonal_apply, Submodule.reflection_singleton_apply]
    simp
  · have hn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
    have hsq : normSq v = ‖v‖ ^ 2 := by rw [Quaternion.normSq_eq_norm_mul_self]; ring
    refine ⟨-(‖v‖⁻¹ • v), ‖v‖⁻¹ • v, ?_, ?_, Or.inr fun x => ?_⟩
    · rw [norm_neg, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]
    · rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]
    · rw [Submodule.reflection_orthogonal_apply, Submodule.reflection_singleton_apply]
      have key : -(‖v‖⁻¹ • v) * star x * (‖v‖⁻¹ • v) = -((‖v‖⁻¹) ^ 2 • (v * star x * v)) := by
        simp only [neg_mul, smul_mul_assoc, mul_smul_comm, smul_neg, smul_smul, pow_two]
      rw [key, quat_sandwich, hsq]
      simp only [RCLike.ofReal_real_eq_id, id_eq]
      match_scalars <;> field_simp

/-- Every linear isometry of `ℍ` is `x ↦ p x q` or `x ↦ p x̄ q`. -/
theorem isometry_isLR (φ : ℍ ≃ₗᵢ[ℝ] ℍ) : IsLR φ := by
  obtain ⟨l, -, rfl⟩ := φ.reflections_generate_dim
  induction l with
  | nil => exact ⟨1, 1, by simp, by simp, Or.inl fun x => by simp⟩
  | cons v l ih =>
    rw [List.map_cons, List.prod_cons]
    exact (reflection_isLR v).comp ih

/-! ## Bridge to the 4×4 matrices `Lmat`, `Rmat` -/

/-- Coordinates of a quaternion. -/
def qc (x : ℍ) : Fin 4 → ℝ := ![x.re, x.imI, x.imJ, x.imK]

/-- The quaternion with given coordinates. -/
def qv (v : Fin 4 → ℝ) : ℍ := ⟨v 0, v 1, v 2, v 3⟩

theorem qc_qv (v : Fin 4 → ℝ) : qc (qv v) = v := by
  ext i; fin_cases i <;> simp [qc, qv]

theorem qv_qc (x : ℍ) : qv (qc x) = x := by
  ext <;> simp [qc, qv]

theorem qc_add (x y : ℍ) : qc (x + y) = qc x + qc y := by
  ext i; fin_cases i <;> simp [qc]

theorem qc_smul (r : ℝ) (x : ℍ) : qc (r • x) = r • qc x := by
  ext i; fin_cases i <;> simp [qc]

theorem qv_add (v w : Fin 4 → ℝ) : qv (v + w) = qv v + qv w := by
  ext <;> simp [qv]

theorem qv_smul (r : ℝ) (v : Fin 4 → ℝ) : qv (r • v) = r • qv v := by
  ext <;> simp [qv]

/-- `Lmat` is left multiplication. -/
theorem Lmat_mulVec (p x : ℍ) : Lmat (qc p) *ᵥ qc x = qc (p * x) := by
  ext i
  fin_cases i <;> simp [Lmat, qc, Matrix.mulVec, dotProduct, Fin.sum_univ_four] <;> ring

/-- `Rmat` is right multiplication. -/
theorem Rmat_mulVec (q x : ℍ) : Rmat (qc q) *ᵥ qc x = qc (x * q) := by
  ext i
  fin_cases i <;> simp [Rmat, qc, Matrix.mulVec, dotProduct, Fin.sum_univ_four] <;> ring

/-- Matrix of quaternion conjugation. -/
def Cmat : Matrix (Fin 4) (Fin 4) ℝ := Matrix.diagonal ![1, -1, -1, -1]

theorem Cmat_mulVec (x : ℍ) : Cmat *ᵥ qc x = qc (star x) := by
  ext i
  fin_cases i <;> simp [Cmat, qc, Matrix.mulVec_diagonal]

theorem Cmat_det : Cmat.det = -1 := by
  simp [Cmat, Matrix.det_diagonal, Fin.prod_univ_four]

theorem Lmat_mul (a b : ℍ) : Lmat (qc (a * b)) = Lmat (qc a) * Lmat (qc b) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Lmat, qc, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

theorem Rmat_mul (a b : ℍ) : Rmat (qc (a * b)) = Rmat (qc b) * Rmat (qc a) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Rmat, qc, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

theorem normSq_eq_dot (x : ℍ) : normSq x = qc x ⬝ᵥ qc x := by
  simp [Quaternion.normSq_def', qc, dotProduct, Fin.sum_univ_four]; ring

theorem unit_qc {p : ℍ} (hp : ‖p‖ = 1) :
    qc p 0 ^ 2 + qc p 1 ^ 2 + qc p 2 ^ 2 + qc p 3 ^ 2 = 1 := by
  have h := Quaternion.normSq_eq_norm_mul_self p
  rw [hp, Quaternion.normSq_def'] at h
  simpa [qc] using h

theorem dot_orth (R : Matrix (Fin 4) (Fin 4) ℝ) (h : Rᵀ * R = 1) (w : Fin 4 → ℝ) :
    (R *ᵥ w) ⬝ᵥ (R *ᵥ w) = w ⬝ᵥ w := by
  rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec, h,
    Matrix.one_mulVec]

/-- The isometry of `ℍ` given by an orthogonal matrix acting on coordinates. -/
def isoOf (R : Matrix (Fin 4) (Fin 4) ℝ) (h : Rᵀ * R = 1) : ℍ ≃ₗᵢ[ℝ] ℍ :=
  LinearIsometry.toLinearIsometryEquiv
    { toFun := fun x => qv (R *ᵥ qc x)
      map_add' := fun x y => by rw [qc_add, Matrix.mulVec_add, qv_add]
      map_smul' := fun r x => by rw [qc_smul, Matrix.mulVec_smul, qv_smul]; rfl
      norm_map' := fun x => by
        have hsq : ‖qv (R *ᵥ qc x)‖ * ‖qv (R *ᵥ qc x)‖ = ‖x‖ * ‖x‖ := by
          rw [← Quaternion.normSq_eq_norm_mul_self, ← Quaternion.normSq_eq_norm_mul_self,
            normSq_eq_dot, qc_qv, dot_orth R h, normSq_eq_dot]
        exact (mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)).mp hsq }
    rfl

theorem isoOf_apply (R : Matrix (Fin 4) (Fin 4) ℝ) (h : Rᵀ * R = 1) (x : ℍ) :
    isoOf R h x = qv (R *ᵥ qc x) := rfl

/-! ## Surjectivity onto `SO(4)` -/

/-- Every rotation of the four channel coordinates is `x ↦ p x q` with `|p| = |q| = 1`. -/
theorem so4_surjective (R : Matrix (Fin 4) (Fin 4) ℝ) (hR : Rᵀ * R = 1) (hdet : R.det = 1) :
    ∃ p q : Fin 4 → ℝ, p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = 1 ∧
      q 0 ^ 2 + q 1 ^ 2 + q 2 ^ 2 + q 3 ^ 2 = 1 ∧ R = Lmat p * Rmat q := by
  obtain ⟨p, q, hp, hq, h⟩ := isometry_isLR (isoOf R hR)
  have hact : ∀ x, R *ᵥ qc x = qc (isoOf R hR x) := fun x => by rw [isoOf_apply, qc_qv]
  rcases h with h | h
  · refine ⟨qc p, qc q, unit_qc hp, unit_qc hq, ?_⟩
    have key : ∀ x, R *ᵥ qc x = (Lmat (qc p) * Rmat (qc q)) *ᵥ qc x := fun x => by
      rw [hact, h, ← Matrix.mulVec_mulVec, Rmat_mulVec, Lmat_mulVec, mul_assoc]
    apply Matrix.mulVec_injective
    funext w
    simpa [qc_qv] using key (qv w)
  · exfalso
    have key : ∀ x, R *ᵥ qc x = (Lmat (qc p) * Rmat (qc q) * Cmat) *ᵥ qc x := fun x => by
      rw [hact, h, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Cmat_mulVec, Rmat_mulVec,
        Lmat_mulVec, mul_assoc]
    have hRm : R = Lmat (qc p) * Rmat (qc q) * Cmat := by
      apply Matrix.mulVec_injective
      funext w
      simpa [qc_qv] using key (qv w)
    rw [hRm, Matrix.det_mul, Matrix.det_mul, Lmat_det, Rmat_det, unit_qc hp, unit_qc hq,
      Cmat_det] at hdet
    norm_num at hdet

/-! ## Kernel `{±(1,1)}` -/

/-- The only pairs of unit quaternions acting trivially are `(1,1)` and `(-1,-1)`. -/
theorem so4_kernel (p q : Fin 4 → ℝ) (hp : p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = 1)
    (h : Lmat p * Rmat q = 1) :
    (p = ![1, 0, 0, 0] ∧ q = ![1, 0, 0, 0]) ∨ (p = ![-1, 0, 0, 0] ∧ q = ![-1, 0, 0, 0]) := by
  have h2 : Rmat q = (Lmat p)ᵀ := by
    have h3 := congrArg (fun M => (Lmat p)ᵀ * M) h
    simp only [← Matrix.mul_assoc, Lmat_orthogonal, hp, one_smul, Matrix.one_mul,
      Matrix.mul_one] at h3
    exact h3
  have a00 := congrFun (congrFun h2 0) 0
  have a10 := congrFun (congrFun h2 1) 0
  have a20 := congrFun (congrFun h2 2) 0
  have a30 := congrFun (congrFun h2 3) 0
  have a12 := congrFun (congrFun h2 1) 2
  have a23 := congrFun (congrFun h2 2) 3
  have a31 := congrFun (congrFun h2 3) 1
  simp [Rmat, Lmat, Matrix.transpose_apply] at a00 a10 a20 a30 a12 a23 a31
  have p1 : p 1 = 0 := by linarith
  have p2 : p 2 = 0 := by linarith
  have p3 : p 3 = 0 := by linarith
  have q1 : q 1 = 0 := by linarith
  have q2 : q 2 = 0 := by linarith
  have q3 : q 3 = 0 := by linarith
  have hp0 : (p 0 - 1) * (p 0 + 1) = 0 := by
    rw [p1, p2, p3] at hp; linarith
  rcases mul_eq_zero.mp hp0 with h0 | h0
  · left
    constructor <;> (funext i; fin_cases i <;> simp <;> linarith)
  · right
    constructor <;> (funext i; fin_cases i <;> simp <;> linarith)

theorem so4_kernel_converse :
    Lmat ![1, 0, 0, 0] * Rmat ![1, 0, 0, 0] = 1 ∧ Lmat ![-1, 0, 0, 0] * Rmat ![-1, 0, 0, 0] = 1 := by
  constructor <;>
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [Lmat, Rmat, Matrix.mul_apply, Fin.sum_univ_four]

/-! ## The group isomorphism `(Sp(1) × Sp(1)) / {±1} ≃* SO(4)` -/

theorem eq_one_of_mul_self {t : ℝ} (h0 : 0 ≤ t) (h : t * t = 1) : t = 1 := by
  have h1 : (t - 1) * (t + 1) = 0 := by ring_nf; linarith
  rcases mul_eq_zero.mp h1 with h2 | h2 <;> linarith

theorem normSq_unitary (p : unitary ℍ) : normSq (p : ℍ) = 1 := by
  have h := (Unitary.mem_iff.mp p.2).2
  rw [Quaternion.self_mul_star] at h
  have h2 := congrArg (fun z : ℍ => z.re) h
  simpa using h2

theorem norm_unitary (p : unitary ℍ) : ‖(p : ℍ)‖ = 1 := by
  have h := normSq_unitary p
  rw [Quaternion.normSq_eq_norm_mul_self] at h
  exact eq_one_of_mul_self (norm_nonneg _) h

theorem mem_unitary_of_norm {a : ℍ} (h : ‖a‖ = 1) : a ∈ unitary ℍ := by
  have hn : normSq a = 1 := by rw [Quaternion.normSq_eq_norm_mul_self, h, mul_one]
  rw [Unitary.mem_iff, Quaternion.star_mul_self, Quaternion.self_mul_star, hn]
  simp

theorem norm_qv {v : Fin 4 → ℝ} (h : v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2 + v 3 ^ 2 = 1) : ‖qv v‖ = 1 := by
  have hn : normSq (qv v) = 1 := by
    rw [Quaternion.normSq_def']; simpa [qv] using h
  rw [Quaternion.normSq_eq_norm_mul_self] at hn
  exact eq_one_of_mul_self (norm_nonneg _) hn

theorem LR_mem_SO4 (p q : ℍ) (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) :
    Lmat (qc p) * Rmat (qc q) ∈ Matrix.specialOrthogonalGroup (Fin 4) ℝ := by
  rw [Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
  constructor
  · rw [Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc (Lmat (qc p))ᵀ,
      Lmat_orthogonal, unit_qc hp, one_smul, Matrix.one_mul, Rmat_orthogonal, unit_qc hq,
      one_smul]
  · rw [Matrix.det_mul, Lmat_det, Rmat_det, unit_qc hp, unit_qc hq]; norm_num

/-- The homomorphism `(p, q) ↦ (x ↦ p x q⁻¹)` from `Sp(1) × Sp(1)` to `SO(4)`. -/
def so4Hom : unitary ℍ × unitary ℍ →* Matrix.specialOrthogonalGroup (Fin 4) ℝ where
  toFun x := ⟨Lmat (qc (x.1 : ℍ)) * Rmat (qc (star (x.2 : ℍ))),
    LR_mem_SO4 _ _ (norm_unitary x.1) (by rw [Quaternion.norm_star]; exact norm_unitary x.2)⟩
  map_one' := by
    apply Subtype.ext
    ext i j
    fin_cases i <;> fin_cases j <;> simp [Lmat, Rmat, qc, Matrix.mul_apply, Fin.sum_univ_four]
  map_mul' x y := by
    apply Subtype.ext
    show Lmat (qc ((x.1 : ℍ) * (y.1 : ℍ))) * Rmat (qc (star ((x.2 : ℍ) * (y.2 : ℍ))))
      = (Lmat (qc (x.1 : ℍ)) * Rmat (qc (star (x.2 : ℍ))))
        * (Lmat (qc (y.1 : ℍ)) * Rmat (qc (star (y.2 : ℍ))))
    rw [star_mul, Lmat_mul, Rmat_mul, Matrix.mul_assoc, Matrix.mul_assoc,
      ← Matrix.mul_assoc (Lmat (qc (y.1 : ℍ))), Lmat_Rmat_comm, Matrix.mul_assoc]

theorem so4Hom_apply (x : unitary ℍ × unitary ℍ) :
    (so4Hom x : Matrix (Fin 4) (Fin 4) ℝ) = Lmat (qc (x.1 : ℍ)) * Rmat (qc (star (x.2 : ℍ))) :=
  rfl

/-- `Sp(1) × Sp(1) → SO(4)` is onto. -/
theorem so4Hom_surjective : Function.Surjective so4Hom := by
  intro A
  have hA := Matrix.mem_specialOrthogonalGroup_iff.mp A.2
  have hR : (A : Matrix (Fin 4) (Fin 4) ℝ)ᵀ * (A : Matrix (Fin 4) (Fin 4) ℝ) = 1 := by
    have h := hA.1
    rw [Matrix.mem_orthogonalGroup_iff'] at h
    exact h
  obtain ⟨p, q, hp, hq, h⟩ := so4_surjective A.1 hR hA.2
  refine ⟨(⟨qv p, mem_unitary_of_norm (norm_qv hp)⟩,
    ⟨star (qv q), mem_unitary_of_norm (by rw [Quaternion.norm_star]; exact norm_qv hq)⟩), ?_⟩
  apply Subtype.ext
  rw [so4Hom_apply]
  simp only [star_star, qc_qv]
  exact h.symm

/-- The kernel is exactly `{(1,1), (-1,-1)}`. -/
theorem so4Hom_ker (x : unitary ℍ × unitary ℍ) :
    so4Hom x = 1 ↔ ((x.1 : ℍ) = 1 ∧ (x.2 : ℍ) = 1) ∨ ((x.1 : ℍ) = -1 ∧ (x.2 : ℍ) = -1) := by
  constructor
  · intro h
    have h1 : Lmat (qc (x.1 : ℍ)) * Rmat (qc (star (x.2 : ℍ))) = 1 := by
      rw [← so4Hom_apply, h]; rfl
    have hq : ∀ a : ℍ, qc a = ![1, 0, 0, 0] → a = 1 := fun a ha => by
      rw [← qv_qc a, ha]; ext <;> simp [qv]
    have hq' : ∀ a : ℍ, qc a = ![-1, 0, 0, 0] → a = -1 := fun a ha => by
      rw [← qv_qc a, ha]; ext <;> simp [qv]
    rcases so4_kernel _ _ (unit_qc (norm_unitary x.1)) h1 with ⟨ha, hb⟩ | ⟨ha, hb⟩
    · left
      refine ⟨hq _ ha, ?_⟩
      have := hq _ hb
      simpa using congrArg star this
    · right
      refine ⟨hq' _ ha, ?_⟩
      have := hq' _ hb
      simpa using congrArg star this
  · rintro (⟨ha, hb⟩ | ⟨ha, hb⟩) <;>
    · apply Subtype.ext
      rw [so4Hom_apply, ha, hb]
      ext i j
      fin_cases i <;> fin_cases j <;>
        simp [Lmat, Rmat, qc, Matrix.mul_apply, Fin.sum_univ_four]

/-- `(Sp(1) × Sp(1)) / {±(1,1)} ≃* SO(4)`. -/
def so4Equiv :
    (unitary ℍ × unitary ℍ) ⧸ so4Hom.ker ≃* Matrix.specialOrthogonalGroup (Fin 4) ℝ :=
  QuotientGroup.quotientKerEquivOfSurjective so4Hom so4Hom_surjective

end

end CalChecks
