/-
Fourth batch: the remaining routine statements of the one-sheet, row by row.
-/
import Mathlib
import CalChecks.RepTheory

open Matrix Complex Filter Topology
open scoped ComplexConjugate Kronecker RealInnerProductSpace ComplexOrder

namespace CalChecks

/-- Closes complex polynomial identities, splitting into real and imaginary parts if needed. -/
macro "cring" : tactic =>
  `(tactic| first
    | done
    | ring
    | (apply Complex.ext <;> (try simp) <;> first | done | ring | linarith))

/-! ## The law at α = 2, and the half turn -/

/-- CAL with `α = 2`, delta kernel, trivial transport: `αJ - (1-α)J' = 2J + J'`. -/
theorem cal_alpha_two (J J' : M2) : (2 : ℂ) • J - (1 - (2 : ℂ)) • J' = (2 : ℂ) • J + J' := by
  module

/-- `θ = απ/2 = π` at `α = 2`, and `e^{iπ} = -1`. -/
theorem half_turn : Complex.exp (((2 * Real.pi / 2 : ℝ) : ℂ) * I) = -1 := by
  have h : ((2 * Real.pi / 2 : ℝ) : ℂ) = (Real.pi : ℂ) := by push_cast; ring
  rw [h, Complex.exp_pi_mul_I]

/-- `Z_a = A e^{-H/τ} + e^{iπ} B e^{iS/ℏ} = A e^{-H/τ} - B e^{iS/ℏ}`. -/
theorem channel_half_turn (A B E P : ℂ) :
    A * E + Complex.exp (Real.pi * I) * (B * P) = A * E - B * P := by
  rw [Complex.exp_pi_mul_I]; ring

/-- The sign `{±1}` is a character: it squares to one. -/
theorem sign_character : Complex.exp (Real.pi * I) * Complex.exp (Real.pi * I) = 1 := by
  rw [Complex.exp_pi_mul_I]; ring

/-! ## The carrier module `M₂(ℂ) ≅ ℂ ⊗ ℍ` -/

/-- Channel coordinates of a matrix. -/
noncomputable def coordsOf (M : M2) : Fin 4 → ℂ :=
  ![(M 0 0 + M 1 1) / 2, I * (M 0 1 + M 1 0) / 2, (M 1 0 - M 0 1) / 2, I * (M 0 0 - M 1 1) / 2]

/-- Every matrix is a carrier: `Z ↦ J` is onto. -/
theorem carrier_coordsOf (M : M2) : carrier (coordsOf M) = M := by
  rw [carrier_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [coordsOf] <;> ring_nf <;> simp [Complex.I_sq] <;> cring

/-- `Z ↦ J` is one-to-one. -/
theorem coordsOf_carrier (Z : Fin 4 → ℂ) : coordsOf (carrier Z) = Z := by
  rw [carrier_eq]
  ext a
  fin_cases a <;> simp [coordsOf] <;> ring_nf <;> simp [Complex.I_sq] <;> cring

theorem carrier_injective : Function.Injective carrier := fun Z W h => by
  rw [← coordsOf_carrier Z, ← coordsOf_carrier W, h]

/-- Left `SL(2,ℂ)` acts column by column: two copies of the spinor `(1/2, 0)`. -/
theorem left_action_columns (G J : M2) (j : Fin 2) :
    (fun i => (G * J) i j) = G *ᵥ (fun i => J i j) := by
  ext i; simp [Matrix.mul_apply, Matrix.mulVec, dotProduct]

/-- `(h i)² = +1`: `i · e₁ = σx`, and likewise for the other two. -/
theorem I_smul_e : I • e1 = σx ∧ I • e2 = σy ∧ I • e3 = σz := by
  refine ⟨?_, ?_, ?_⟩ <;>
    (ext i j; fin_cases i <;> fin_cases j <;> simp [e1, e2, e3, σx, σy, σz])

theorem σ_sq : σx * σx = 1 ∧ σy * σy = 1 ∧ σz * σz = 1 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    (ext i j; fin_cases i <;> fin_cases j <;> simp [σx, σy, σz])

/-- `det` is preserved by `SU(2)_L × SU(2)_R`. -/
theorem det_two_sided (U V J : M2) (hU : U.det = 1) (hV : V.det = 1) :
    (U * J * Vᴴ).det = J.det := by
  rw [det_mul, det_mul, det_conjTranspose, hU, hV]; simp

/-! ## The positive cone -/

/-- `J J†` is positive semidefinite. -/
theorem born_posSemidef (J : M2) : (J * Jᴴ).PosSemidef :=
  Matrix.posSemidef_self_mul_conjTranspose J

/-- Stabiliser of the base point `1` under congruence is `U(2)`. -/
theorem cone_stabilizer (G : M2) : G * 1 * Gᴴ = 1 ↔ G * Gᴴ = 1 := by
  rw [Matrix.mul_one]

/-- Rank is congruence invariant: `det(GϱG†) = 0 ↔ det ϱ = 0` for invertible `G`. -/
theorem rank_invariant (G ρ : M2) (hG : G.det ≠ 0) : (G * ρ * Gᴴ).det = 0 ↔ ρ.det = 0 := by
  rw [det_congruence]
  have : (normSq G.det : ℂ) ≠ 0 := by
    exact_mod_cast (normSq_pos.mpr hG).ne'
  simp [this]

/-- Boundary of the cone: `J J†` is singular exactly when `Σ Z_a² = 0`. -/
theorem born_singular_iff (Z : Fin 4 → ℂ) :
    (carrier Z * (carrier Z)ᴴ).det = 0 ↔ Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2 = 0 := by
  rw [det_born]
  simp [normSq_eq_zero]

/-! ## Hermitian and generator slices -/

/-- Congruence preserves Hermitian matrices. -/
theorem congruence_hermitian (G X : M2) (hX : Xᴴ = X) : (G * X * Gᴴ)ᴴ = G * X * Gᴴ := by
  simp [conjTranspose_mul, hX, Matrix.mul_assoc]

/-- Every Hermitian matrix is `t + xσx + yσy + zσz` with real coordinates. -/
theorem exists_herm (X : M2) (hX : Xᴴ = X) : ∃ t x y z : ℝ, herm t x y z = X := by
  have h00 := congrFun (congrFun hX 0) 0
  have h01 := congrFun (congrFun hX 0) 1
  have h11 := congrFun (congrFun hX 1) 1
  simp only [conjTranspose_apply] at h00 h01 h11
  have a := congrArg Complex.im h00
  have b := congrArg Complex.im h11
  have c := congrArg Complex.re h01
  have d := congrArg Complex.im h01
  simp at a b c d
  refine ⟨((X 0 0).re + (X 1 1).re) / 2, (X 1 0).re, (X 1 0).im,
    ((X 0 0).re - (X 1 1).re) / 2, ?_⟩
  rw [herm_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> apply Complex.ext <;> simp <;> linarith

/-- The coordinates on the Hermitian slice are unique. -/
theorem herm_injective (t x y z t' x' y' z' : ℝ) (h : herm t x y z = herm t' x' y' z') :
    t = t' ∧ x = x' ∧ y = y' ∧ z = z' := by
  rw [herm_eq, herm_eq] at h
  have h00 := congrArg Complex.re (congrFun (congrFun h 0) 0)
  have h11 := congrArg Complex.re (congrFun (congrFun h 1) 1)
  have h10r := congrArg Complex.re (congrFun (congrFun h 1) 0)
  have h10i := congrArg Complex.im (congrFun (congrFun h 1) 0)
  simp at h00 h11 h10r h10i
  refine ⟨by linarith, h10r, h10i, by linarith⟩

/-- `A(q) = i · H(-i q)`: the anti-Hermitian slice is `i` times the Hermitian one. -/
theorem A_eq_I_smul_H (q : M2) : A q = I • H ((-I) • q) := by
  unfold A H
  rw [conjTranspose_smul]
  simp only [star_neg, RCLike.star_def, Complex.conj_I, neg_neg, smul_add, smul_smul, smul_sub]
  ext i j
  simp
  ring_nf
  simp [Complex.I_sq]
  cring

/-- Every anti-Hermitian matrix is `y⁰(i1) + y_j e_j`: the generators span `u(2)`. -/
theorem exists_gen (Y : M2) (hY : Yᴴ = -Y) : ∃ y0 y1 y2 y3 : ℝ, gen y0 y1 y2 y3 = Y := by
  have hH : ((-I) • Y)ᴴ = (-I) • Y := by
    rw [conjTranspose_smul, hY]
    simp
  obtain ⟨t, x, y, z, h⟩ := exists_herm _ hH
  refine ⟨t, -x, -y, -z, ?_⟩
  rw [gen_eq_I_smul_herm]
  simp only [neg_neg]
  rw [h, smul_smul]
  simp

/-- `sl(2,ℂ) = su(2) ⊕ i·su(2)`: a traceless matrix is a rotation generator plus a boost. -/
theorem sl2_decomposition (M : M2) (hM : M.trace = 0) :
    ∃ y1 y2 y3 x y z : ℝ, M = gen 0 y1 y2 y3 + herm 0 x y z := by
  obtain ⟨t, x, y, z, hh⟩ := exists_herm (H M) (H_isHermitian M)
  obtain ⟨y0, y1, y2, y3, hg⟩ := exists_gen (A M) (A_antiHermitian M)
  have hsum : M = gen y0 y1 y2 y3 + herm t x y z := by
    rw [hg, hh, add_comm, H_add_A]
  have htr := congrArg Matrix.trace hsum
  rw [hM, Matrix.trace_add, gen_eq_I_smul_herm, herm_eq, herm_eq] at htr
  simp [trace_fin_two] at htr
  have hre := congrArg Complex.re htr
  have him := congrArg Complex.im htr
  simp at hre him
  have ht : t = 0 := by linarith
  have hy : y0 = 0 := by linarith
  exact ⟨y1, y2, y3, x, y, z, by rw [hsum, ht, hy]⟩

/-! ## POVM normalisation -/

/-- Under the contragredient action the resolution of identity becomes `G⁻† G⁻¹`. -/
theorem povm_sum {ι : Type*} [Fintype ι] (E : ι → M2) (Gi : M2) (hE : ∑ n, E n = 1) :
    ∑ n, Giᴴ * E n * Gi = Giᴴ * Gi := by
  rw [← Finset.sum_mul, ← Finset.mul_sum, hE, Matrix.mul_one]

/-- `Σ E_n = 1` survives exactly for unitary `G`. -/
theorem povm_sum_survives_iff (G Gi : M2) (h : Gi * G = 1) : Giᴴ * Gi = 1 ↔ G * Gᴴ = 1 := by
  have h' : G * Gi = 1 := mul_eq_one_comm.mp h
  constructor
  · intro hu
    have hG : G = Giᴴ := by
      calc G = (Giᴴ * Gi) * G := by rw [hu, Matrix.one_mul]
        _ = Giᴴ * (Gi * G) := Matrix.mul_assoc _ _ _
        _ = Giᴴ := by rw [h, Matrix.mul_one]
    rw [hG, conjTranspose_conjTranspose, hu]
  · intro hu
    have hGi : Gi = Gᴴ := by
      calc Gi = Gi * (G * Gᴴ) := by rw [hu, Matrix.mul_one]
        _ = (Gi * G) * Gᴴ := (Matrix.mul_assoc _ _ _).symm
        _ = Gᴴ := by rw [h, Matrix.one_mul]
    rw [hGi, conjTranspose_conjTranspose, hu]

/-- The Born weights sum to one: `Σ_m tr(E_m ϱ) / tr ϱ = 1`. -/
theorem born_weights_sum {ι : Type*} [Fintype ι] (E : ι → M2) (ρ : M2) (hE : ∑ n, E n = 1)
    (hρ : ρ.trace ≠ 0) : ∑ m, (E m * ρ).trace / ρ.trace = 1 := by
  rw [← Finset.sum_div, ← Matrix.trace_sum, ← Finset.sum_mul, hE, Matrix.one_mul, div_self hρ]

/-- Additivity: the weight of a coarse-grained effect is the sum of the weights. -/
theorem born_additive (E F ρ : M2) :
    ((E + F) * ρ).trace / ρ.trace = (E * ρ).trace / ρ.trace + (F * ρ).trace / ρ.trace := by
  rw [Matrix.add_mul, Matrix.trace_add, add_div]

/-! ## Purity -/

/-- `1 - tr ϱ² = 1/(2γ²)` with `tr ϱ² = (1+v²)/2` and `γ = 1/√(1-v²)`. -/
theorem purity_gamma (v : ℝ) (hv : v ^ 2 < 1) :
    1 - (1 + v ^ 2) / 2 = 1 / (2 * (1 / Real.sqrt (1 - v ^ 2)) ^ 2) := by
  have h : (0 : ℝ) < 1 - v ^ 2 := by linarith
  rw [div_pow, Real.sq_sqrt h.le]
  field_simp
  ring

/-- Purity is `SU(2)`-invariant (indeed `U(2)`-invariant). -/
theorem purity_unitary_invariant (U ρ : M2) (hU : Uᴴ * U = 1) :
    ((U * ρ * Uᴴ) * (U * ρ * Uᴴ)).trace = (ρ * ρ).trace ∧ (U * ρ * Uᴴ).trace = ρ.trace := by
  constructor
  · have : (U * ρ * Uᴴ) * (U * ρ * Uᴴ) = U * (ρ * ρ) * Uᴴ := by
      simp only [Matrix.mul_assoc]
      rw [← Matrix.mul_assoc Uᴴ U, hU, Matrix.one_mul]
    rw [this, Matrix.trace_mul_cycle, hU, Matrix.one_mul]
  · rw [Matrix.trace_mul_cycle, hU, Matrix.one_mul]

/-- A boost has `det = 1` but does not preserve the trace. -/
theorem boost_changes_trace :
    ∃ G : M2, G.det = 1 ∧ Gᴴ = G ∧ (G * 1 * Gᴴ).trace ≠ (1 : M2).trace := by
  refine ⟨!![2, 0; 0, 1 / 2], ?_, ?_, ?_⟩
  · simp [det_fin_two_of]
  · ext i j; fin_cases i <;> fin_cases j <;> simp [conjTranspose_apply]
  · simp [trace_fin_two, Matrix.vecMul, dotProduct, Fin.sum_univ_two, conjTranspose_apply,
      map_ofNat]
    norm_num

/-- `v²` as a function of the channels. -/
noncomputable def vsq (Z : Fin 4 → ℂ) : ℝ :=
  1 - normSq (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2)
      / (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) ^ 2

/-- Purity of the normalised closure `ϱ = J J† / tr(J J†)` is `(1 + v²)/2`. -/
theorem purity_channels (Z : Fin 4 → ℂ)
    (hN : normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3) ≠ 0) :
    let N := normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)
    let ρ := ((1 / (2 * N) : ℝ) : ℂ) • (carrier Z * (carrier Z)ᴴ)
    ρ.trace = 1 ∧ (ρ * ρ).trace = (((1 + vsq Z) / 2 : ℝ) : ℂ) := by
  intro N ρ
  have hNc : (N : ℂ) ≠ 0 := by exact_mod_cast hN
  have hsum : ((normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3) : ℝ) : ℂ) = (N : ℂ) :=
    rfl
  have htr : ρ.trace = 1 := by
    show (((1 / (2 * N) : ℝ) : ℂ) • (carrier Z * (carrier Z)ᴴ)).trace = 1
    rw [trace_smul, trace_carrier_mul_conjTranspose, hsum, smul_eq_mul]
    push_cast
    field_simp
  refine ⟨htr, ?_⟩
  have hv : vsq Z = 1 - normSq (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2) / N ^ 2 := rfl
  rw [purity_unit_trace ρ htr, hv]
  show 1 - 2 * (((1 / (2 * N) : ℝ) : ℂ) • (carrier Z * (carrier Z)ᴴ)).det = _
  rw [det_smul, det_born]
  simp only [Fintype.card_fin]
  push_cast
  field_simp
  ring

/-- Lagrange form: `v² = 4 Σ_{a<b} Im(Z_a conj Z_b)² / (Σ|Z_a|²)²`. -/
theorem vsq_lagrange (Z : Fin 4 → ℂ)
    (hN : normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3) ≠ 0) :
    vsq Z = 4 * ((Z 0 * conj (Z 1)).im ^ 2 + (Z 0 * conj (Z 2)).im ^ 2
        + (Z 0 * conj (Z 3)).im ^ 2 + (Z 1 * conj (Z 2)).im ^ 2 + (Z 1 * conj (Z 3)).im ^ 2
        + (Z 2 * conj (Z 3)).im ^ 2)
      / (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) ^ 2 := by
  rw [← lagrange_channels, vsq]
  field_simp

/-- `0 ≤ v² ≤ 1`: the closure is never spacelike. -/
theorem vsq_bounds (Z : Fin 4 → ℂ)
    (hN : normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3) ≠ 0) :
    0 ≤ vsq Z ∧ vsq Z ≤ 1 := by
  constructor
  · rw [vsq_lagrange Z hN]; positivity
  · have : 0 ≤ normSq (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2)
        / (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) ^ 2 :=
      div_nonneg (normSq_nonneg _) (sq_nonneg _)
    unfold vsq; linarith

/-- Purity is a phase condition: `v = 1 ⟺ Σ Z_a² = det J = 0`. -/
theorem vsq_eq_one_iff (Z : Fin 4 → ℂ)
    (hN : normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3) ≠ 0) :
    vsq Z = 1 ↔ Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2 = 0 := by
  unfold vsq
  rw [sub_eq_self, div_eq_zero_iff]
  simp [hN, normSq_eq_zero]

/-- Real channels (Boltzmann branch alone) give `v = 0`. -/
theorem vsq_real (r : Fin 4 → ℝ) (hr : r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 + r 3 ^ 2 ≠ 0) :
    vsq (fun a => (r a : ℂ)) = 0 := by
  unfold vsq
  have h1 : normSq ((r 0 : ℂ) ^ 2 + (r 1 : ℂ) ^ 2 + (r 2 : ℂ) ^ 2 + (r 3 : ℂ) ^ 2)
      = (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 + r 3 ^ 2) ^ 2 := by
    rw [show (r 0 : ℂ) ^ 2 + (r 1 : ℂ) ^ 2 + (r 2 : ℂ) ^ 2 + (r 3 : ℂ) ^ 2
        = ((r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 + r 3 ^ 2 : ℝ) : ℂ) by push_cast; ring, normSq_ofReal]
    ring
  simp only [h1, normSq_ofReal]
  rw [show r 0 * r 0 + r 1 * r 1 + r 2 * r 2 + r 3 * r 3 = r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 + r 3 ^ 2
    by ring, div_self (pow_ne_zero 2 hr), sub_self]

/-- `v` is unchanged by a common complex factor (overall scale and global phase). -/
theorem vsq_scale (l : ℂ) (hl : l ≠ 0) (Z : Fin 4 → ℂ) : vsq (fun a => l * Z a) = vsq Z := by
  unfold vsq
  have h1 : (l * Z 0) ^ 2 + (l * Z 1) ^ 2 + (l * Z 2) ^ 2 + (l * Z 3) ^ 2
      = l ^ 2 * (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2) := by ring
  have hl' : normSq l ≠ 0 := (normSq_pos.mpr hl).ne'
  simp only [h1, map_mul, map_pow]
  rw [show normSq l * normSq (Z 0) + normSq l * normSq (Z 1) + normSq l * normSq (Z 2)
      + normSq l * normSq (Z 3)
      = normSq l * (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) by ring,
    mul_pow, mul_div_mul_left _ _ (pow_ne_zero 2 hl')]

/-- `v` is invariant under the real `SO(4)` rotations of the channel index. -/
theorem vsq_rotation (p : Fin 4 → ℝ) (hp : p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = 1)
    (Z : Fin 4 → ℂ) : vsq (actC (Lmat p) Z) = vsq Z ∧ vsq (actC (Rmat p) Z) = vsq Z := by
  unfold vsq
  rw [sumSq_left, normSq_left, sumSq_right, normSq_right, hp]
  simp

/-- The thermal branch switches off as `τ_E → 0⁺` when `H > 0`. -/
theorem boltzmann_limit (Hc : ℝ) (hH : 0 < Hc) :
    Tendsto (fun τ : ℝ => Real.exp (-Hc / τ)) (𝓝[>] 0) (𝓝 0) := by
  have h1 : Tendsto (fun τ : ℝ => -Hc * τ⁻¹) (𝓝[>] 0) atBot :=
    (tendsto_inv_nhdsGT_zero (𝕜 := ℝ)).const_mul_atTop_of_neg (neg_lt_zero.mpr hH)
  have h2 : (fun τ : ℝ => Real.exp (-Hc / τ)) = fun τ => Real.exp (-Hc * τ⁻¹) := by
    funext τ; rw [div_eq_mul_inv]
  rw [h2]
  exact Real.tendsto_exp_atBot.comp h1

/-! ## Killing form: Ad-invariance -/

theorem killing_ad_invariant (G Gi A B : M2) (h : Gi * G = 1) :
    LinearMap.trace ℂ M2 (adM (G * A * Gi) ∘ₗ adM (G * B * Gi))
      = LinearMap.trace ℂ M2 (adM A ∘ₗ adM B) := by
  have t1 : ∀ X : M2, (G * X * Gi).trace = X.trace := fun X => by
    rw [Matrix.trace_mul_cycle, h, Matrix.one_mul]
  have t2 : G * A * Gi * (G * B * Gi) = G * (A * B) * Gi := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Gi G, h, Matrix.one_mul]
  rw [killing_gl2, killing_gl2, t2, t1, t1, t1]

/-! ## SU(2) conjugation acts on the Bloch vector by SO(3) -/

/-- Rotation matrix of the quaternion `p` (homogeneous of degree two). -/
def Rot (p : Fin 4 → ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![p 0 ^ 2 + p 1 ^ 2 - p 2 ^ 2 - p 3 ^ 2, 2 * (p 1 * p 2 - p 0 * p 3), 2 * (p 1 * p 3 + p 0 * p 2);
     2 * (p 1 * p 2 + p 0 * p 3), p 0 ^ 2 - p 1 ^ 2 + p 2 ^ 2 - p 3 ^ 2, 2 * (p 2 * p 3 - p 0 * p 1);
     2 * (p 1 * p 3 - p 0 * p 2), 2 * (p 2 * p 3 + p 0 * p 1), p 0 ^ 2 - p 1 ^ 2 - p 2 ^ 2 + p 3 ^ 2]

/-- `U (a·σ) U† = (R a)·σ` for `U` the carrier of a real quaternion. -/
theorem su2_acts_by_rotation (p : Fin 4 → ℝ) (a : Fin 3 → ℝ) :
    carrier (fun k => (p k : ℂ)) * dotσ a * (carrier (fun k => (p k : ℂ)))ᴴ
      = dotσ (Rot p *ᵥ a) := by
  rw [carrier_eq, dotσ_eq, dotσ_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Rot, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply, Matrix.mulVec,
      dotProduct, Fin.sum_univ_three] <;>
    ring_nf <;> simp [Complex.I_sq] <;> cring

theorem Rot_orthogonal (p : Fin 4 → ℝ) :
    (Rot p)ᵀ * Rot p
      = ((p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2) ^ 2) • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Rot, Matrix.mul_apply, Fin.sum_univ_three, Matrix.transpose_apply] <;> ring

theorem Rot_det (p : Fin 4 → ℝ) :
    (Rot p).det = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2) ^ 3 := by
  simp [Rot, det_fin_three]
  ring

/-! ## Uniqueness of α = 2 for `R_n = n·σ` -/

theorem dotσ_hermitian (n : Fin 3 → ℝ) : (dotσ n)ᴴ = dotσ n := by
  rw [dotσ_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [conjTranspose_apply] <;> cring

theorem dotσ_sq (n : Fin 3 → ℝ) (hn : n 0 ^ 2 + n 1 ^ 2 + n 2 ^ 2 = 1) :
    dotσ n * dotσ n = 1 := by
  have h : dotσ n * dotσ n = algebraMap ℝ M2 (Q3 n) := pauliVec_sq n
  have hQ : Q3 n = 1 := by
    simp [Q3, QuadraticMap.weightedSumSquares_apply, Fin.sum_univ_three]
    linarith
  rw [hQ] at h
  simpa using h

theorem dotσ_trace (n : Fin 3 → ℝ) : (dotσ n).trace = 0 := by
  rw [dotσ_eq]; simp [trace_fin_two]

/-- A traceless involution is linearly independent of the identity. -/
theorem indep_of_trace_zero (R : M2) (htr : R.trace = 0) (hsq : R * R = 1) (a b : ℂ)
    (h : a • (1 : M2) + b • R = 0) : a = 0 ∧ b = 0 := by
  have ht := congrArg Matrix.trace h
  simp [htr, trace_fin_two] at ht
  have ha : a = 0 := ht
  refine ⟨ha, ?_⟩
  rw [ha, zero_smul, zero_add] at h
  have h2 := congrArg (fun M => M * R) h
  simp only [Matrix.smul_mul, hsq, Matrix.zero_mul] at h2
  have h3 := congrFun (congrFun h2 0) 0
  simpa using h3

/-- Uniqueness for the actual reflection `R_n = n·σ`, with no extra hypothesis. -/
theorem uniq_alpha_dotσ (n : Fin 3 → ℝ) (hn : n 0 ^ 2 + n 1 ^ 2 + n 2 ^ 2 = 1)
    (α l : ℂ) (hl : l * conj l = 1)
    (h : α • (1 + l • dotσ n) = (1 + l • dotσ n) * (1 + l • dotσ n)ᴴ) :
    α = 2 ∧ (l = 1 ∨ l = -1) :=
  uniq_alpha (dotσ n) (dotσ_hermitian n) (dotσ_sq n hn)
    (indep_of_trace_zero _ (dotσ_trace n) (dotσ_sq n hn)) α l hl h

/-! ## Bell: the joint closure is the singlet projector, and Tsirelson's bound -/

theorem jointClosure_idem : jointClosure * jointClosure = jointClosure := by
  ext ⟨i, j⟩ ⟨k, l⟩
  fin_cases i <;> fin_cases j <;> fin_cases k <;> fin_cases l <;>
    simp [jointClosure, Matrix.mul_apply, Fintype.sum_prod_type, Fin.sum_univ_two,
      e1, e2, e3, σx, σy, σz] <;>
    ring_nf <;> simp [Complex.I_sq] <;> cring

theorem jointClosure_hermitian : jointClosureᴴ = jointClosure := by
  ext ⟨i, j⟩ ⟨k, l⟩
  fin_cases i <;> fin_cases j <;> fin_cases k <;> fin_cases l <;>
    simp [jointClosure, conjTranspose_apply, e1, e2, e3, σx, σy, σz] <;> cring

theorem bell_marginal_right (b : Fin 3 → ℝ) :
    (jointClosure * ((1 : M2) ⊗ₖ dotσ b)).trace = 0 := by
  rw [dotσ_eq]
  simp [jointClosure, Matrix.trace, Matrix.mul_apply, Fintype.sum_prod_type, Fin.sum_univ_two,
    e1, e2, e3, σx, σy, σz]
  cring

/-- Tsirelson's bound for `E(a,b) = -a·b` and arbitrary unit settings: `|S| ≤ 2√2`. -/
theorem chsh_tsirelson (a a' b b' : EuclideanSpace ℝ (Fin 3))
    (ha : ‖a‖ = 1) (ha' : ‖a'‖ = 1) (hb : ‖b‖ = 1) (hb' : ‖b'‖ = 1) :
    |(-⟪a, b⟫) - (-⟪a, b'⟫) + (-⟪a', b⟫) + (-⟪a', b'⟫)| ≤ 2 * Real.sqrt 2 := by
  have e : (-⟪a, b⟫) - (-⟪a, b'⟫) + (-⟪a', b⟫) + (-⟪a', b'⟫)
      = ⟪a, b' - b⟫ - ⟪a', b + b'⟫ := by
    rw [inner_sub_right, inner_add_right]; ring
  have h1 : |⟪a, b' - b⟫| ≤ ‖b' - b‖ := by
    have := abs_real_inner_le_norm a (b' - b); rwa [ha, one_mul] at this
  have h2 : |⟪a', b + b'⟫| ≤ ‖b + b'‖ := by
    have := abs_real_inner_le_norm a' (b + b'); rwa [ha', one_mul] at this
  have hp : ‖b' - b‖ ^ 2 + ‖b + b'‖ ^ 2 = 4 := by
    rw [norm_sub_sq_real, norm_add_sq_real, hb, hb', real_inner_comm b b']; ring
  have hx : |⟪a, b' - b⟫ - ⟪a', b + b'⟫| ≤ ‖b' - b‖ + ‖b + b'‖ :=
    (abs_sub _ _).trans (add_le_add h1 h2)
  have hs : ‖b' - b‖ + ‖b + b'‖ ≤ 2 * Real.sqrt 2 := by
    have key : (‖b' - b‖ + ‖b + b'‖) ^ 2 ≤ (2 * Real.sqrt 2) ^ 2 := by
      nlinarith [sq_nonneg (‖b' - b‖ - ‖b + b'‖), Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)]
    exact (pow_le_pow_iff_left₀ (by positivity) (by positivity)
      (by norm_num : (2 : ℕ) ≠ 0)).mp key
  rw [e]
  exact hx.trans hs

end CalChecks
