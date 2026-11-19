/-
Machine-checked versions of the algebraic identities in
`onesheet_letter_native_v9.pdf`.  Lean 4 + Mathlib; independent of the sympy script
in `../python/check_onesheet_sympy.py`.

Conventions follow the one-sheet: e₀ = 1, e_j = -i σ_j,  J = Σ_a Z_a e_a ∈ M₂(ℂ).
-/
import Mathlib

open Matrix Complex
open scoped ComplexConjugate

namespace CalChecks

abbrev M2 := Matrix (Fin 2) (Fin 2) ℂ

/-! ## Pauli matrices and the quaternion units e_j = -i σ_j -/

def σx : M2 := !![0, 1; 1, 0]
def σy : M2 := !![0, -I; I, 0]
def σz : M2 := !![1, 0; 0, -1]

def e1 : M2 := (-I) • σx
def e2 : M2 := (-I) • σy
def e3 : M2 := (-I) • σz

/-- The carrier `J = Z₀ + Z₁e₁ + Z₂e₂ + Z₃e₃`. -/
def carrier (Z : Fin 4 → ℂ) : M2 := Z 0 • 1 + Z 1 • e1 + Z 2 • e2 + Z 3 • e3

/-- Hermitian slice `X = t + xσx + yσy + zσz` with real coordinates. -/
def herm (t x y z : ℝ) : M2 := (t : ℂ) • 1 + (x : ℂ) • σx + (y : ℂ) • σy + (z : ℂ) • σz

/-- Explicit entries of the carrier. -/
theorem carrier_eq (Z : Fin 4 → ℂ) :
    carrier Z = !![Z 0 - I * Z 3, -(I * Z 1) - Z 2; -(I * Z 1) + Z 2, Z 0 + I * Z 3] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [carrier, e1, e2, e3, σx, σy, σz] <;> ring

/-! ## Quaternion relations -/

theorem e1_sq : e1 * e1 = -1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e1, σx]
theorem e2_sq : e2 * e2 = -1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e2, σy]
theorem e3_sq : e3 * e3 = -1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e3, σz]
theorem e1_e2 : e1 * e2 = e3 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e1, e2, e3, σx, σy, σz]
theorem e2_e3 : e2 * e3 = e1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e1, e2, e3, σx, σy, σz]
theorem e3_e1 : e3 * e1 = e2 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e1, e2, e3, σx, σy, σz]
theorem e2_e1 : e2 * e1 = -e3 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [e1, e2, e3, σx, σy, σz]

/-! ## Born rule block -/

/-- `det J = Σ_a Z_a²`. -/
theorem det_carrier (Z : Fin 4 → ℂ) :
    (carrier Z).det = Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2 := by
  rw [carrier_eq, det_fin_two_of]
  linear_combination (-(Z 1 ^ 2 + Z 3 ^ 2)) * Complex.I_sq

/-- `tr(J J†) = 2 Σ_a |Z_a|²`. -/
theorem trace_carrier_mul_conjTranspose (Z : Fin 4 → ℂ) :
    (carrier Z * (carrier Z)ᴴ).trace =
      2 * ((normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3) : ℝ) : ℂ) := by
  rw [carrier_eq]
  simp [trace_fin_two, Matrix.vecMul, dotProduct, Fin.sum_univ_two, conjTranspose_apply]
  apply Complex.ext <;> simp [normSq_apply] <;> ring

/-- Right `U(2)` invariance of the Born quadratic. -/
theorem born_right_unitary (J U : M2) (hU : U * Uᴴ = 1) :
    (J * U) * (J * U)ᴴ = J * Jᴴ := by
  rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc U, hU, Matrix.one_mul]

/-- Congruence covariance under left `GL(2,ℂ)`. -/
theorem born_left_congruence (G J : M2) :
    (G * J) * (G * J)ᴴ = G * (J * Jᴴ) * Gᴴ := by
  simp [conjTranspose_mul, Matrix.mul_assoc]

/-- Hilbert–Schmidt form is `U(2)_L × U(2)_R` invariant. -/
theorem hs_invariant (J U V : M2) (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) :
    ((U * J * Vᴴ) * (U * J * Vᴴ)ᴴ).trace = (J * Jᴴ).trace := by
  have h : (U * J * Vᴴ) * (U * J * Vᴴ)ᴴ = U * (J * Jᴴ) * Uᴴ := by
    simp only [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Vᴴ V, hV, Matrix.one_mul]
  rw [h, Matrix.trace_mul_cycle, hU, Matrix.one_mul]

/-- Dual pairing: `tr(E ϱ)` is invariant under `ϱ ↦ GϱG†`, `E ↦ G⁻†EG⁻¹`. -/
theorem povm_pairing (G Gi E ρ : M2) (h : Gi * G = 1) :
    ((Giᴴ * E * Gi) * (G * ρ * Gᴴ)).trace = (E * ρ).trace := by
  have h' : Gᴴ * Giᴴ = 1 := by rw [← conjTranspose_mul, h, conjTranspose_one]
  calc ((Giᴴ * E * Gi) * (G * ρ * Gᴴ)).trace
      = (Giᴴ * (E * ρ * Gᴴ)).trace := by
        congr 1
        simp only [Matrix.mul_assoc]
        rw [← Matrix.mul_assoc Gi G, h, Matrix.one_mul]
    _ = ((E * ρ * Gᴴ) * Giᴴ).trace := Matrix.trace_mul_comm _ _
    _ = (E * ρ).trace := by rw [Matrix.mul_assoc, h', Matrix.mul_one]

/-! ## Lorentz signature block -/

/-- Hermitian projector `H(q) = (q + q†)/2`. -/
noncomputable def H (q : M2) : M2 := (1 / 2 : ℂ) • (q + qᴴ)
/-- Anti-Hermitian projector `A(q) = (q - q†)/2`. -/
noncomputable def A (q : M2) : M2 := (1 / 2 : ℂ) • (q - qᴴ)

theorem H_add_A (q : M2) : H q + A q = q := by
  unfold H A
  rw [← smul_add]
  have : q + qᴴ + (q - qᴴ) = (2 : ℂ) • q := by rw [two_smul]; abel
  rw [this, smul_smul]; norm_num

theorem H_isHermitian (q : M2) : (H q)ᴴ = H q := by
  unfold H
  rw [conjTranspose_smul, conjTranspose_add, conjTranspose_conjTranspose, add_comm]
  simp

theorem A_antiHermitian (q : M2) : (A q)ᴴ = -A q := by
  unfold A
  rw [conjTranspose_smul, conjTranspose_sub, conjTranspose_conjTranspose, ← smul_neg, neg_sub]
  simp

theorem H_idem (q : M2) : H (H q) = H q := by
  have h := H_isHermitian q
  show (1 / 2 : ℂ) • (H q + (H q)ᴴ) = H q
  rw [h, ← two_smul ℂ (H q), smul_smul]; norm_num

/-- `H` commutes with congruence `q ↦ G q G†`: it is an intertwiner. -/
theorem H_intertwines (G q : M2) : H (G * q * Gᴴ) = G * H q * Gᴴ := by
  unfold H
  simp [conjTranspose_mul, Matrix.mul_assoc, Matrix.mul_add, Matrix.add_mul]

theorem A_intertwines (G q : M2) : A (G * q * Gᴴ) = G * A q * Gᴴ := by
  unfold A
  simp [conjTranspose_mul, Matrix.mul_assoc, Matrix.mul_sub, Matrix.sub_mul]

theorem herm_eq (t x y z : ℝ) :
    herm t x y z = !![(t : ℂ) + z, x - I * y; x + I * y, (t : ℂ) - z] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [herm, σx, σy, σz] <;> ring

theorem herm_isHermitian (t x y z : ℝ) : (herm t x y z)ᴴ = herm t x y z := by
  rw [herm_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [conjTranspose_apply] <;> ring

/-- `det X = t² - x² - y² - z²`: the determinant is the Minkowski form. -/
theorem det_herm (t x y z : ℝ) :
    (herm t x y z).det = ((t ^ 2 - x ^ 2 - y ^ 2 - z ^ 2 : ℝ) : ℂ) := by
  rw [herm_eq, det_fin_two_of]
  push_cast
  ring_nf
  simp [Complex.I_sq]
  ring

/-- `det(G X G†) = |det G|² det X`: invariant for `SL(2,ℂ)`, conformal for `GL(2,ℂ)`. -/
theorem det_congruence (G X : M2) :
    (G * X * Gᴴ).det = (normSq G.det : ℂ) * X.det := by
  rw [det_mul, det_mul, det_conjTranspose, normSq_eq_conj_mul_self]
  simp only [RCLike.star_def]
  ring

theorem det_congruence_SL (G X : M2) (hG : G.det = 1) : (G * X * Gᴴ).det = X.det := by
  rw [det_congruence, hG]; simp

/-- Cayley–Hamilton trace identity for 2×2: `tr(M²) = (tr M)² - 2 det M`. -/
theorem trace_sq_fin_two (M : M2) : (M * M).trace = M.trace ^ 2 - 2 * M.det := by
  simp [trace_fin_two, det_fin_two, mul_apply, Fin.sum_univ_two]
  ring

/-- Purity at unit trace: `tr ϱ² = 1 - 2 det ϱ`. -/
theorem purity_unit_trace (ρ : M2) (h : ρ.trace = 1) : (ρ * ρ).trace = 1 - 2 * ρ.det := by
  rw [trace_sq_fin_two, h]; ring

/-- Purity of the normalised Hermitian slice: `tr ϱ² = (1 + v²)/2`, `v² = |x|²/t²`. -/
theorem purity_velocity (t x y z : ℝ) (ht : t ≠ 0) :
    (((1 / (2 * t) : ℝ) : ℂ) • herm t x y z * (((1 / (2 * t) : ℝ) : ℂ) • herm t x y z)).trace
      = (((1 + (x ^ 2 + y ^ 2 + z ^ 2) / t ^ 2) / 2 : ℝ) : ℂ) := by
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, trace_smul, trace_sq_fin_two, det_herm]
  have htr : (herm t x y z).trace = ((2 * t : ℝ) : ℂ) := by
    rw [herm_eq]; simp [trace_fin_two]; ring
  rw [htr, smul_eq_mul]
  norm_cast
  field_simp
  ring

/-! ## Purity from the channels -/

/-- Lagrange identity: `(Σ|Z_a|²)² - |Σ Z_a²|² = 4 Σ_{a<b} Im(Z_a conj Z_b)²`. -/
theorem lagrange_channels (Z : Fin 4 → ℂ) :
    (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) ^ 2
        - normSq (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2)
      = 4 * ((Z 0 * conj (Z 1)).im ^ 2 + (Z 0 * conj (Z 2)).im ^ 2 + (Z 0 * conj (Z 3)).im ^ 2
           + (Z 1 * conj (Z 2)).im ^ 2 + (Z 1 * conj (Z 3)).im ^ 2
           + (Z 2 * conj (Z 3)).im ^ 2) := by
  simp [normSq_apply, pow_two]
  ring

/-- `|w|² - Re(w²) = 2 (Im w)²`. -/
theorem normSq_sub_re_sq (w : ℂ) : normSq w - (w ^ 2).re = 2 * w.im ^ 2 := by
  simp [normSq_apply, pow_two]; ring

/-- `det(J J†) = |Σ Z_a²|²`, so the closure is rank ≤ 1 (null, pure) iff `Σ Z_a² = 0`. -/
theorem det_born (Z : Fin 4 → ℂ) :
    (carrier Z * (carrier Z)ᴴ).det
      = (normSq (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2) : ℂ) := by
  rw [det_mul, det_conjTranspose, det_carrier, normSq_eq_conj_mul_self]
  simp only [RCLike.star_def]
  ring

/-- Real channels (Boltzmann branch alone): `J J† = (Σ Z_a²)·1`, maximally mixed. -/
theorem born_real_channels (r : Fin 4 → ℝ) :
    carrier (fun a => (r a : ℂ)) * (carrier (fun a => (r a : ℂ)))ᴴ
      = ((r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 + r 3 ^ 2 : ℝ) : ℂ) • (1 : M2) := by
  rw [carrier_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [mul_apply, Fin.sum_univ_two, conjTranspose_apply] <;>
    ring_nf <;> simp [Complex.I_sq] <;> ring

/-! ## Generator slice u(2) -/

/-- Generator `A = y₀(i·1) + y_j e_j`. -/
def gen (y0 y1 y2 y3 : ℝ) : M2 :=
  ((y0 : ℂ) * I) • 1 + (y1 : ℂ) • e1 + (y2 : ℂ) • e2 + (y3 : ℂ) • e3

theorem gen_eq_I_smul_herm (y0 y1 y2 y3 : ℝ) :
    gen y0 y1 y2 y3 = I • herm y0 (-y1) (-y2) (-y3) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [gen, herm, e1, e2, e3, σx, σy, σz] <;> ring_nf <;> simp [Complex.I_sq] <;> ring

/-- `det A = -(y⁰)² + |y|²`: signature (-,+,+,+) on the generator slice. -/
theorem det_gen (y0 y1 y2 y3 : ℝ) :
    (gen y0 y1 y2 y3).det = ((-(y0 ^ 2) + y1 ^ 2 + y2 ^ 2 + y3 ^ 2 : ℝ) : ℂ) := by
  rw [gen_eq_I_smul_herm, det_smul, det_herm]
  simp [Complex.I_sq]
  ring

/-- Multiplication by `i` flips the sign of the Minkowski form: `det(iX) = -det X`. -/
theorem det_I_smul (X : M2) : (I • X).det = -X.det := by
  rw [det_smul]; simp [Complex.I_sq]

/-- On `su(2)`: `det A = -(1/2) tr A²`. -/
theorem det_su2 (y1 y2 y3 : ℝ) :
    (gen 0 y1 y2 y3).det = -(1 / 2) * (gen 0 y1 y2 y3 * gen 0 y1 y2 y3).trace := by
  have htr : (gen 0 y1 y2 y3).trace = 0 := by
    simp [gen, e1, e2, e3, σx, σy, σz, trace_fin_two]
  rw [trace_sq_fin_two, htr]; ring

/-! ## Bell / CHSH -/

/-- A fixed ±1 answer sheet gives `S = ±2`. -/
theorem chsh_local (a a' b b' : ℤ) (ha : a = 1 ∨ a = -1) (ha' : a' = 1 ∨ a' = -1)
    (hb : b = 1 ∨ b = -1) (hb' : b' = 1 ∨ b' = -1) :
    a * (b - b') + a' * (b + b') = 2 ∨ a * (b - b') + a' * (b + b') = -2 := by
  rcases ha with rfl | rfl <;> rcases ha' with rfl | rfl <;>
    rcases hb with rfl | rfl <;> rcases hb' with rfl | rfl <;> simp

/-- Tsirelson-side bound used in the text: `√(2-2c) + √(2+2c) ≤ 2√2`. -/
theorem chsh_quantum_bound (c : ℝ) (h1 : -1 ≤ c) (h2 : c ≤ 1) :
    Real.sqrt (2 - 2 * c) + Real.sqrt (2 + 2 * c) ≤ 2 * Real.sqrt 2 := by
  have hu : (0 : ℝ) ≤ 2 - 2 * c := by linarith
  have hw : (0 : ℝ) ≤ 2 + 2 * c := by linarith
  have h2' : (0 : ℝ) ≤ 2 := by norm_num
  have key : (Real.sqrt (2 - 2 * c) + Real.sqrt (2 + 2 * c)) ^ 2 ≤ (2 * Real.sqrt 2) ^ 2 := by
    nlinarith [sq_nonneg (Real.sqrt (2 - 2 * c) - Real.sqrt (2 + 2 * c)),
      Real.sq_sqrt hu, Real.sq_sqrt hw, Real.sq_sqrt h2']
  have hpos : 0 ≤ Real.sqrt (2 - 2 * c) + Real.sqrt (2 + 2 * c) := by positivity
  exact (pow_le_pow_iff_left₀ hpos (by positivity) (by norm_num : (2 : ℕ) ≠ 0)).mp key

/-- Bound is attained at `b ⟂ b'` (`c = 0`). -/
theorem chsh_quantum_attained :
    Real.sqrt (2 - 2 * 0) + Real.sqrt (2 + 2 * 0) = 2 * Real.sqrt 2 := by
  norm_num; ring

/-! ## Uniqueness of α = 2 -/

/-- With `R† = R`, `R² = 1`: `J_λ J_λ† = (1 + |λ|²) + (λ + conj λ) R`. -/
theorem uniq_expand (R : M2) (hH : Rᴴ = R) (hsq : R * R = 1) (l : ℂ) :
    (1 + l • R) * (1 + l • R)ᴴ = (1 + l * conj l) • (1 : M2) + (l + conj l) • R := by
  rw [conjTranspose_add, conjTranspose_one, conjTranspose_smul, hH]
  have h : (1 + l • R) * (1 + star l • R)
      = 1 + l • R + star l • R + (l * star l) • (R * R) := by
    simp only [Matrix.add_mul, Matrix.mul_add, Matrix.one_mul, Matrix.mul_one,
      Matrix.smul_mul, Matrix.mul_smul]
    module
  rw [h, hsq]
  simp only [RCLike.star_def]
  module

/-- If `1, R` are linearly independent, `α J_λ = J_λ J_λ†` with `|λ| = 1` forces
`α = 2` and `λ = ±1`. -/
theorem uniq_alpha (R : M2) (hH : Rᴴ = R) (hsq : R * R = 1)
    (hind : ∀ a b : ℂ, a • (1 : M2) + b • R = 0 → a = 0 ∧ b = 0)
    (α l : ℂ) (hl : l * conj l = 1) (h : α • (1 + l • R) = (1 + l • R) * (1 + l • R)ᴴ) :
    α = 2 ∧ (l = 1 ∨ l = -1) := by
  rw [uniq_expand R hH hsq, hl] at h
  have h0 : (α - 2) • (1 : M2) + (α * l - (l + conj l)) • R = 0 := by
    have : α • (1 + l • R) - ((1 + 1 : ℂ) • (1 : M2) + (l + conj l) • R) = 0 := sub_eq_zero.mpr h
    rw [← this]
    module
  obtain ⟨ha, hb⟩ := hind _ _ h0
  have hα : α = 2 := sub_eq_zero.mp ha
  refine ⟨hα, ?_⟩
  rw [hα] at hb
  have hc : conj l = l := by linear_combination -hb
  rw [hc] at hl
  have : (l - 1) * (l + 1) = 0 := by linear_combination hl
  rcases mul_eq_zero.mp this with h1 | h1
  · left; exact sub_eq_zero.mp h1
  · right; exact eq_neg_of_add_eq_zero_left h1

/-- At `α = 2`, `λ = ±1`: `J/2 = (1 ± R)/2` is a projector. -/
theorem uniq_projector (R : M2) (hsq : R * R = 1) (s : ℂ) (hs : s * s = 1) :
    ((1 / 2 : ℂ) • (1 + s • R)) * ((1 / 2 : ℂ) • (1 + s • R)) = (1 / 2 : ℂ) • (1 + s • R) := by
  have h : (1 + s • R) * (1 + s • R) = (2 : ℂ) • (1 + s • R) := by
    have h' : (1 + s • R) * (1 + s • R) = 1 + s • R + s • R + (s * s) • (R * R) := by
      simp only [Matrix.add_mul, Matrix.mul_add, Matrix.one_mul, Matrix.mul_one,
        Matrix.smul_mul, Matrix.mul_smul]
      module
    rw [h', hsq, hs]
    module
  rw [Matrix.smul_mul, Matrix.mul_smul, h]
  module

end CalChecks
