/-
Third batch: the representation-theory items that were previously checked only in sympy.
-/
import Mathlib
import CalChecks.Extras

open Matrix Complex
open scoped ComplexConjugate

namespace CalChecks

/-! ## Clebsch–Gordan for SL(2,ℂ) labels (j₁, j₂)

Two formulations: character identities, and equality of weight multisets.
Weights are stored doubled, so spin n/2 has weights n, n-2, …, -n. -/

/-- `1/2 ⊗ 1/2 = 0 ⊕ 1` for SU(2), as a character identity. -/
theorem char_half_sq (x : ℂ) (hx : x ≠ 0) :
    (x + x⁻¹) ^ 2 = 1 + (x ^ 2 + 1 + x⁻¹ ^ 2) := by
  field_simp; ring

/-- `(1/2,1/2) ⊗ (1/2,1/2) = (0,0) ⊕ (1,0) ⊕ (0,1) ⊕ (1,1)`, as a character identity. -/
theorem char_vector_sq (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) :
    ((x + x⁻¹) * (y + y⁻¹)) ^ 2
      = 1 + (x ^ 2 + 1 + x⁻¹ ^ 2) + (y ^ 2 + 1 + y⁻¹ ^ 2)
          + (x ^ 2 + 1 + x⁻¹ ^ 2) * (y ^ 2 + 1 + y⁻¹ ^ 2) := by
  field_simp; ring

/-- Symmetric square of the vector: `(0,0) ⊕ (1,1)` (metric trace + traceless symmetric). -/
theorem char_vector_sym (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) :
    (((x + x⁻¹) * (y + y⁻¹)) ^ 2 + (x ^ 2 + x⁻¹ ^ 2) * (y ^ 2 + y⁻¹ ^ 2)) / 2
      = 1 + (x ^ 2 + 1 + x⁻¹ ^ 2) * (y ^ 2 + 1 + y⁻¹ ^ 2) := by
  field_simp; ring

/-- Antisymmetric square of the vector: `(1,0) ⊕ (0,1)` (self-dual + anti-self-dual 2-forms). -/
theorem char_vector_alt (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) :
    (((x + x⁻¹) * (y + y⁻¹)) ^ 2 - (x ^ 2 + x⁻¹ ^ 2) * (y ^ 2 + y⁻¹ ^ 2)) / 2
      = (x ^ 2 + 1 + x⁻¹ ^ 2) + (y ^ 2 + 1 + y⁻¹ ^ 2) := by
  field_simp; ring

/-- Doubled weights of the spin-`n/2` representation of SU(2). -/
def spinW (n : ℕ) : Multiset ℤ := (Multiset.range (n + 1)).map fun k => (n : ℤ) - 2 * k

/-- Doubled weights of the `(m/2, n/2)` representation. -/
def repW (m n : ℕ) : Multiset (ℤ × ℤ) := spinW m ×ˢ spinW n

/-- Weights of a tensor product: all pairwise sums. -/
def tensorW (a b : Multiset (ℤ × ℤ)) : Multiset (ℤ × ℤ) :=
  (a ×ˢ b).map fun p => (p.1.1 + p.2.1, p.1.2 + p.2.2)

/-- Weight multisets: `(1/2,1/2)⊗(1/2,1/2) = (0,0)+(1,0)+(0,1)+(1,1)`. -/
theorem weights_vector_sq :
    tensorW (repW 1 1) (repW 1 1) = repW 0 0 + repW 2 0 + repW 0 2 + repW 2 2 := by
  decide

/-- Weight multisets: `(1/2,0)⊗(0,1/2) = (1/2,1/2)`. -/
theorem weights_weyl_tensor : tensorW (repW 1 0) (repW 0 1) = repW 1 1 := by
  decide

/-- Weight multisets for SU(2): `1/2 ⊗ 1/2 = 0 ⊕ 1`. -/
theorem weights_half_sq :
    ((spinW 1 ×ˢ spinW 1).map fun p => p.1 + p.2) = spinW 0 + spinW 2 := by
  decide

/-! ## SO(4) on the channel index

Left and right multiplication by a real quaternion act on `(Z_a)` by real 4×4 matrices
that are orthogonal up to the scale `|p|²`, and they preserve both `Σ Z_a²` and `Σ |Z_a|²`. -/

/-- Matrix of left multiplication by the quaternion `p`. -/
def Lmat (p : Fin 4 → ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![p 0, -p 1, -p 2, -p 3;
     p 1,  p 0, -p 3,  p 2;
     p 2,  p 3,  p 0, -p 1;
     p 3, -p 2,  p 1,  p 0]

/-- Matrix of right multiplication by the quaternion `p`. -/
def Rmat (p : Fin 4 → ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![p 0, -p 1, -p 2, -p 3;
     p 1,  p 0,  p 3, -p 2;
     p 2, -p 3,  p 0,  p 1;
     p 3,  p 2, -p 1,  p 0]

/-- Action of a real 4×4 matrix on complex channel coordinates. -/
def actC (M : Matrix (Fin 4) (Fin 4) ℝ) (Z : Fin 4 → ℂ) : Fin 4 → ℂ :=
  fun a => (M a 0 : ℂ) * Z 0 + (M a 1 : ℂ) * Z 1 + (M a 2 : ℂ) * Z 2 + (M a 3 : ℂ) * Z 3

theorem carrier_left_mul (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    carrier (fun a => (p a : ℂ)) * carrier Z = carrier (actC (Lmat p) Z) := by
  rw [carrier_eq, carrier_eq, carrier_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [actC, Lmat, Matrix.mul_apply, Fin.sum_univ_two] <;>
    ring_nf <;> simp [Complex.I_sq] <;> ring

theorem carrier_right_mul (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    carrier Z * carrier (fun a => (p a : ℂ)) = carrier (actC (Rmat p) Z) := by
  rw [carrier_eq, carrier_eq, carrier_eq]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [actC, Rmat, Matrix.mul_apply, Fin.sum_univ_two] <;>
    ring_nf <;> simp [Complex.I_sq] <;> ring

theorem Lmat_orthogonal (p : Fin 4 → ℝ) :
    (Lmat p)ᵀ * Lmat p = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2) • (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Lmat, Matrix.mul_apply, Fin.sum_univ_four, Matrix.transpose_apply] <;> ring

theorem Rmat_orthogonal (p : Fin 4 → ℝ) :
    (Rmat p)ᵀ * Rmat p = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2) • (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Rmat, Matrix.mul_apply, Fin.sum_univ_four, Matrix.transpose_apply] <;> ring

/-- The two factors commute: `SU(2)_L` and `SU(2)_R`. -/
theorem Lmat_Rmat_comm (p q : Fin 4 → ℝ) : Lmat p * Rmat q = Rmat q * Lmat p := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Lmat, Rmat, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

/-- `det L = |p|⁴ > 0`: orientation preserving, so `SO(4)` (not just `O(4)`) at `|p| = 1`. -/
theorem Lmat_det (p : Fin 4 → ℝ) :
    (Lmat p).det = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2) ^ 2 := by
  simp [Lmat, Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  ring

theorem Rmat_det (p : Fin 4 → ℝ) :
    (Rmat p).det = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2) ^ 2 := by
  simp [Rmat, Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  ring

/-- The bilinear invariant `Σ Z_a² = det J` is preserved (complex `Z`, real rotation). -/
theorem sumSq_left (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    actC (Lmat p) Z 0 ^ 2 + actC (Lmat p) Z 1 ^ 2 + actC (Lmat p) Z 2 ^ 2 + actC (Lmat p) Z 3 ^ 2
      = ((p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 : ℝ) : ℂ)
          * (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2) := by
  simp [actC, Lmat]; ring

theorem sumSq_right (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    actC (Rmat p) Z 0 ^ 2 + actC (Rmat p) Z 1 ^ 2 + actC (Rmat p) Z 2 ^ 2 + actC (Rmat p) Z 3 ^ 2
      = ((p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 : ℝ) : ℂ)
          * (Z 0 ^ 2 + Z 1 ^ 2 + Z 2 ^ 2 + Z 3 ^ 2) := by
  simp [actC, Rmat]; ring

/-- The Hermitian invariant `Σ |Z_a|²` is preserved. -/
theorem normSq_left (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    normSq (actC (Lmat p) Z 0) + normSq (actC (Lmat p) Z 1) + normSq (actC (Lmat p) Z 2)
        + normSq (actC (Lmat p) Z 3)
      = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2)
          * (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) := by
  simp [actC, Lmat, normSq_apply]; ring

theorem normSq_right (p : Fin 4 → ℝ) (Z : Fin 4 → ℂ) :
    normSq (actC (Rmat p) Z 0) + normSq (actC (Rmat p) Z 1) + normSq (actC (Rmat p) Z 2)
        + normSq (actC (Rmat p) Z 3)
      = (p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2)
          * (normSq (Z 0) + normSq (Z 1) + normSq (Z 2) + normSq (Z 3)) := by
  simp [actC, Rmat, normSq_apply]; ring

/-! ## Signature (3,3) of `2 Re κ` on `sl(2,ℂ) = su(2) ⊕ i·su(2)` -/

/-- Rotations (anti-Hermitian, traceless): `2 Re κ(A,A) = -16 |y|² ≤ 0`. -/
theorem killingR_rotation (y1 y2 y3 : ℝ) :
    2 * (LinearMap.trace ℂ M2 (adM (gen 0 y1 y2 y3) ∘ₗ adM (gen 0 y1 y2 y3))).re
      = -16 * (y1 ^ 2 + y2 ^ 2 + y3 ^ 2) := by
  have htr : (gen 0 y1 y2 y3).trace = 0 := by
    simp [gen, e1, e2, e3, σx, σy, σz, trace_fin_two]
  rw [killing_gl2, trace_sq_fin_two, htr, det_gen]
  simp [← Complex.ofReal_pow]
  ring

/-- Boosts (Hermitian, traceless): `2 Re κ(X,X) = +16 |x|² ≥ 0`. -/
theorem killingR_boost (x y z : ℝ) :
    2 * (LinearMap.trace ℂ M2 (adM (herm 0 x y z) ∘ₗ adM (herm 0 x y z))).re
      = 16 * (x ^ 2 + y ^ 2 + z ^ 2) := by
  have htr : (herm 0 x y z).trace = 0 := by
    rw [herm_eq]; simp [trace_fin_two]
  rw [killing_gl2, trace_sq_fin_two, htr, det_herm]
  simp [← Complex.ofReal_pow]
  ring

/-- Rotations and boosts are orthogonal for `2 Re κ`. -/
theorem killingR_orthogonal (y1 y2 y3 x y z : ℝ) :
    2 * (LinearMap.trace ℂ M2 (adM (gen 0 y1 y2 y3) ∘ₗ adM (herm 0 x y z))).re = 0 := by
  have htr : (herm 0 x y z).trace = 0 := by
    rw [herm_eq]; simp [trace_fin_two]
  rw [killing_gl2, htr, gen_eq_I_smul_herm, herm_eq, herm_eq]
  simp [trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two]

/-! ## Uniqueness of the invariant form on su(2) ≅ (ℝ³, ×) -/

/-- Any bilinear form on `ℝ³` that is invariant under the cross-product adjoint action,
`B(x×y, z) + B(y, x×z) = 0`, is a multiple of the dot product. -/
theorem invariant_form_unique (B : Matrix (Fin 3) (Fin 3) ℝ)
    (h : ∀ x y z : Fin 3 → ℝ,
      (crossProduct x y) ⬝ᵥ (B *ᵥ z) + y ⬝ᵥ (B *ᵥ (crossProduct x z)) = 0) :
    B = B 0 0 • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  have a1 := h (Pi.single 0 1) (Pi.single 1 1) (Pi.single 2 1)
  have a2 := h (Pi.single 1 1) (Pi.single 2 1) (Pi.single 0 1)
  have b1 := h (Pi.single 0 1) (Pi.single 0 1) (Pi.single 1 1)
  have b2 := h (Pi.single 0 1) (Pi.single 0 1) (Pi.single 2 1)
  have b3 := h (Pi.single 1 1) (Pi.single 1 1) (Pi.single 0 1)
  have b4 := h (Pi.single 1 1) (Pi.single 1 1) (Pi.single 2 1)
  have b5 := h (Pi.single 2 1) (Pi.single 2 1) (Pi.single 0 1)
  have b6 := h (Pi.single 2 1) (Pi.single 2 1) (Pi.single 1 1)
  simp [cross_apply, dotProduct, Matrix.mulVec, Fin.sum_univ_three, Pi.single_apply]
    at a1 a2 b1 b2 b3 b4 b5 b6
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> linarith

/-! ## Memory kernel: exponentials are eigenfunctions of the fractional integral -/

/-- `I^α e^{sτ} = s^{-α} e^{sτ}` for the Liouville integral (lower limit `-∞`), `α, s > 0`. -/
theorem frac_integral_exp (α s τ : ℝ) (hα : 0 < α) (hs : 0 < s) :
    (1 / Real.Gamma α) * ∫ u in Set.Ioi (0 : ℝ), u ^ (α - 1) * Real.exp (s * (τ - u))
      = s ^ (-α) * Real.exp (s * τ) := by
  have key := Real.integral_rpow_mul_exp_neg_mul_Ioi hα hs
  have hfun : (fun u : ℝ => u ^ (α - 1) * Real.exp (s * (τ - u)))
      = fun u => Real.exp (s * τ) * (u ^ (α - 1) * Real.exp (-(s * u))) := by
    funext u
    rw [show s * (τ - u) = s * τ + -(s * u) by ring, Real.exp_add]
    ring
  rw [hfun, MeasureTheory.integral_const_mul, key]
  have hΓ : Real.Gamma α ≠ 0 := (Real.Gamma_pos_of_pos hα).ne'
  simp only [one_div]
  rw [Real.rpow_neg hs.le, Real.inv_rpow hs.le]
  field_simp

end CalChecks
