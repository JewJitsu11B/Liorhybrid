/-
Copyright (c) 2026 Sam Leizerman. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Sam Leizerman
-/
module

public import Mathlib.Analysis.Quaternion
public import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Module
/-!

# Linear isometries of the quaternions

Every linear isometry of the real quaternions is a two-sided multiplication by unit quaternions.

## i. Overview

The real quaternions `ℍ` are a four-dimensional real inner product space with
`⟪x, y⟫ = Re (x * star y)`. Left and right multiplication by unit quaternions are linear
isometries, and so is quaternion conjugation `star`. This file proves the converse: every linear
isometry of `ℍ` is `x ↦ p * x * q` or `x ↦ p * star x * q` for unit quaternions `p` and `q`.

The proof uses the Cartan–Dieudonné theorem from Mathlib, by which every linear isometry is a
product of reflections, together with the observation that the reflection in the hyperplane
orthogonal to a unit quaternion `u` is `x ↦ -(u * star x * u)`.

## ii. Key results

- `Quaternion.mul_star_mul_self` : the identity `u * star x * u = 2⟪u, x⟫ u - ‖u‖² x`.
- `Quaternion.IsUnitTwoSided` : the predicate that a map is `x ↦ p * x * q` or
  `x ↦ p * star x * q` with `‖p‖ = ‖q‖ = 1`.
- `Quaternion.isUnitTwoSided_reflection` : reflections in hyperplanes are of this form.
- `Quaternion.isUnitTwoSided_linearIsometryEquiv` : every linear isometry is of this form.

## iii. Table of contents

- A. The two-sided product identity
- B. Maps given by two-sided multiplication
- C. Reflections
- D. Linear isometries

## iv. References

- J. H. Conway and D. A. Smith, *On Quaternions and Octonions*, A K Peters, 2003.

-/

@[expose] public section

namespace Quaternion

open scoped Quaternion

/-!

## A. The two-sided product identity

-/

/-- For quaternions `u` and `x`, `u * star x * u = 2⟪u, x⟫ u - ‖u‖² x`. For a unit `u` this says
that `x ↦ -(u * star x * u)` is the reflection in the hyperplane orthogonal to `u`. -/
lemma mul_star_mul_self (u x : ℍ) :
    u * star x * u = (2 * inner ℝ u x) • u - normSq u • x := by
  ext <;> simp [inner_def, normSq_def'] <;> ring

/-!

## B. Maps given by two-sided multiplication

-/

/-- A map `ℍ → ℍ` of the form `x ↦ p * x * q` or `x ↦ p * star x * q` for unit quaternions
`p` and `q`. -/
def IsUnitTwoSided (f : ℍ → ℍ) : Prop :=
  ∃ p q : ℍ, ‖p‖ = 1 ∧ ‖q‖ = 1 ∧ ((∀ x, f x = p * x * q) ∨ (∀ x, f x = p * star x * q))

/-- The identity map is a two-sided multiplication, with `p = q = 1`. -/
lemma isUnitTwoSided_id : IsUnitTwoSided id :=
  ⟨1, 1, by simp, by simp, Or.inl fun x => by simp⟩

/-- Two-sided multiplications are closed under composition. Composing two maps that both
involve `star` gives one that does not, since `star` reverses products. -/
lemma IsUnitTwoSided.comp {f g : ℍ → ℍ} (hf : IsUnitTwoSided f) (hg : IsUnitTwoSided g) :
    IsUnitTwoSided (f ∘ g) := by
  obtain ⟨p₁, q₁, hp₁, hq₁, h₁⟩ := hf
  obtain ⟨p₂, q₂, hp₂, hq₂, h₂⟩ := hg
  rcases h₁ with h₁ | h₁ <;> rcases h₂ with h₂ | h₂
  · exact ⟨p₁ * p₂, q₂ * q₁, by simp [hp₁, hp₂], by simp [hq₁, hq₂],
      Or.inl fun x => by simp [h₁, h₂, mul_assoc]⟩
  · exact ⟨p₁ * p₂, q₂ * q₁, by simp [hp₁, hp₂], by simp [hq₁, hq₂],
      Or.inr fun x => by simp [h₁, h₂, mul_assoc]⟩
  · exact ⟨p₁ * star q₂, star p₂ * q₁, by simp [hp₁, hq₂], by simp [hq₁, hp₂],
      Or.inr fun x => by simp [h₁, h₂, mul_assoc]⟩
  · exact ⟨p₁ * star q₂, star p₂ * q₁, by simp [hp₁, hq₂], by simp [hq₁, hp₂],
      Or.inl fun x => by simp [h₁, h₂, mul_assoc]⟩

/-!

## C. Reflections

-/

/-- For `v ≠ 0`, the reflection in the hyperplane orthogonal to `v` is
`x ↦ -(u * star x * u)` where `u = ‖v‖⁻¹ • v` is the unit quaternion along `v`. -/
lemma reflection_orthogonal_span_singleton {v : ℍ} (hv : v ≠ 0) (x : ℍ) :
    (ℝ ∙ v)ᗮ.reflection x = -(‖v‖⁻¹ • v) * star x * (‖v‖⁻¹ • v) := by
  have hn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  have hsq : normSq v = ‖v‖ ^ 2 := by rw [normSq_eq_norm_mul_self]; ring
  have key : -(‖v‖⁻¹ • v) * star x * (‖v‖⁻¹ • v) = -((‖v‖⁻¹) ^ 2 • (v * star x * v)) := by
    simp only [neg_mul, smul_mul_assoc, mul_smul_comm, smul_neg, smul_smul, pow_two]
  rw [Submodule.reflection_orthogonal_apply, Submodule.reflection_singleton_apply, key,
    mul_star_mul_self, hsq]
  simp only [RCLike.ofReal_real_eq_id, id_eq]
  match_scalars <;> field_simp

/-- The reflection in the hyperplane orthogonal to any quaternion is a two-sided
multiplication. For `v = 0` the reflection is the identity. -/
lemma isUnitTwoSided_reflection (v : ℍ) : IsUnitTwoSided ((ℝ ∙ v)ᗮ.reflection) := by
  by_cases hv : v = 0
  · subst hv
    refine ⟨1, 1, by simp, by simp, Or.inl fun x => ?_⟩
    rw [Submodule.reflection_orthogonal_apply, Submodule.reflection_singleton_apply]
    simp
  · have hn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
    refine ⟨-(‖v‖⁻¹ • v), ‖v‖⁻¹ • v, ?_, ?_,
      Or.inr (reflection_orthogonal_span_singleton hv)⟩
    · rw [norm_neg, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]
    · rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]

/-!

## D. Linear isometries

-/

/-- Every linear isometry of the quaternions is `x ↦ p * x * q` or `x ↦ p * star x * q`
for unit quaternions `p` and `q`. -/
lemma isUnitTwoSided_linearIsometryEquiv (φ : ℍ ≃ₗᵢ[ℝ] ℍ) : IsUnitTwoSided φ := by
  obtain ⟨l, -, rfl⟩ := φ.reflections_generate_dim
  induction l with
  | nil => exact isUnitTwoSided_id
  | cons v l ih =>
    rw [List.map_cons, List.prod_cons]
    exact (isUnitTwoSided_reflection v).comp ih

end Quaternion
