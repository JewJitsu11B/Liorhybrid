/-
Agent L. Independent Lean 4 / Mathlib checks of finite and algebraic statements in the
"alpha = 2" paper (reference/paper_snapshot.tex). Each theorem is labelled with the paper's
name for the claim it supports. No sorry, no new axioms.
-/
import Mathlib

open scoped Nat

namespace CalChecks.AlphaTwo

/-! ## Thm lorentz -/

/-- Thm lorentz, core lemma. If `sin (a θ) = b sin θ` on the open interval `(0, π/2)` and
`a ≠ 0`, then `a = 1` or `a = -1`. -/
theorem sin_mul_eq_const_mul_sin_on (a b : ℝ) (ha : a ≠ 0)
    (h : ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), Real.sin (a * θ) = b * Real.sin θ) :
    a = 1 ∨ a = -1 := by
  have hpi := Real.pi_pos
  set x : ℝ := Real.pi / 4 with hx
  have hxmem : x ∈ Set.Ioo (0 : ℝ) (Real.pi / 2) := ⟨by positivity, by rw [hx]; linarith⟩
  have hsinx : 0 < Real.sin x := Real.sin_pos_of_pos_of_lt_pi hxmem.1 (by rw [hx]; linarith)
  have hf1 : ∀ θ, HasDerivAt (fun θ => Real.sin (a * θ)) (a * Real.cos (a * θ)) θ := by
    intro θ
    have h1 : HasDerivAt (fun y : ℝ => a * y) a θ := by
      simpa using (hasDerivAt_id θ).const_mul a
    have := h1.sin
    convert this using 1
    ring
  have hf2 : ∀ θ, HasDerivAt (fun θ => a * Real.cos (a * θ)) (-(a * a) * Real.sin (a * θ)) θ := by
    intro θ
    have h1 : HasDerivAt (fun y : ℝ => a * y) a θ := by
      simpa using (hasDerivAt_id θ).const_mul a
    have := (h1.cos).const_mul a
    convert this using 1
    ring
  have hg1 : ∀ θ, HasDerivAt (fun θ => b * Real.sin θ) (b * Real.cos θ) θ :=
    fun θ => (Real.hasDerivAt_sin θ).const_mul b
  have hg2 : ∀ θ, HasDerivAt (fun θ => b * Real.cos θ) (b * -Real.sin θ) θ :=
    fun θ => (Real.hasDerivAt_cos θ).const_mul b
  have hev : (fun θ => Real.sin (a * θ)) =ᶠ[nhds x] (fun θ => b * Real.sin θ) :=
    Filter.eventually_of_mem (isOpen_Ioo.mem_nhds hxmem) h
  have hd1 := hev.deriv
  have e1 : deriv (fun θ => Real.sin (a * θ)) = fun θ => a * Real.cos (a * θ) :=
    funext fun θ => (hf1 θ).deriv
  have e2 : deriv (fun θ => b * Real.sin θ) = fun θ => b * Real.cos θ :=
    funext fun θ => (hg1 θ).deriv
  rw [e1, e2] at hd1
  have h1x : a * Real.cos (a * x) = b * Real.cos x := hd1.eq_of_nhds
  have h2x : -(a * a) * Real.sin (a * x) = b * -Real.sin x := by
    have := hd1.deriv_eq
    rwa [(hf2 x).deriv, (hg2 x).deriv] at this
  have h0x := h x hxmem
  rw [h0x] at h2x
  have hfac : b * Real.sin x * (a * a - 1) = 0 := by linarith
  rcases mul_eq_zero.1 hfac with hb | hb
  · -- b = 0 is impossible
    exfalso
    have hb0 : b = 0 := by
      rcases mul_eq_zero.1 hb with hb | hb
      · exact hb
      · exact absurd hb hsinx.ne'
    rw [hb0, zero_mul] at h0x h1x
    have hc : Real.cos (a * x) = 0 := by
      rcases mul_eq_zero.1 h1x with h' | h'
      · exact absurd h' ha
      · exact h'
    have := Real.sin_sq_add_cos_sq (a * x)
    rw [h0x, hc] at this
    norm_num at this
  · have : (a - 1) * (a + 1) = 0 := by linarith
    rcases mul_eq_zero.1 this with h' | h'
    · left; linarith
    · right; linarith

/-- Thm lorentz, target 1 as stated in the brief. -/
theorem sin_mul_eq_mul_sin_iff (a : ℝ) (ha : a ≠ 0) :
    (∀ θ : ℝ, Real.sin (a * θ) = a * Real.sin θ) ↔ a = 1 ∨ a = -1 := by
  constructor
  · intro h
    exact sin_mul_eq_const_mul_sin_on a a ha (fun θ _ => h θ)
  · rintro (rfl | rfl) θ
    · simp
    · simp

/-- The paper's angular factor `h_α(θ)` (Prop symbol), including its `α = 1` value. -/
noncomputable def hfun (α θ : ℝ) : ℝ :=
  if α = 1 then θ / Real.sin θ else Real.sin ((α - 1) * θ) / ((α - 1) * Real.sin θ)

/-- Thm lorentz: `h_2 = 1` wherever `sin θ ≠ 0`. -/
theorem hfun_two (θ : ℝ) (hθ : Real.sin θ ≠ 0) : hfun 2 θ = 1 := by
  unfold hfun
  rw [if_neg (by norm_num)]
  have : (2 : ℝ) - 1 = 1 := by norm_num
  rw [this, one_mul, one_mul, div_self hθ]

/-- Thm lorentz: the symmetry `h_{2-α} = h_α`. -/
theorem hfun_symm (α θ : ℝ) : hfun (2 - α) θ = hfun α θ := by
  unfold hfun
  by_cases h : α = 1
  · subst h; norm_num
  · have h' : ¬ (2 - α = 1) := fun h2 => h (by linarith)
    rw [if_neg h, if_neg h']
    have e : (2 - α - 1) = -(α - 1) := by ring
    rw [e, neg_mul, Real.sin_neg, neg_mul, neg_div_neg_eq]

/-- Thm lorentz at `α = 1`: `θ / sin θ` is not constant on `(0, π/2)`. -/
theorem hfun_one_not_const :
    ¬ ∃ C : ℝ, ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), θ / Real.sin θ = C := by
  rintro ⟨C, hC⟩
  have hpi := Real.pi_pos
  have hsin : ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), 0 < Real.sin θ := fun θ hθ =>
    Real.sin_pos_of_pos_of_lt_pi hθ.1 (by linarith [hθ.2])
  have h : ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), θ = C * Real.sin θ := fun θ hθ =>
    (div_eq_iff (hsin θ hθ).ne').1 (hC θ hθ)
  set x : ℝ := Real.pi / 4 with hx
  have hxmem : x ∈ Set.Ioo (0 : ℝ) (Real.pi / 2) := ⟨by positivity, by rw [hx]; linarith⟩
  have hg1 : ∀ θ, HasDerivAt (fun θ => C * Real.sin θ) (C * Real.cos θ) θ :=
    fun θ => (Real.hasDerivAt_sin θ).const_mul C
  have hg2 : ∀ θ, HasDerivAt (fun θ => C * Real.cos θ) (C * -Real.sin θ) θ :=
    fun θ => (Real.hasDerivAt_cos θ).const_mul C
  have hev : (fun θ : ℝ => θ) =ᶠ[nhds x] (fun θ => C * Real.sin θ) :=
    Filter.eventually_of_mem (isOpen_Ioo.mem_nhds hxmem) h
  have hd1 := hev.deriv
  have e1 : deriv (fun θ : ℝ => θ) = fun _ => (1 : ℝ) :=
    funext fun θ => (hasDerivAt_id' θ).deriv
  have e2 : deriv (fun θ => C * Real.sin θ) = fun θ => C * Real.cos θ :=
    funext fun θ => (hg1 θ).deriv
  rw [e1, e2] at hd1
  have h1x : (1 : ℝ) = C * Real.cos x := hd1.eq_of_nhds
  have h2x : (0 : ℝ) = C * -Real.sin x := by
    have := hd1.deriv_eq
    rwa [(hasDerivAt_const x (1 : ℝ)).deriv, (hg2 x).deriv] at this
  have hC0 : C = 0 := by
    rcases mul_eq_zero.1 h2x.symm with h' | h'
    · exact h'
    · exact absurd (neg_eq_zero.1 h') (hsin x hxmem).ne'
  rw [hC0, zero_mul] at h1x
  exact one_ne_zero h1x

/-- Thm lorentz (given Prop symbol): for `α > 0`, the angular factor `h_α` is constant on
`(0, π/2)` if and only if `α = 2`. -/
theorem lorentz_selects_two (α : ℝ) (hα : 0 < α) :
    (∃ C : ℝ, ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), hfun α θ = C) ↔ α = 2 := by
  have hpi := Real.pi_pos
  have hsin : ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), 0 < Real.sin θ := fun θ hθ =>
    Real.sin_pos_of_pos_of_lt_pi hθ.1 (by linarith [hθ.2])
  constructor
  · rintro ⟨C, hC⟩
    by_cases h1 : α = 1
    · exfalso
      apply hfun_one_not_const
      refine ⟨C, fun θ hθ => ?_⟩
      have := hC θ hθ
      unfold hfun at this
      rwa [if_pos h1] at this
    · have ha : α - 1 ≠ 0 := sub_ne_zero.2 h1
      have key : ∀ θ ∈ Set.Ioo (0 : ℝ) (Real.pi / 2),
          Real.sin ((α - 1) * θ) = (C * (α - 1)) * Real.sin θ := by
        intro θ hθ
        have := hC θ hθ
        unfold hfun at this
        rw [if_neg h1, div_eq_iff (mul_ne_zero ha (hsin θ hθ).ne')] at this
        rw [this]; ring
      rcases sin_mul_eq_const_mul_sin_on (α - 1) (C * (α - 1)) ha key with h | h
      · linarith
      · exfalso; linarith
  · rintro rfl
    exact ⟨1, fun θ hθ => hfun_two θ (hsin θ hθ).ne'⟩

/-- Thm lorentz: the two Taylor coefficients combine to the stated `α(2-α)/6`. -/
theorem lorentz_defect_coeff (α : ℝ) : (1 - (α - 1) ^ 2) / 6 = α * (2 - α) / 6 := by ring

/-! ## Thm KG -/

/-- Thm KG (3), real core: `2 + τ⁻² / X = 0` iff `X = -1/(2 τ²)`; here `X = s² + c²k²`. -/
theorem kg_symbol_zero (τ X : ℝ) (hτ : τ ≠ 0) (hX : X ≠ 0) :
    2 + (τ ^ 2)⁻¹ / X = 0 ↔ X = -(1 / (2 * τ ^ 2)) := by
  have hτ2 : τ ^ 2 ≠ 0 := pow_ne_zero 2 hτ
  have key : 2 + (τ ^ 2)⁻¹ / X = (2 * τ ^ 2 * X + 1) / (τ ^ 2 * X) := by
    field_simp
  have key2 : X = -(1 / (2 * τ ^ 2)) ↔ 2 * τ ^ 2 * X + 1 = 0 := by
    constructor
    · intro h; rw [h]; field_simp; ring
    · intro h
      have : X = -1 / (2 * τ ^ 2) := by
        rw [eq_div_iff (by positivity)]; linarith
      rw [this]; ring
  rw [key, key2, div_eq_zero_iff]
  constructor
  · rintro (h | h)
    · exact h
    · exact absurd h (mul_ne_zero hτ2 hX)
  · exact Or.inl

/-- Remark phase / Thm KG: with the opposite sign on the memory, `X = +1/(2 τ²)`. -/
theorem kg_symbol_zero_flipped (τ X : ℝ) (hτ : τ ≠ 0) (hX : X ≠ 0) :
    2 - (τ ^ 2)⁻¹ / X = 0 ↔ X = 1 / (2 * τ ^ 2) := by
  have h := kg_symbol_zero τ (-X) hτ (neg_ne_zero.2 hX)
  have e : 2 + (τ ^ 2)⁻¹ / (-X) = 2 - (τ ^ 2)⁻¹ / X := by rw [div_neg]; ring
  rw [e] at h
  rw [h]
  constructor <;> intro h' <;> linarith

/-- Thm KG: `μ² = 1/(2 τ_c²)` with `μ, τ_c > 0` is the same as `μ τ_c = 1/√2`. -/
theorem mu_tau (μ τ : ℝ) (hμ : 0 < μ) (hτ : 0 < τ) :
    μ ^ 2 = 1 / (2 * τ ^ 2) ↔ μ * τ = 1 / Real.sqrt 2 := by
  have hs : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hsq : (1 / Real.sqrt 2) ^ 2 = 1 / 2 := by rw [div_pow, hs2]; norm_num
  have e : μ ^ 2 = 1 / (2 * τ ^ 2) ↔ (μ * τ) ^ 2 = 1 / 2 := by
    rw [eq_div_iff (by positivity), mul_pow]
    constructor <;> intro h <;> linarith
  rw [e, ← hsq]
  constructor
  · intro h
    have hpos : 0 < μ * τ := mul_pos hμ hτ
    have hpos2 : 0 < 1 / Real.sqrt 2 := by positivity
    nlinarith [sq_nonneg (μ * τ - 1 / Real.sqrt 2), sq_nonneg (μ * τ + 1 / Real.sqrt 2)]
  · intro h; rw [h]

/-- Thm KG (3): on the carrier `s = -iω` (ω real, `ck` standing for `c|k|`), the symbol of
`T = 2J + K_2 J` vanishes iff `ω² = c²k² + 1/(2 τ_c²)`. -/
theorem kg_dispersion (τ ω ck : ℝ) (hτ : τ ≠ 0) (hne : ω ^ 2 ≠ ck ^ 2) :
    (2 : ℂ) + ((τ : ℂ) ^ 2)⁻¹ / ((-Complex.I * ω) ^ 2 + (ck : ℂ) ^ 2) = 0
      ↔ ω ^ 2 = ck ^ 2 + 1 / (2 * τ ^ 2) := by
  have hX : ck ^ 2 - ω ^ 2 ≠ 0 := fun h => hne (by linarith)
  have e : (-Complex.I * ω) ^ 2 + (ck : ℂ) ^ 2 = ((ck ^ 2 - ω ^ 2 : ℝ) : ℂ) := by
    push_cast; rw [mul_pow, neg_sq, Complex.I_sq]; ring
  have e2 : (2 : ℂ) + ((τ : ℂ) ^ 2)⁻¹ / ((ck ^ 2 - ω ^ 2 : ℝ) : ℂ)
      = ((2 + (τ ^ 2)⁻¹ / (ck ^ 2 - ω ^ 2) : ℝ) : ℂ) := by push_cast; ring
  rw [e, e2, Complex.ofReal_eq_zero, kg_symbol_zero τ _ hτ hX]
  constructor <;> intro h <;> linarith

/-- Remark phase: with a factor `-1` on the memory the root is `ω² = c²k² - 1/(2 τ_c²)`. -/
theorem kg_dispersion_flipped (τ ω ck : ℝ) (hτ : τ ≠ 0) (hne : ω ^ 2 ≠ ck ^ 2) :
    (2 : ℂ) - ((τ : ℂ) ^ 2)⁻¹ / ((-Complex.I * ω) ^ 2 + (ck : ℂ) ^ 2) = 0
      ↔ ω ^ 2 = ck ^ 2 - 1 / (2 * τ ^ 2) := by
  have hX : ck ^ 2 - ω ^ 2 ≠ 0 := fun h => hne (by linarith)
  have e : (-Complex.I * ω) ^ 2 + (ck : ℂ) ^ 2 = ((ck ^ 2 - ω ^ 2 : ℝ) : ℂ) := by
    push_cast; rw [mul_pow, neg_sq, Complex.I_sq]; ring
  have e2 : (2 : ℂ) - ((τ : ℂ) ^ 2)⁻¹ / ((ck ^ 2 - ω ^ 2 : ℝ) : ℂ)
      = ((2 - (τ ^ 2)⁻¹ / (ck ^ 2 - ω ^ 2) : ℝ) : ℂ) := by push_cast; ring
  rw [e, e2, Complex.ofReal_eq_zero, kg_symbol_zero_flipped τ _ hτ hX]
  constructor <;> intro h <;> linarith

/-- Thm KG (2), algebra of the proof: `(1/2)(□S) + (1/2)μ²S - (1/2)μ²S = (1/2)□S`
once `□(G_μ*S) = S - μ² G_μ*S`. Here `bS, bG` stand for `□S, □(G_μ*S)`. -/
theorem kg_part2_algebra (μ S GS bS bG : ℝ) (h : bG = S - μ ^ 2 * GS) :
    (1 / 2 * bS - 1 / 2 * μ ^ 2 * bG) + μ ^ 2 * (1 / 2 * S - 1 / 2 * μ ^ 2 * GS)
      = 1 / 2 * bS := by
  rw [h]; ring

/-! ## Prop nulllegs -/

/-- Prop nulllegs: with `∂_t = c(∂_u + ∂_v)`, `∂_x = ∂_v - ∂_u` the wave symbol is
`4c² ∂_u ∂_v`; and `u v = c²t² - x²`. -/
theorem nulllegs_symbol (c a b : ℝ) :
    (c * (a + b)) ^ 2 - c ^ 2 * (b - a) ^ 2 = 4 * c ^ 2 * (a * b) := by ring

theorem nulllegs_uv (c t x : ℝ) : (c * t - x) * (c * t + x) = c ^ 2 * t ^ 2 - x ^ 2 := by ring

/-! ## Prop checker -/

/-- Prop checker / Remark checker: `(-μ²)^n = (iμ)^(2n)`. -/
theorem neg_mu_sq_pow (μ : ℂ) (n : ℕ) : (-μ ^ 2) ^ n = (Complex.I * μ) ^ (2 * n) := by
  rw [pow_mul, mul_pow, Complex.I_sq]; ring

/-- Prop checker (2): `(iμ)^(2n+1) / (2 i c μ) = (-μ²)^n / (2c)`. -/
theorem checker_factor (μ c : ℂ) (hμ : μ ≠ 0) (hc : c ≠ 0) (n : ℕ) :
    (Complex.I * μ) ^ (2 * n + 1) / (2 * Complex.I * c * μ) = (-μ ^ 2) ^ n / (2 * c) := by
  rw [pow_succ, ← neg_mu_sq_pow]
  have hI := Complex.I_ne_zero
  field_simp

/-- Prop checker (1), induction step: the claimed term
`(1/2c) (uv/4c²)^n / (n!)²` is mapped to the next one by
`(1/4c²) ∫_0^u ∫_0^v` (the form of `G*` in Prop nulllegs). -/
theorem checker_step (c u v : ℝ) (hc : c ≠ 0) (n : ℕ) :
    (1 / (4 * c ^ 2)) * ∫ u' in (0 : ℝ)..u, ∫ v' in (0 : ℝ)..v,
        (1 / (2 * c)) * (u' * v' / (4 * c ^ 2)) ^ n / ((n ! : ℝ)) ^ 2
      = (1 / (2 * c)) * (u * v / (4 * c ^ 2)) ^ (n + 1) / (((n + 1) ! : ℝ)) ^ 2 := by
  have hf : (n ! : ℝ) ≠ 0 := by positivity
  have hn : ((n : ℝ) + 1) ≠ 0 := by positivity
  have inner : ∀ u' : ℝ, (∫ v' in (0 : ℝ)..v,
      (1 / (2 * c)) * (u' * v' / (4 * c ^ 2)) ^ n / ((n ! : ℝ)) ^ 2)
      = ((1 / (2 * c)) / (4 * c ^ 2) ^ n / ((n ! : ℝ)) ^ 2 * (v ^ (n + 1) / (n + 1))) * u' ^ n := by
    intro u'
    have : (fun v' : ℝ => (1 / (2 * c)) * (u' * v' / (4 * c ^ 2)) ^ n / ((n ! : ℝ)) ^ 2)
        = fun v' => ((1 / (2 * c)) * u' ^ n / (4 * c ^ 2) ^ n / ((n ! : ℝ)) ^ 2) * v' ^ n := by
      funext v'; rw [div_pow, mul_pow]; ring
    rw [this, intervalIntegral.integral_const_mul, integral_pow,
      zero_pow (Nat.succ_ne_zero n), sub_zero]
    ring
  simp_rw [inner]
  rw [intervalIntegral.integral_const_mul, integral_pow, zero_pow (Nat.succ_ne_zero n), sub_zero,
    Nat.factorial_succ]
  have hd : (4 * c ^ 2) ≠ 0 := by positivity
  generalize 4 * c ^ 2 = d at hd ⊢
  push_cast
  rw [div_pow, mul_pow, pow_succ d]
  field_simp

/-- Prop checker (3), domination: `ε^n C(N-1, n) ≤ (N ε)^n / n!` for `ε ≥ 0`. -/
theorem checker_binomial_bound (N n : ℕ) (ε : ℝ) (hε : 0 ≤ ε) :
    ε ^ n * ((N - 1).choose n : ℝ) ≤ (N * ε) ^ n / (n ! : ℝ) := by
  have h1 : ((N - 1).choose n : ℝ) ≤ ((N - 1 : ℕ) : ℝ) ^ n / (n ! : ℝ) :=
    Nat.choose_le_pow_div n (N - 1)
  have h2 : ((N - 1 : ℕ) : ℝ) ^ n ≤ (N : ℝ) ^ n := by
    apply pow_le_pow_left₀ (Nat.cast_nonneg _)
    exact_mod_cast Nat.sub_le N 1
  have hf : (0 : ℝ) < (n ! : ℝ) := by positivity
  calc ε ^ n * ((N - 1).choose n : ℝ) ≤ ε ^ n * ((N : ℝ) ^ n / (n ! : ℝ)) := by
        apply mul_le_mul_of_nonneg_left _ (pow_nonneg hε n)
        exact h1.trans (div_le_div_of_nonneg_right h2 hf.le)
    _ = (N * ε) ^ n / (n ! : ℝ) := by rw [mul_pow]; ring

/-! ## Thm pathsum -/

/-- Thm pathsum (1), induction step of the proof: the order-two Riemann-Liouville integral
`∫_0^T τ (T-τ)^m dτ = T^(m+2) / ((m+1)(m+2))`. -/
theorem rl_two_step (T : ℝ) (m : ℕ) :
    ∫ τ in (0 : ℝ)..T, τ * (T - τ) ^ m = T ^ (m + 2) / (((m : ℝ) + 1) * ((m : ℝ) + 2)) := by
  have h1 : ((m : ℝ) + 1) ≠ 0 := by positivity
  have h2 : ((m : ℝ) + 2) ≠ 0 := by positivity
  have hderiv : ∀ τ ∈ Set.uIcc (0 : ℝ) T,
      HasDerivAt (fun τ : ℝ => -(τ * (T - τ) ^ (m + 1)) / ((m : ℝ) + 1)
          - (T - τ) ^ (m + 2) / (((m : ℝ) + 1) * ((m : ℝ) + 2)))
        (τ * (T - τ) ^ m) τ := by
    intro τ _
    have hsub : HasDerivAt (fun τ : ℝ => T - τ) (-1) τ := by
      simpa using (hasDerivAt_id τ).const_sub T
    have hp1 := hsub.pow (m + 1)
    have hp2 := hsub.pow (m + 2)
    have hid : HasDerivAt (fun τ : ℝ => τ) 1 τ := hasDerivAt_id' τ
    have := (((hid.mul hp1).neg).div_const ((m : ℝ) + 1)).sub
      (hp2.div_const (((m : ℝ) + 1) * ((m : ℝ) + 2)))
    convert this using 1
    simp only [Pi.pow_apply, Nat.add_sub_cancel, Nat.cast_add, Nat.cast_one, Nat.cast_ofNat]
    have e : m + 2 - 1 = m + 1 := rfl
    rw [e]
    field_simp
    ring
  have hint : IntervalIntegrable (fun τ : ℝ => τ * (T - τ) ^ m) MeasureTheory.volume 0 T :=
    (by fun_prop : Continuous (fun τ : ℝ => τ * (T - τ) ^ m)).intervalIntegrable _ _
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint]
  simp

/-- Thm pathsum (1): the bound `(t-t0)^(2n)/(2n)!` is reproduced by one more order-two
integration, `∫_0^T τ (T-τ)^(2n)/(2n)! dτ = T^(2n+2)/(2n+2)!`. -/
theorem rl_two_factorial (T : ℝ) (n : ℕ) :
    ∫ τ in (0 : ℝ)..T, τ * ((T - τ) ^ (2 * n) / ((2 * n) ! : ℝ))
      = T ^ (2 * (n + 1)) / ((2 * (n + 1)) ! : ℝ) := by
  have hf : ((2 * n) ! : ℝ) ≠ 0 := by positivity
  have e : (fun τ : ℝ => τ * ((T - τ) ^ (2 * n) / ((2 * n) ! : ℝ)))
      = fun τ => (1 / ((2 * n) ! : ℝ)) * (τ * (T - τ) ^ (2 * n)) := by
    funext τ; ring
  rw [e, intervalIntegral.integral_const_mul, rl_two_step]
  have e2 : 2 * (n + 1) = (2 * n + 1) + 1 := by ring
  rw [e2, Nat.factorial_succ, Nat.factorial_succ]
  push_cast
  field_simp
  ring

/-- Thm pathsum (1), the summation: if `|a n| ≤ T^(2n)/(2n)! * M` (the paper's bound on
`G^{*n} * S`), then `(1/2) Σ (-μ²)^n a n` converges absolutely and is bounded by
`(1/2) cosh(μ T) M`. -/
theorem pathsum_majorant (μ T M : ℝ) (a : ℕ → ℝ)
    (h : ∀ n, |a n| ≤ T ^ (2 * n) / ((2 * n) ! : ℝ) * M) :
    Summable (fun n => |1 / 2 * (-μ ^ 2) ^ n * a n|) ∧
      |∑' n, 1 / 2 * (-μ ^ 2) ^ n * a n| ≤ 1 / 2 * Real.cosh (μ * T) * M := by
  have hg : HasSum (fun n => 1 / 2 * ((μ * T) ^ (2 * n) / ((2 * n) ! : ℝ)) * M)
      (1 / 2 * Real.cosh (μ * T) * M) :=
    ((Real.hasSum_cosh (μ * T)).mul_left (1 / 2)).mul_right M
  have hb : ∀ n, |1 / 2 * (-μ ^ 2) ^ n * a n|
      ≤ 1 / 2 * ((μ * T) ^ (2 * n) / ((2 * n) ! : ℝ)) * M := by
    intro n
    have e1 : |1 / 2 * (-μ ^ 2) ^ n * a n| = 1 / 2 * (μ ^ 2) ^ n * |a n| := by
      rw [abs_mul, abs_mul, abs_pow, abs_neg, abs_of_nonneg (sq_nonneg μ)]
      norm_num
    rw [e1]
    calc 1 / 2 * (μ ^ 2) ^ n * |a n|
        ≤ 1 / 2 * (μ ^ 2) ^ n * (T ^ (2 * n) / ((2 * n) ! : ℝ) * M) :=
          mul_le_mul_of_nonneg_left (h n) (by positivity)
      _ = 1 / 2 * ((μ * T) ^ (2 * n) / ((2 * n) ! : ℝ)) * M := by
          rw [mul_pow, pow_mul]; ring
  refine ⟨Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hb hg.summable, ?_⟩
  have := tsum_of_norm_bounded hg (f := fun n => 1 / 2 * (-μ ^ 2) ^ n * a n)
    (fun n => by simpa [Real.norm_eq_abs] using hb n)
  simpa [Real.norm_eq_abs] using this

/-- Thm pathsum (3): the symbol series is geometric,
`Σ (-μ²)^n X^(-n-1) = 1/(X + μ²)` when `|μ²/X| < 1` (`X = s² + c²k²`). -/
theorem pathsum_symbol_geometric (μ X : ℝ) (hX : X ≠ 0) (hr : |μ ^ 2 / X| < 1) :
    HasSum (fun n : ℕ => (-μ ^ 2) ^ n * (X⁻¹) ^ (n + 1)) (1 / (X + μ ^ 2)) := by
  have hr' : |-(μ ^ 2 / X)| < 1 := by rwa [abs_neg]
  have hgeo := (hasSum_geometric_of_abs_lt_one hr').mul_left X⁻¹
  have hne : X + μ ^ 2 ≠ 0 := by
    intro h0
    have : μ ^ 2 / X = -1 := by
      rw [div_eq_iff hX]; linarith
    rw [this] at hr
    norm_num at hr
  have e1 : (fun n : ℕ => (-μ ^ 2) ^ n * (X⁻¹) ^ (n + 1))
      = fun n : ℕ => X⁻¹ * (-(μ ^ 2 / X)) ^ n := by
    funext n
    rw [div_eq_mul_inv, ← neg_mul, mul_pow, pow_succ]; ring
  have e2 : X⁻¹ * (1 - -(μ ^ 2 / X))⁻¹ = 1 / (X + μ ^ 2) := by
    have e3 : 1 + μ ^ 2 / X = (X + μ ^ 2) / X := by field_simp
    rw [sub_neg_eq_add, e3, inv_div]
    field_simp
  rw [e1, ← e2]
  exact hgeo

/-! ## Prop pair -/

/-- The mode propagator `M_k(t)` of Prop pair. -/
noncomputable def Mk (ω t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos (ω * t), Real.sin (ω * t) / ω; -ω * Real.sin (ω * t), Real.cos (ω * t)]

/-- Prop pair: the group law `M_k(t + t') = M_k(t) M_k(t')` for `ω ≠ 0`. -/
theorem Mk_add (ω t t' : ℝ) (hω : ω ≠ 0) : Mk ω (t + t') = Mk ω t * Mk ω t' := by
  unfold Mk
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two, mul_add, Real.cos_add, Real.sin_add] <;>
    field_simp <;> ring

/-- Prop pair: `det M_k(t) = 1`. -/
theorem Mk_det (ω t : ℝ) (hω : ω ≠ 0) : (Mk ω t).det = 1 := by
  unfold Mk
  rw [Matrix.det_fin_two_of]
  have := Real.sin_sq_add_cos_sq (ω * t)
  field_simp
  nlinarith

/-! ## Thm unitary -/

/-- Thm unitary (1): for real `ω, t` the multiplier `exp(-iωt)` has norm one. -/
theorem norm_phase (ω t : ℝ) : ‖Complex.exp (-(Complex.I * ω * t))‖ = 1 := by
  rw [Complex.norm_exp]
  simp

/-- Thm unitary (1): each Fourier mode keeps its modulus. -/
theorem norm_mode (ω t : ℝ) (a : ℂ) : ‖Complex.exp (-(Complex.I * ω * t)) * a‖ = ‖a‖ := by
  rw [norm_mul, norm_phase, one_mul]

/-- Thm unitary (2), carrier core: a unimodular scalar leaves `J Jᴴ` unchanged. -/
theorem closure_phase_invariant {n : Type*} [Fintype n] (z : ℂ) (hz : ‖z‖ = 1)
    (A : Matrix n n ℂ) : (z • A) * (z • A).conjTranspose = A * A.conjTranspose := by
  have hzz : z * star z = 1 := by
    rw [Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq, hz]; norm_num
  rw [Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul, smul_smul, hzz, one_smul]

/-- Thm unitary (2) for the 2x2 carrier `J(t) = exp(-iωt) J0`. -/
theorem closure_const (ω t : ℝ) (J0 : Matrix (Fin 2) (Fin 2) ℂ) :
    (Complex.exp (-(Complex.I * ω * t)) • J0)
        * (Complex.exp (-(Complex.I * ω * t)) • J0).conjTranspose
      = J0 * J0.conjTranspose :=
  closure_phase_invariant _ (norm_phase ω t) J0

/-- Thm unitary (3), the cross term: for `J = z̄ A + z B` with `|z| = 1`
(`z = exp(iωt)`), `J Jᴴ = A Aᴴ + B Bᴴ + z̄² A Bᴴ + z² B Aᴴ`; `z² = exp(2iωt)`. -/
theorem closure_two_modes {n : Type*} [Fintype n] (z : ℂ) (hz : ‖z‖ = 1)
    (A B : Matrix n n ℂ) :
    (star z • A + z • B) * (star z • A + z • B).conjTranspose
      = A * A.conjTranspose + B * B.conjTranspose
        + (star z * star z) • (A * B.conjTranspose) + (z * z) • (B * A.conjTranspose) := by
  have hzz : z * star z = 1 := by
    rw [Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq, hz]; norm_num
  have hzz' : star z * z = 1 := by rw [mul_comm]; exact hzz
  rw [Matrix.conjTranspose_add, Matrix.conjTranspose_smul, Matrix.conjTranspose_smul, star_star,
    Matrix.add_mul, Matrix.mul_add, Matrix.mul_add]
  simp only [Matrix.smul_mul, Matrix.mul_smul, smul_smul, hzz, hzz', one_smul]
  abel

/-- Thm unitary (3), smallest example: the mixed mode `exp(-iωt) + exp(iωt)` has modulus
`|2 cos(ωt)|`, which is not constant when `ω ≠ 0`. -/
theorem mixed_mode_norm (ω t : ℝ) :
    ‖Complex.exp (-(Complex.I * ω * t)) + Complex.exp (Complex.I * ω * t)‖
      = |2 * Real.cos (ω * t)| := by
  have key : Complex.exp (-(Complex.I * ω * t)) + Complex.exp (Complex.I * ω * t)
      = ((2 * Real.cos (ω * t) : ℝ) : ℂ) := by
    push_cast
    rw [Complex.cos]
    ring_nf
  rw [key, Complex.norm_real, Real.norm_eq_abs]

theorem mixed_mode_not_const (ω : ℝ) (hω : ω ≠ 0) :
    ¬ ∃ C : ℝ, ∀ t : ℝ,
      ‖Complex.exp (-(Complex.I * ω * t)) + Complex.exp (Complex.I * ω * t)‖ = C := by
  rintro ⟨C, hC⟩
  have h0 := hC 0
  have h1 := hC (Real.pi / (2 * ω))
  rw [mixed_mode_norm] at h0 h1
  have e : ω * (Real.pi / (2 * ω)) = Real.pi / 2 := by field_simp
  rw [e, Real.cos_pi_div_two] at h1
  rw [mul_zero, Real.cos_zero] at h0
  norm_num at h0 h1
  linarith

/-- Thm unitary (4), last line of the proof: `1 ≤ √(1+x) ≤ 1 + x/2` for `x ≥ 0`. -/
theorem charge_ratio_bounds (x : ℝ) (hx : 0 ≤ x) :
    1 ≤ Real.sqrt (1 + x) ∧ Real.sqrt (1 + x) ≤ 1 + x / 2 := by
  have hr0 : 0 ≤ Real.sqrt (1 + x) := Real.sqrt_nonneg _
  have hr2 : Real.sqrt (1 + x) ^ 2 = 1 + x := Real.sq_sqrt (by linarith)
  constructor
  · nlinarith
  · nlinarith [sq_nonneg (Real.sqrt (1 + x) - 1)]

/-! ## Thm NR and Cor feynman -/

/-- Thm NR, first line of the proof: `-x²/8 ≤ √(1+x) - 1 - x/2 ≤ 0` for `x ≥ 0`. -/
theorem sqrt_expansion_bounds (x : ℝ) (hx : 0 ≤ x) :
    -x ^ 2 / 8 ≤ Real.sqrt (1 + x) - 1 - x / 2 ∧ Real.sqrt (1 + x) - 1 - x / 2 ≤ 0 := by
  have hr2 : Real.sqrt (1 + x) ^ 2 = 1 + x := Real.sq_sqrt (by linarith)
  have hr1 : 1 ≤ Real.sqrt (1 + x) := (charge_ratio_bounds x hx).1
  set r := Real.sqrt (1 + x) with hr
  have hxr : x = r ^ 2 - 1 := by linarith
  constructor
  · rw [hxr]
    have : 0 ≤ (r - 1) ^ 2 * ((r + 1) ^ 2 - 4) :=
      mul_nonneg (sq_nonneg _) (by nlinarith)
    nlinarith
  · rw [hxr]; nlinarith [sq_nonneg (r - 1)]

/-- Thm NR (target 6): the dispersion error,
`0 ≤ p²/(2m) - (√(m²c⁴ + p²c²) - mc²) ≤ p⁴/(8 m³ c²)`. -/
theorem nr_dispersion_bounds (m c p : ℝ) (hm : 0 < m) (hc : 0 < c) :
    0 ≤ p ^ 2 / (2 * m) - (Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) - m * c ^ 2) ∧
      p ^ 2 / (2 * m) - (Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) - m * c ^ 2)
        ≤ p ^ 4 / (8 * m ^ 3 * c ^ 2) := by
  have hA : 0 < m * c ^ 2 := by positivity
  have hE2 : Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) ^ 2 = m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2 :=
    Real.sq_sqrt (by positivity)
  have hE0 : 0 ≤ Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) := Real.sqrt_nonneg _
  set E := Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) with hE
  set A := m * c ^ 2 with hAdef
  have hpc : p ^ 2 * c ^ 2 = E ^ 2 - A ^ 2 := by rw [hE2, hAdef]; ring
  have hEA : A ≤ E := by
    have : A ^ 2 ≤ E ^ 2 := by nlinarith [mul_nonneg (sq_nonneg p) (sq_nonneg c)]
    nlinarith
  have e1 : p ^ 2 / (2 * m) = (E ^ 2 - A ^ 2) / (2 * A) := by
    rw [← hpc, hAdef]; field_simp
  have e2 : p ^ 4 / (8 * m ^ 3 * c ^ 2) = (E ^ 2 - A ^ 2) ^ 2 / (8 * A ^ 3) := by
    rw [← hpc, hAdef]; field_simp
  have e3 : (E ^ 2 - A ^ 2) / (2 * A) - (E - A) = (E - A) ^ 2 / (2 * A) := by
    field_simp; ring
  rw [e1, e2, e3]
  constructor
  · positivity
  · rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have h4 : 0 ≤ (E - A) ^ 2 * ((E + A) ^ 2 - 4 * A ^ 2) :=
      mul_nonneg (sq_nonneg _) (by nlinarith)
    have h5 : 0 ≤ 2 * A ^ 3 * ((E - A) ^ 2 * ((E + A) ^ 2 - 4 * A ^ 2)) :=
      mul_nonneg (by positivity) h4
    nlinarith

/-- Thm NR, second line of the proof: `|e^{ia} - e^{ib}| ≤ |a - b|` for real `a, b`. -/
theorem norm_exp_sub_exp_le (a b : ℝ) :
    ‖Complex.exp (Complex.I * a) - Complex.exp (Complex.I * b)‖ ≤ |a - b| := by
  have e : Complex.exp (Complex.I * a) - Complex.exp (Complex.I * b)
      = Complex.exp (Complex.I * b) * (Complex.exp (Complex.I * ((a - b : ℝ) : ℂ)) - 1) := by
    rw [mul_sub, ← Complex.exp_add, mul_one]
    congr 2
    push_cast; ring
  rw [e, norm_mul, Complex.norm_exp_I_mul_ofReal, one_mul]
  exact Real.norm_exp_I_mul_ofReal_sub_one_le

/-- Thm NR, mode by mode: with `p = ħk`, the relativistic phase (rest phase removed) and the
Schrodinger phase differ by at most `|t| p⁴ / (8 m³ c² ħ) = |t| ħ³ k⁴ / (8 m³ c²)`. -/
theorem nr_mode_bound (m c p hbar t : ℝ) (hm : 0 < m) (hc : 0 < c) (hh : 0 < hbar) :
    ‖Complex.exp (Complex.I *
          ((-((Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) - m * c ^ 2) * t / hbar) : ℝ) : ℂ))
        - Complex.exp (Complex.I * ((-(p ^ 2 / (2 * m) * t / hbar) : ℝ) : ℂ))‖
      ≤ |t| * p ^ 4 / (8 * m ^ 3 * c ^ 2 * hbar) := by
  obtain ⟨h0, h1⟩ := nr_dispersion_bounds m c p hm hc
  set D := p ^ 2 / (2 * m) - (Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) - m * c ^ 2) with hD
  refine (norm_exp_sub_exp_le _ _).trans ?_
  have e : -((Real.sqrt (m ^ 2 * c ^ 4 + p ^ 2 * c ^ 2) - m * c ^ 2) * t / hbar)
      - -(p ^ 2 / (2 * m) * t / hbar) = D * t / hbar := by
    rw [hD]; ring
  rw [e, abs_div, abs_mul, abs_of_nonneg h0, abs_of_pos hh]
  have e2 : |t| * p ^ 4 / (8 * m ^ 3 * c ^ 2 * hbar)
      = p ^ 4 / (8 * m ^ 3 * c ^ 2) * |t| / hbar := by
    field_simp
  rw [e2]
  apply div_le_div_of_nonneg_right _ hh.le
  exact mul_le_mul_of_nonneg_right h1 (abs_nonneg t)

/-- Cor feynman, amplitude factor: `0 ≤ 1 - 1/√(1+x) ≤ x/2` for `x ≥ 0`
(`x = ħ²k²/m²c²`, `1/√(1+x) = mc²/ħω`). -/
theorem feynman_amplitude_bounds (x : ℝ) (hx : 0 ≤ x) :
    0 ≤ 1 - 1 / Real.sqrt (1 + x) ∧ 1 - 1 / Real.sqrt (1 + x) ≤ x / 2 := by
  obtain ⟨h1, h2⟩ := charge_ratio_bounds x hx
  have hpos : 0 < Real.sqrt (1 + x) := by linarith
  have hle : 1 / Real.sqrt (1 + x) ≤ 1 := by
    rw [div_le_one hpos]; exact h1
  have hge : 1 - x / 2 ≤ 1 / Real.sqrt (1 + x) := by
    rw [le_div_iff₀ hpos]
    nlinarith [mul_nonneg hx (sub_nonneg.2 h1)]
  constructor <;> linarith

end CalChecks.AlphaTwo

#print axioms CalChecks.AlphaTwo.sin_mul_eq_mul_sin_iff
#print axioms CalChecks.AlphaTwo.lorentz_selects_two
#print axioms CalChecks.AlphaTwo.hfun_two
#print axioms CalChecks.AlphaTwo.hfun_symm
#print axioms CalChecks.AlphaTwo.kg_symbol_zero
#print axioms CalChecks.AlphaTwo.kg_symbol_zero_flipped
#print axioms CalChecks.AlphaTwo.mu_tau
#print axioms CalChecks.AlphaTwo.kg_dispersion
#print axioms CalChecks.AlphaTwo.kg_dispersion_flipped
#print axioms CalChecks.AlphaTwo.neg_mu_sq_pow
#print axioms CalChecks.AlphaTwo.checker_factor
#print axioms CalChecks.AlphaTwo.checker_step
#print axioms CalChecks.AlphaTwo.checker_binomial_bound
#print axioms CalChecks.AlphaTwo.rl_two_factorial
#print axioms CalChecks.AlphaTwo.pathsum_majorant
#print axioms CalChecks.AlphaTwo.pathsum_symbol_geometric
#print axioms CalChecks.AlphaTwo.Mk_add
#print axioms CalChecks.AlphaTwo.Mk_det
#print axioms CalChecks.AlphaTwo.norm_mode
#print axioms CalChecks.AlphaTwo.closure_const
#print axioms CalChecks.AlphaTwo.closure_two_modes
#print axioms CalChecks.AlphaTwo.mixed_mode_not_const
#print axioms CalChecks.AlphaTwo.charge_ratio_bounds
#print axioms CalChecks.AlphaTwo.sqrt_expansion_bounds
#print axioms CalChecks.AlphaTwo.nr_dispersion_bounds
#print axioms CalChecks.AlphaTwo.nr_mode_bound
#print axioms CalChecks.AlphaTwo.feynman_amplitude_bounds
