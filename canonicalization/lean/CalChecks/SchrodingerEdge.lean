/-
Independent Lean checks for the Schrodinger-edge paper (agent L).

Conventions used in this file.
* Memory weights are indexed from 0: `w j` here is the paper's `w_{j+1}`, so the paper's
  `W(z) = sum_{j>=1} w_j z^(-j)` is `∑' j : ℕ, w j * z ^ (-((j : ℤ) + 1))`.
* `g` stands for the paper's spatial factor `ghat`.
-/
import Mathlib
import CalChecks.Basic

open Matrix Complex Filter Topology
open scoped ComplexConjugate

namespace CalChecks.Edge

/-! ## 1. Thm normalizable -/

theorem norm_zpow_neg_succ_le_one (z : ℂ) (hz : 1 ≤ ‖z‖) (j : ℕ) :
    ‖z ^ (-((j : ℤ) + 1))‖ ≤ 1 := by
  have h : z ^ (-((j : ℤ) + 1)) = (z ^ (j + 1))⁻¹ := by
    rw [_root_.zpow_neg]
    congr 1
  rw [h, norm_inv, norm_pow]
  exact inv_le_one_of_one_le₀ (one_le_pow₀ hz)

/-- Norm bound on the memory sum: `|W(z)| ≤ 1` for `|z| ≥ 1` when the weights are
nonnegative with total 1. -/
theorem norm_memory_le_one (w : ℕ → ℝ) (hw : ∀ j, 0 ≤ w j) (hs : Summable w)
    (h1 : ∑' j, w j = 1) (z : ℂ) (hz : 1 ≤ ‖z‖) :
    ‖∑' j : ℕ, (w j : ℂ) * z ^ (-((j : ℤ) + 1))‖ ≤ 1 := by
  have hterm : ∀ j : ℕ, ‖(w j : ℂ) * z ^ (-((j : ℤ) + 1))‖ ≤ w j := by
    intro j
    rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg (hw j)]
    calc w j * ‖z ^ (-((j : ℤ) + 1))‖ ≤ w j * 1 :=
          mul_le_mul_of_nonneg_left (norm_zpow_neg_succ_le_one z hz j) (hw j)
      _ = w j := mul_one _
  have hsn : Summable (fun j : ℕ => ‖(w j : ℂ) * z ^ (-((j : ℤ) + 1))‖) :=
    Summable.of_nonneg_of_le (fun _ => norm_nonneg _) hterm hs
  calc ‖∑' j : ℕ, (w j : ℂ) * z ^ (-((j : ℤ) + 1))‖
      ≤ ∑' j : ℕ, ‖(w j : ℂ) * z ^ (-((j : ℤ) + 1))‖ := norm_tsum_le_tsum_norm hsn
    _ ≤ ∑' j, w j := hsn.tsum_le_tsum hterm hs
    _ = 1 := h1

/-- **Thm normalizable.** Nonnegative summable weights of total 1, `0 < α < 1`,
`0 ≤ g ≤ 1`, `|z| ≥ 1`: the characteristic function does not vanish. -/
theorem normalizable (w : ℕ → ℝ) (hw : ∀ j, 0 ≤ w j) (hs : Summable w)
    (h1 : ∑' j, w j = 1) (α g : ℝ) (hα0 : 0 < α) (hα1 : α < 1) (hg0 : 0 ≤ g) (hg1 : g ≤ 1)
    (z : ℂ) (hz : 1 ≤ ‖z‖) :
    z - (α : ℂ) + (1 - (α : ℂ)) * (g : ℂ) * ∑' j : ℕ, (w j : ℂ) * z ^ (-((j : ℤ) + 1)) ≠ 0 := by
  intro hF
  set S : ℂ := ∑' j : ℕ, (w j : ℂ) * z ^ (-((j : ℤ) + 1)) with hSdef
  have hS : ‖S‖ ≤ 1 := norm_memory_le_one w hw hs h1 z hz
  have hε : 0 < 1 - α := by linarith
  -- the paper's chain
  have heq : z - (α : ℂ) = -((1 - (α : ℂ)) * (g : ℂ) * S) := by linear_combination hF
  have hn : ‖z - (α : ℂ)‖ = (1 - α) * g * ‖S‖ := by
    rw [heq, norm_neg, norm_mul, norm_mul, Complex.norm_real, Real.norm_of_nonneg hg0]
    have : (1 - (α : ℂ)) = ((1 - α : ℝ) : ℂ) := by push_cast; ring
    rw [this, Complex.norm_real, Real.norm_of_nonneg hε.le]
  have hup : ‖z - (α : ℂ)‖ ≤ 1 - α := by
    rw [hn]
    have h2 : g * ‖S‖ ≤ 1 := by
      calc g * ‖S‖ ≤ 1 * 1 := mul_le_mul hg1 hS (norm_nonneg _) zero_le_one
        _ = 1 := mul_one _
    calc (1 - α) * g * ‖S‖ = (1 - α) * (g * ‖S‖) := by ring
      _ ≤ (1 - α) * 1 := mul_le_mul_of_nonneg_left h2 hε.le
      _ = 1 - α := mul_one _
  have hlow : ‖z‖ - α ≤ ‖z - (α : ℂ)‖ := by
    have h := norm_sub_norm_le z (α : ℂ)
    rw [Complex.norm_real, Real.norm_of_nonneg hα0.le] at h
    exact h
  -- equality case: |z| = 1 and |z - α| = 1 - α
  have hz1 : ‖z‖ = 1 := le_antisymm (by linarith) hz
  have hza : ‖z - (α : ℂ)‖ = 1 - α := le_antisymm hup (by linarith)
  have e1 : z.re ^ 2 + z.im ^ 2 = 1 := by
    have h := Complex.sq_norm z
    rw [hz1, Complex.normSq_apply] at h
    nlinarith [h]
  have e2 : (z.re - α) ^ 2 + z.im ^ 2 = (1 - α) ^ 2 := by
    have h := Complex.sq_norm (z - (α : ℂ))
    rw [hza, Complex.normSq_apply] at h
    simp only [Complex.sub_re, Complex.sub_im, Complex.ofReal_re, Complex.ofReal_im,
      sub_zero] at h
    nlinarith [h]
  have hre : z.re = 1 := by
    have h3 : α * (z.re - 1) = 0 := by nlinarith [e1, e2]
    rcases mul_eq_zero.mp h3 with h | h
    · exact absurd h hα0.ne'
    · linarith
  have him : z.im = 0 := by
    have h3 : z.im ^ 2 = 0 := by nlinarith [e1, hre]
    exact pow_eq_zero_iff (two_ne_zero) |>.mp h3
  have hzone : z = 1 := Complex.ext (by simp [hre]) (by simp [him])
  -- F(1) = (1 - α)(1 + g) ≠ 0
  have hS1 : S = 1 := by
    rw [hSdef, hzone]
    simp only [_root_.one_zpow, mul_one]
    rw [← Complex.ofReal_tsum, h1, Complex.ofReal_one]
  rw [hzone, hS1] at hF
  have hr : ((1 - α + (1 - α) * g : ℝ) : ℂ) = 0 := by
    push_cast
    linear_combination hF
  have hr' : 1 - α + (1 - α) * g = 0 := by exact_mod_cast hr
  nlinarith [mul_nonneg hε.le hg0]

/-- The paper's equality case: `F(1) = (1 - α)(1 + g)`, which is `2(1 - α)` at `g = 1`. -/
theorem F_one (w : ℕ → ℝ) (hw1 : ∑' j, w j = 1) (α g : ℝ) :
    (1 : ℂ) - (α : ℂ) + (1 - (α : ℂ)) * (g : ℂ) * ∑' j : ℕ, (w j : ℂ) * (1 : ℂ) ^ (-((j : ℤ) + 1))
      = (((1 - α) * (1 + g) : ℝ) : ℂ) := by
  simp only [_root_.one_zpow, mul_one]
  rw [← Complex.ofReal_tsum, hw1]
  push_cast
  ring

/-- The hypothesis `0 < α` cannot be dropped to `0 ≤ α`: at `α = 0`, `g = 1`, Dirac kernel
(all weight on lag 1), `z = i` is a root on the unit circle. Stated without the sum:
`W(i) = i^(-1)`. -/
theorem alpha_zero_root : (I : ℂ) - 0 + (1 - 0) * 1 * (I : ℂ) ^ (-(1 : ℤ)) = 0 ∧ ‖(I : ℂ)‖ = 1 := by
  refine ⟨?_, Complex.norm_I⟩
  rw [_root_.zpow_neg_one, Complex.inv_I]
  ring

/-! ## 2. Lemma flat -/

/-- Flat weights: `W(z) = sum_{j>=1} z^(-j) = 1/(z-1)` for `|z| > 1`. -/
theorem flat_W (z : ℂ) (hz : 1 < ‖z‖) :
    ∑' j : ℕ, z ^ (-((j : ℤ) + 1)) = 1 / (z - 1) := by
  have hz0 : z ≠ 0 := by
    intro h
    rw [h, norm_zero] at hz
    linarith
  have hlt : ‖z⁻¹‖ < 1 := by
    rw [norm_inv]
    exact inv_lt_one_of_one_lt₀ hz
  have hterm : ∀ j : ℕ, z ^ (-((j : ℤ) + 1)) = z⁻¹ * z⁻¹ ^ j := by
    intro j
    rw [_root_.zpow_neg, ← _root_.inv_zpow, zpow_add_one₀ (inv_ne_zero hz0), zpow_natCast]
    ring
  simp_rw [hterm]
  rw [tsum_mul_left, tsum_geometric_of_norm_lt_one hlt]
  have h1 : z - 1 ≠ 0 := by
    intro h
    have : z = 1 := by linear_combination h
    rw [this, norm_one] at hz
    linarith
  have h2 : (1 : ℂ) - z⁻¹ ≠ 0 := by
    intro h
    apply h1
    have h3 : z⁻¹ = 1 := by linear_combination -h
    have h4 : z = 1 := by rw [← inv_inv z, h3, inv_one]
    linear_combination h4
  field_simp

/-- Lemma flat, displayed identity. -/
theorem flat_factor (α g z : ℂ) (hz : z ≠ 1) :
    (z - 1) * (z - α + (1 - α) * g / (z - 1)) = z ^ 2 - (1 + α) * z + α + (1 - α) * g := by
  have h : z - 1 ≠ 0 := sub_ne_zero.mpr hz
  field_simp
  ring

/-- Lemma flat (1), algebra: if `cos φ = (1+α)/2` then at `g = 1` the quadratic factors as
`(z - e^{iφ})(z - e^{-iφ})`. -/
theorem flat_lt_factor (α φ : ℝ) (hc : Real.cos φ = (1 + α) / 2) (z : ℂ) :
    z ^ 2 - (1 + (α : ℂ)) * z + α + (1 - α) * 1
      = (z - Complex.exp (φ * I)) * (z - Complex.exp (-(φ * I))) := by
  have hc' : Complex.cos (φ : ℂ) = (1 + (α : ℂ)) / 2 := by
    rw [← Complex.ofReal_cos, hc]
    push_cast
    ring
  have hsum : Complex.exp (φ * I) + Complex.exp (-(φ * I)) = 1 + (α : ℂ) := by
    have h := Complex.cos.eq_1 (φ : ℂ)
    rw [hc'] at h
    have e : -(φ : ℂ) * I = -((φ : ℂ) * I) := by ring
    rw [e] at h
    linear_combination -2 * h
  have hprod : Complex.exp (φ * I) * Complex.exp (-(φ * I)) = 1 := by
    rw [← Complex.exp_add]
    simp
  linear_combination z * hsum - hprod

/-- Lemma flat (1): for `0 < α < 1` there is a real angle `φ` with `cos φ = (1+α)/2`;
`e^{iφ}` and `e^{-iφ}` are roots at `g = 1`, both of norm exactly 1. -/
theorem flat_lt (α : ℝ) (hα0 : 0 < α) (hα1 : α < 1) :
    ∃ φ : ℝ, 0 < φ ∧ Real.cos φ = (1 + α) / 2 ∧
      (Complex.exp (φ * I)) ^ 2 - (1 + (α : ℂ)) * Complex.exp (φ * I) + α + (1 - α) * 1 = 0 ∧
      (Complex.exp (-(φ * I))) ^ 2 - (1 + (α : ℂ)) * Complex.exp (-(φ * I)) + α + (1 - α) * 1 = 0 ∧
      ‖Complex.exp (φ * I)‖ = 1 ∧ ‖Complex.exp (-(φ * I))‖ = 1 := by
  have hc : Real.cos (Real.arccos ((1 + α) / 2)) = (1 + α) / 2 :=
    Real.cos_arccos (by linarith) (by linarith)
  refine ⟨Real.arccos ((1 + α) / 2), ?_, hc, ?_, ?_, ?_, ?_⟩
  · exact Real.arccos_pos.mpr (by linarith)
  · rw [flat_lt_factor α _ hc]; simp
  · rw [flat_lt_factor α _ hc]; simp
  · exact Complex.norm_exp_ofReal_mul_I _
  · have e : -((Real.arccos ((1 + α) / 2) : ℂ) * I) = ((-Real.arccos ((1 + α) / 2) : ℝ) : ℂ) * I := by
      push_cast; ring
    rw [e]
    exact Complex.norm_exp_ofReal_mul_I _

/-- Lemma flat (2): at `α = 1`, `g = 1` the quadratic is `(z-1)^2`. -/
theorem flat_eq_one (z : ℂ) :
    z ^ 2 - (1 + (1 : ℂ)) * z + 1 + (1 - 1) * 1 = (z - 1) ^ 2 := by
  ring

/-- Lemma flat (3), algebra: if `cosh s = (1+α)/2` then at `g = 1` the quadratic factors as
`(z - e^s)(z - e^{-s})`. -/
theorem flat_gt_factor (α s : ℝ) (hc : Real.cosh s = (1 + α) / 2) (z : ℂ) :
    z ^ 2 - (1 + (α : ℂ)) * z + α + (1 - α) * 1
      = (z - (Real.exp s : ℂ)) * (z - (Real.exp (-s) : ℂ)) := by
  have hsum : Real.exp s + Real.exp (-s) = 1 + α := by
    rw [Real.cosh_eq] at hc
    linarith
  have hprod : Real.exp s * Real.exp (-s) = 1 := by
    rw [← Real.exp_add]; simp
  have hsumC : (Real.exp s : ℂ) + (Real.exp (-s) : ℂ) = 1 + (α : ℂ) := by
    exact_mod_cast hsum
  have hprodC : (Real.exp s : ℂ) * (Real.exp (-s) : ℂ) = 1 := by
    exact_mod_cast hprod
  linear_combination z * hsumC - hprodC

/-- Lemma flat (3): for `α > 1` there is `s > 0` with `cosh s = (1+α)/2`, and the roots at
`g = 1` are exactly `e^s` and `e^{-s}`. -/
theorem flat_gt (α : ℝ) (hα : 1 < α) :
    ∃ s : ℝ, 0 < s ∧ Real.cosh s = (1 + α) / 2 ∧
      ∀ z : ℂ, z ^ 2 - (1 + (α : ℂ)) * z + α + (1 - α) * 1 = 0 ↔
        z = (Real.exp s : ℂ) ∨ z = (Real.exp (-s) : ℂ) := by
  have h1 : 1 < (1 + α) / 2 := by linarith
  have hc : Real.cosh (Real.arcosh ((1 + α) / 2)) = (1 + α) / 2 := Real.cosh_arcosh h1.le
  refine ⟨Real.arcosh ((1 + α) / 2), Real.arcosh_pos h1, hc, ?_⟩
  intro z
  rw [flat_gt_factor α _ hc, mul_eq_zero, sub_eq_zero, sub_eq_zero]

/-! ## 3. Thm feedback (1) and (3) -/

/-- Thm feedback (1): at `α = 1` the right side of the update rule is `J n`, whatever the
weights and the past. (In Lean a divergent `tsum` is 0, so this says nothing about whether
the memory sum converges.) -/
theorem feedback_frozen (J : ℤ → ℂ) (w : ℕ → ℝ) (g : ℂ) (n : ℤ) :
    ((1 : ℝ) : ℂ) * J n - (1 - ((1 : ℝ) : ℂ)) * g * ∑' j : ℕ, (w j : ℂ) * J (n - (j + 1)) = J n := by
  simp

theorem exp_three_term (s t : ℝ) :
    Real.exp (-s * (t + 1)) - (Real.exp s + Real.exp (-s)) * Real.exp (-s * t)
      + Real.exp (-s * (t - 1)) = 0 := by
  have a : Real.exp (-s * (t + 1)) = Real.exp (-s * t) * Real.exp (-s) := by
    rw [← Real.exp_add]; ring_nf
  have b : Real.exp (-s * (t - 1)) = Real.exp (-s * t) * Real.exp s := by
    rw [← Real.exp_add]; ring_nf
  rw [a, b]; ring

/-- The two-sided profile `G n = e^{-s|n-n0|}`. -/
noncomputable def G (s : ℝ) (n0 n : ℤ) : ℝ := Real.exp (-s * ((|n - n0| : ℤ) : ℝ))

/-- Thm feedback (3), away from the source: `G(n+1) - (1+α) G(n) + G(n-1) = 0`. -/
theorem green_homogeneous (α s : ℝ) (hc : Real.cosh s = (1 + α) / 2) (n0 n : ℤ) (hn : n ≠ n0) :
    G s n0 (n + 1) - (1 + α) * G s n0 n + G s n0 (n - 1) = 0 := by
  have hα : 1 + α = Real.exp s + Real.exp (-s) := by
    rw [Real.cosh_eq] at hc; linarith
  unfold G
  rw [hα]
  rcases lt_or_gt_of_ne hn with h | h
  · have h1 : |n + 1 - n0| = (n0 - n) - 1 := by
      rw [abs_of_nonpos (by omega)]; ring
    have h2 : |n - n0| = n0 - n := by
      rw [abs_of_nonpos (by omega)]; ring
    have h3 : |n - 1 - n0| = (n0 - n) + 1 := by
      rw [abs_of_nonpos (by omega)]; ring
    rw [h1, h2, h3]
    push_cast
    have := exp_three_term s ((n0 : ℝ) - n)
    linarith
  · have h1 : |n + 1 - n0| = (n - n0) + 1 := by
      rw [abs_of_nonneg (by omega)]; ring
    have h2 : |n - n0| = n - n0 := abs_of_nonneg (by omega)
    have h3 : |n - 1 - n0| = (n - n0) - 1 := by
      rw [abs_of_nonneg (by omega)]; ring
    rw [h1, h2, h3]
    push_cast
    have := exp_three_term s ((n : ℝ) - n0)
    linarith

/-- Thm feedback (3), at the source: the jump is `-2 sinh s`. -/
theorem green_jump (α s : ℝ) (hc : Real.cosh s = (1 + α) / 2) (n0 : ℤ) :
    G s n0 (n0 + 1) - (1 + α) * G s n0 n0 + G s n0 (n0 - 1) = -(2 * Real.sinh s) := by
  have hα : 1 + α = Real.exp s + Real.exp (-s) := by
    rw [Real.cosh_eq] at hc; linarith
  unfold G
  rw [hα, Real.sinh_eq]
  have h1 : |n0 + 1 - n0| = 1 := by simp
  have h2 : |n0 - n0| = 0 := by simp
  have h3 : |n0 - 1 - n0| = 1 := by simp
  rw [h1, h2, h3]
  simp
  ring

/-- Thm feedback (3), normalized as in the paper's proof: with `Gn = G/(2 sinh s)` the
recurrence operator gives `-1` at the source and `0` elsewhere. -/
theorem green_normalized (α s : ℝ) (hs : 0 < s) (hc : Real.cosh s = (1 + α) / 2) (n0 n : ℤ) :
    G s n0 (n + 1) / (2 * Real.sinh s) - (1 + α) * (G s n0 n / (2 * Real.sinh s))
      + G s n0 (n - 1) / (2 * Real.sinh s) = if n = n0 then -1 else 0 := by
  have hsh : 0 < Real.sinh s := Real.sinh_pos_iff.mpr hs
  have hne : 2 * Real.sinh s ≠ 0 := by positivity
  have key : G s n0 (n + 1) / (2 * Real.sinh s) - (1 + α) * (G s n0 n / (2 * Real.sinh s))
      + G s n0 (n - 1) / (2 * Real.sinh s)
      = (G s n0 (n + 1) - (1 + α) * G s n0 n + G s n0 (n - 1)) / (2 * Real.sinh s) := by
    field_simp
  rw [key]
  split_ifs with h
  · subst h
    rw [green_jump α s hc]
    field_simp
  · rw [green_homogeneous α s hc n0 n h]
    simp

/-- `G` is bounded by 1 for `s ≥ 0`. -/
theorem green_bounded (s : ℝ) (hs : 0 ≤ s) (n0 n : ℤ) : 0 < G s n0 n ∧ G s n0 n ≤ 1 := by
  unfold G
  refine ⟨Real.exp_pos _, ?_⟩
  rw [Real.exp_le_one_iff]
  have : (0 : ℝ) ≤ ((|n - n0| : ℤ) : ℝ) := by exact_mod_cast abs_nonneg (n - n0)
  nlinarith

/-! ## 4. Prop accumulator, discrete part -/

/-- For each fixed `j ≥ 1`, `(j^α - (j-1)^α)/Γ(α+1) → 1` as `α → 1⁻`. -/
theorem weight_tendsto_one (j : ℕ) (_hj : 1 ≤ j) :
    Tendsto (fun α : ℝ => (((j : ℝ) ^ α - ((j : ℝ) - 1) ^ α)) / Real.Gamma (α + 1))
      (𝓝[<] 1) (𝓝 1) := by
  have hcont : ContinuousAt
      (fun α : ℝ => (((j : ℝ) ^ α - ((j : ℝ) - 1) ^ α)) / Real.Gamma (α + 1)) 1 := by
    have h1 : ContinuousAt (fun α : ℝ => (j : ℝ) ^ α) 1 :=
      Real.continuousAt_const_rpow' one_ne_zero
    have h2 : ContinuousAt (fun α : ℝ => ((j : ℝ) - 1) ^ α) 1 :=
      Real.continuousAt_const_rpow' one_ne_zero
    have hG : ContinuousAt Real.Gamma ((1 : ℝ) + 1) := by
      refine (Real.differentiableAt_Gamma (fun m h => ?_)).continuousAt
      have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
      linarith
    have h3 : ContinuousAt (fun α : ℝ => Real.Gamma (α + 1)) 1 :=
      ContinuousAt.comp (f := fun α : ℝ => α + 1) (g := Real.Gamma) hG
        (continuousAt_id.add continuousAt_const)
    refine (h1.sub h2).div h3 ?_
    have : (1 : ℝ) + 1 = 2 := by norm_num
    rw [this, Real.Gamma_two]
    exact one_ne_zero
  have hval : (((j : ℝ) ^ (1 : ℝ) - ((j : ℝ) - 1) ^ (1 : ℝ))) / Real.Gamma (1 + 1) = 1 := by
    have : (1 : ℝ) + 1 = 2 := by norm_num
    rw [this, Real.Gamma_two, Real.rpow_one, Real.rpow_one]
    ring
  have h := hcont.tendsto
  rw [hval] at h
  exact h.mono_left nhdsWithin_le_nhds

/-! ## 5. Prop carrier and Prop lorentz -/

/-- Prop carrier (3): `det(J J†) = |det J|²`. -/
theorem det_closure (J : M2) : (J * Jᴴ).det = ((‖J.det‖ ^ 2 : ℝ) : ℂ) := by
  rw [det_mul, det_conjTranspose, RCLike.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq]

/-- The concurrence `C = 2|det J| / tr(J J†)`. -/
noncomputable def conc (J : M2) : ℝ := 2 * ‖J.det‖ / (J * Jᴴ).trace.re

theorem closure_data (J : M2) (t x y z : ℝ) (hQ : J * Jᴴ = herm t x y z) :
    ‖J.det‖ ^ 2 = t ^ 2 - x ^ 2 - y ^ 2 - z ^ 2 ∧ (J * Jᴴ).trace.re = 2 * t := by
  constructor
  · have h := det_closure J
    rw [hQ, det_herm] at h
    exact_mod_cast h.symm
  · rw [hQ, herm_eq]
    simp [trace_fin_two]
    ring

/-- Prop carrier (4), squared form: with `J J† = x0 + x.σ`, `x0 ≠ 0`,
`C² = 1 - |x|²/x0²`. -/
theorem conc_sq (J : M2) (t x y z : ℝ) (ht : t ≠ 0) (hQ : J * Jᴴ = herm t x y z) :
    conc J ^ 2 = 1 - (x ^ 2 + y ^ 2 + z ^ 2) / t ^ 2 := by
  obtain ⟨hd, htr⟩ := closure_data J t x y z hQ
  unfold conc
  rw [htr, div_pow, mul_pow, hd]
  field_simp
  ring

/-- Prop carrier (4): for `x0 > 0`, `C = sqrt(1 - |x|²/x0²)`. -/
theorem conc_eq_sqrt (J : M2) (t x y z : ℝ) (ht : 0 < t) (hQ : J * Jᴴ = herm t x y z) :
    conc J = Real.sqrt (1 - (x ^ 2 + y ^ 2 + z ^ 2) / t ^ 2) := by
  have h := conc_sq J t x y z ht.ne' hQ
  have hnn : 0 ≤ conc J := by
    obtain ⟨_, htr⟩ := closure_data J t x y z hQ
    unfold conc
    rw [htr]
    positivity
  rw [← h, Real.sqrt_sq hnn]

/-- Prop lorentz (3): with `x0 = ρ cosh η`, `|x| = ρ sinh η`, `ρ > 0`: `C = 1/cosh η`. -/
theorem conc_sech (J : M2) (ρ η x y z : ℝ) (hρ : 0 < ρ)
    (hQ : J * Jᴴ = herm (ρ * Real.cosh η) x y z)
    (hx : x ^ 2 + y ^ 2 + z ^ 2 = (ρ * Real.sinh η) ^ 2) :
    conc J = 1 / Real.cosh η := by
  obtain ⟨hd, htr⟩ := closure_data J _ x y z hQ
  have hch : 0 < Real.cosh η := Real.cosh_pos η
  have hcs := Real.cosh_sq η
  have hd2 : ‖J.det‖ ^ 2 = ρ ^ 2 := by
    rw [hd]
    nlinarith [hx, hcs]
  have hdet : ‖J.det‖ = ρ := by
    rw [← abs_of_nonneg (norm_nonneg J.det), ← abs_of_pos hρ]
    exact (sq_eq_sq_iff_abs_eq_abs _ _).mp hd2
  unfold conc
  rw [htr, hdet]
  field_simp

/-- Prop lorentz (3): Schmidt weights `p± = (x0 ± |x|)/(2 x0) = e^{±η}/(2 cosh η)`. -/
theorem schmidt_weights (ρ η : ℝ) (hρ : 0 < ρ) :
    (ρ * Real.cosh η + ρ * Real.sinh η) / (2 * (ρ * Real.cosh η))
        = Real.exp η / (2 * Real.cosh η) ∧
    (ρ * Real.cosh η - ρ * Real.sinh η) / (2 * (ρ * Real.cosh η))
        = Real.exp (-η) / (2 * Real.cosh η) := by
  have hch : 0 < Real.cosh η := Real.cosh_pos η
  have h1 : Real.cosh η + Real.sinh η = Real.exp η := Real.cosh_add_sinh η
  have h2 : Real.cosh η - Real.sinh η = Real.exp (-η) := Real.cosh_sub_sinh η
  constructor
  · rw [← h1]; field_simp
  · rw [← h2]; field_simp

/-- Prop lorentz (1): `|det(G J)| = |det J|` for `det G = 1`, and the closure moves by
congruence. Reuses `CalChecks.det_congruence_SL`. -/
theorem lorentz_invariant_density (G J : M2) (hG : G.det = 1) :
    ‖(G * J).det‖ = ‖J.det‖ ∧ (G * J) * (G * J)ᴴ = G * (J * Jᴴ) * Gᴴ ∧
      (G * (J * Jᴴ) * Gᴴ).det = (J * Jᴴ).det := by
  refine ⟨?_, ?_, CalChecks.det_congruence_SL G (J * Jᴴ) hG⟩
  · rw [det_mul, hG, one_mul]
  · rw [conjTranspose_mul]
    simp only [Matrix.mul_assoc]

/-! ## 6. Lemma minors -/

/-- Lemma minors, proof step: `2J = J J†` forces `J† = J` and `J² = 2J` (any size). -/
theorem self_normalized {n : Type*} [Fintype n] [DecidableEq n] (J : Matrix n n ℂ)
    (h : (2 : ℂ) • J = J * Jᴴ) : Jᴴ = J ∧ J * J = (2 : ℂ) • J := by
  have hH : Jᴴ = J := by
    have h2 : ((2 : ℂ) • J)ᴴ = (2 : ℂ) • J := by
      rw [h, conjTranspose_mul, conjTranspose_conjTranspose]
    rw [conjTranspose_smul] at h2
    have h3 : star (2 : ℂ) = 2 := by simp
    rw [h3] at h2
    exact smul_right_injective (Matrix n n ℂ) (two_ne_zero : (2 : ℂ) ≠ 0) h2
  refine ⟨hH, ?_⟩
  rw [h, hH]

/-- Lemma minors, the `N = 3` example `J = diag(2,2,0)`: self-normalized, `det J = 0`,
yet `tr(J/2) = 2`. -/
theorem minors_example :
    let J : Matrix (Fin 3) (Fin 3) ℂ := Matrix.diagonal ![2, 2, 0]
    (2 : ℂ) • J = J * Jᴴ ∧ J.det = 0 ∧ (((1 : ℂ) / 2) • J).trace = 2 := by
  intro J
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;>
      simp [J, Matrix.mul_apply, Matrix.diagonal, conjTranspose_apply, map_ofNat]
  · simp [J, Matrix.det_diagonal, Fin.prod_univ_three]
  · simp [J, Matrix.trace, Fin.sum_univ_three]
    norm_num

end CalChecks.Edge

#print axioms CalChecks.Edge.normalizable
#print axioms CalChecks.Edge.F_one
#print axioms CalChecks.Edge.alpha_zero_root
#print axioms CalChecks.Edge.flat_W
#print axioms CalChecks.Edge.flat_factor
#print axioms CalChecks.Edge.flat_lt
#print axioms CalChecks.Edge.flat_eq_one
#print axioms CalChecks.Edge.flat_gt
#print axioms CalChecks.Edge.feedback_frozen
#print axioms CalChecks.Edge.green_homogeneous
#print axioms CalChecks.Edge.green_jump
#print axioms CalChecks.Edge.green_normalized
#print axioms CalChecks.Edge.green_bounded
#print axioms CalChecks.Edge.weight_tendsto_one
#print axioms CalChecks.Edge.det_closure
#print axioms CalChecks.Edge.conc_sq
#print axioms CalChecks.Edge.conc_eq_sqrt
#print axioms CalChecks.Edge.conc_sech
#print axioms CalChecks.Edge.schmidt_weights
#print axioms CalChecks.Edge.lorentz_invariant_density
#print axioms CalChecks.Edge.self_normalized
#print axioms CalChecks.Edge.minors_example
