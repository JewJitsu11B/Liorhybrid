/-
Native Clifford checks for Cl(3,0).

Statements are made with Mathlib's `CliffordAlgebra.reverse` and `CliffordAlgebra.involute`
on `Cl3 = CliffordAlgebra Q3`.  The dictionary to `M₂(ℂ)` is proved here as theorems about
`cliffordToM2` / `cliffordEquiv`; the matrix-free statements of part 2 are then proved by
transporting through `cliffordToM2`, which is injective.

Definitions used (all intrinsic to `Cl3`):
* `cconj q      = reverse (involute q)`                       Clifford conjugation
* `N q          = q * cconj q`
* `gradeZero q  = (1/4) (q + reverse q + involute q + cconj q)`
* `gradeOne q   = (1/4) (q + reverse q - involute q - cconj q)`
* `gradeTwo q   = (1/4) (q - reverse q + involute q - cconj q)`
* `gradeThree q = (1/4) (q - reverse q - involute q + cconj q)`
* `par q        = (1/2) (q + reverse q)`   (grades 0+1)
* `apar q       = (1/2) (q - reverse q)`   (grades 3+2)
* `Ips          = γ 0 * γ 1 * γ 2`         pseudoscalar
* `cl J         = J * reverse J`           closure
* `biv a b c    = a γ1γ2 + b γ2γ0 + c γ0γ1`
* `spin3`       = the real Lie subalgebra `{A | reverse A = -A ∧ involute A = A}` (bivectors)
-/
import Mathlib
import CalChecks.Final
import Physlib.Relativity.PauliMatrices.Basic

open Matrix Complex
open scoped ComplexConjugate

namespace CalChecks

attribute [local instance] LieRing.ofAssociativeRing LieAlgebra.ofAssociativeAlgebra

noncomputable section

/-! ## Definitions -/

/-- Clifford conjugation: reverse of involute. -/
def cconj (q : Cl3) : Cl3 := CliffordAlgebra.reverse (CliffordAlgebra.involute q)

/-- `N q = q * conj q`. -/
def N (q : Cl3) : Cl3 := q * cconj q

def gradeZero (q : Cl3) : Cl3 :=
  (1 / 4 : ℝ) • (q + CliffordAlgebra.reverse q + CliffordAlgebra.involute q + cconj q)
def gradeOne (q : Cl3) : Cl3 :=
  (1 / 4 : ℝ) • (q + CliffordAlgebra.reverse q - CliffordAlgebra.involute q - cconj q)
def gradeTwo (q : Cl3) : Cl3 :=
  (1 / 4 : ℝ) • (q - CliffordAlgebra.reverse q + CliffordAlgebra.involute q - cconj q)
def gradeThree (q : Cl3) : Cl3 :=
  (1 / 4 : ℝ) • (q - CliffordAlgebra.reverse q - CliffordAlgebra.involute q + cconj q)

/-- Reverse-fixed part (grades 0+1). -/
def par (q : Cl3) : Cl3 := (1 / 2 : ℝ) • (q + CliffordAlgebra.reverse q)
/-- Reverse-odd part (grades 3+2). -/
def apar (q : Cl3) : Cl3 := (1 / 2 : ℝ) • (q - CliffordAlgebra.reverse q)

/-- Pseudoscalar. -/
def Ips : Cl3 := γ 0 * γ 1 * γ 2

/-- Closure `c J = J * reverse J`. -/
def cl (J : Cl3) : Cl3 := J * CliffordAlgebra.reverse J

/-- Bivector `a γ1γ2 + b γ2γ0 + c γ0γ1`. -/
def biv (a b c : ℝ) : Cl3 := a • (γ 1 * γ 2) + b • (γ 2 * γ 0) + c • (γ 0 * γ 1)

theorem cliffordEquiv_apply (q : Cl3) : cliffordEquiv q = cliffordToM2 q := rfl

/-! ## Matrix lemmas used for transport -/

theorem algebraMap_M2_conjTranspose (r : ℝ) : (algebraMap ℝ M2 r)ᴴ = algebraMap ℝ M2 r := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.algebraMap_matrix_apply, conjTranspose_apply]

theorem adj_add (A B : M2) : adjugate (A + B) = adjugate A + adjugate B := by
  rw [adjugate_fin_two, adjugate_fin_two, adjugate_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> ring

theorem adj_algebraMap (r : ℝ) : adjugate (algebraMap ℝ M2 r) = algebraMap ℝ M2 r := by
  rw [adjugate_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.algebraMap_matrix_apply]

theorem adj_dotσ (v : Fin 3 → ℝ) : adjugate (dotσ v) = -dotσ v := by
  rw [dotσ_eq, adjugate_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

theorem real_half_smul (X : M2) : (1 / 2 : ℝ) • X = (1 / 2 : ℂ) • X := by
  ext i j
  simp

theorem four_parts (M : M2) :
    M + Mᴴ + (adjugate M)ᴴ + adjugate M = ((2 * M.trace.re : ℝ) : ℂ) • (1 : M2) := by
  rw [adjugate_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;> apply Complex.ext <;>
    simp [trace_fin_two, conjTranspose_apply] <;> ring

theorem I_herm_explicit (a b c : ℝ) :
    I • herm 0 a b c
      = !![(c : ℂ) * I, (a : ℂ) * I + b; (a : ℂ) * I - b, -((c : ℂ) * I)] := by
  rw [herm_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> apply Complex.ext <;> simp

theorem I_herm_conjTranspose (a b c : ℝ) : (I • herm 0 a b c)ᴴ = -(I • herm 0 a b c) := by
  rw [conjTranspose_smul, herm_isHermitian]
  simp

theorem adj_I_herm (a b c : ℝ) : adjugate (I • herm 0 a b c) = -(I • herm 0 a b c) := by
  rw [I_herm_explicit, adjugate_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

/-! ## 1. The dictionary -/

/-- 1a. Reversion is the conjugate transpose. -/
theorem toM2_reverse (q : Cl3) :
    cliffordToM2 (CliffordAlgebra.reverse q) = (cliffordToM2 q)ᴴ := by
  induction q using CliffordAlgebra.induction with
  | algebraMap r =>
    simp only [CliffordAlgebra.reverse.commutes, AlgHom.commutes, algebraMap_M2_conjTranspose]
  | ι x => rw [CliffordAlgebra.reverse_ι, cliffordToM2_ι, dotσ_hermitian]
  | mul a b ha hb =>
    rw [CliffordAlgebra.reverse.map_mul, map_mul, map_mul, conjTranspose_mul, ha, hb]
  | add a b ha hb => rw [map_add, map_add, map_add, conjTranspose_add, ha, hb]

/-- Grade involution is `M ↦ (adjugate M)ᴴ`. -/
theorem toM2_involute (q : Cl3) :
    cliffordToM2 (CliffordAlgebra.involute q) = (adjugate (cliffordToM2 q))ᴴ := by
  induction q using CliffordAlgebra.induction with
  | algebraMap r =>
    simp only [AlgHom.commutes, adj_algebraMap, algebraMap_M2_conjTranspose]
  | ι x =>
    rw [CliffordAlgebra.involute_ι, map_neg, cliffordToM2_ι, adj_dotσ, conjTranspose_neg,
      dotσ_hermitian]
  | mul a b ha hb =>
    rw [map_mul, map_mul, map_mul, adjugate_mul_distrib, conjTranspose_mul, ha, hb]
  | add a b ha hb => rw [map_add, map_add, map_add, adj_add, conjTranspose_add, ha, hb]

/-- 1b. Clifford conjugation is the adjugate. -/
theorem toM2_cconj (q : Cl3) : cliffordToM2 (cconj q) = adjugate (cliffordToM2 q) := by
  rw [cconj, toM2_reverse, toM2_involute, conjTranspose_conjTranspose]

/-- 1c. `N q = q * conj q` is `det`. -/
theorem toM2_N (q : Cl3) : cliffordToM2 (N q) = (cliffordToM2 q).det • (1 : M2) := by
  rw [N, map_mul, toM2_cconj, mul_adjugate]

theorem dict_reverse (q : Cl3) :
    cliffordEquiv (CliffordAlgebra.reverse q) = (cliffordEquiv q)ᴴ := toM2_reverse q
theorem dict_involute (q : Cl3) :
    cliffordEquiv (CliffordAlgebra.involute q) = (adjugate (cliffordEquiv q))ᴴ := toM2_involute q
theorem dict_cconj (q : Cl3) : cliffordEquiv (cconj q) = adjugate (cliffordEquiv q) :=
  toM2_cconj q
theorem dict_N (q : Cl3) :
    cliffordEquiv (q * cconj q) = (cliffordEquiv q).det • (1 : M2) := toM2_N q

/-- 1d. Scalar part on the matrix side. -/
theorem toM2_gradeZero (q : Cl3) :
    cliffordToM2 (gradeZero q) = algebraMap ℝ M2 ((cliffordToM2 q).trace.re / 2) := by
  rw [gradeZero, map_smul, map_add, map_add, map_add, toM2_reverse, toM2_involute, toM2_cconj,
    four_parts]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.algebraMap_matrix_apply] <;> ring

/-- 1d. The grade-0 part is the scalar `(1/2) Re tr`. -/
theorem gradeZero_eq (q : Cl3) :
    gradeZero q = algebraMap ℝ Cl3 ((cliffordEquiv q).trace.re / 2) := by
  apply cliffordToM2_injective
  rw [toM2_gradeZero, AlgHom.commutes, cliffordEquiv_apply]

/-- 1d. `2 <q>_0 = Re tr`. -/
theorem dict_scalar (q : Cl3) :
    (2 : ℝ) • gradeZero q = algebraMap ℝ Cl3 ((cliffordEquiv q).trace.re) := by
  rw [gradeZero_eq, Algebra.smul_def, ← map_mul]
  congr 1
  ring

/-- 1d. Grades 0+1 (reverse-fixed part) is the Hermitian part. -/
theorem dict_par (q : Cl3) : cliffordEquiv (par q) = H (cliffordEquiv q) := by
  rw [cliffordEquiv_apply, cliffordEquiv_apply, par, map_smul, map_add, toM2_reverse, H,
    real_half_smul]

/-- 1d. Grades 3+2 (reverse-odd part) is the anti-Hermitian part. -/
theorem dict_apar (q : Cl3) : cliffordEquiv (apar q) = A (cliffordEquiv q) := by
  rw [cliffordEquiv_apply, cliffordEquiv_apply, apar, map_smul, map_sub, toM2_reverse, A,
    real_half_smul]

theorem grade_sum (q : Cl3) : gradeZero q + gradeOne q + gradeTwo q + gradeThree q = q := by
  simp only [gradeZero, gradeOne, gradeTwo, gradeThree]
  module

theorem par_eq (q : Cl3) : par q = gradeZero q + gradeOne q := by
  simp only [par, gradeZero, gradeOne]
  module

theorem apar_eq (q : Cl3) : apar q = gradeTwo q + gradeThree q := by
  simp only [apar, gradeTwo, gradeThree]
  module

/-- 1e. The pseudoscalar maps to `i·1`. -/
theorem toM2_Ips : cliffordToM2 Ips = I • (1 : M2) := by
  rw [Ips, mul_assoc, F_vol]

theorem dict_Ips : cliffordEquiv Ips = I • (1 : M2) := toM2_Ips

/-- 1e. The pseudoscalar is central. -/
theorem Ips_central (q : Cl3) : Ips * q = q * Ips := by
  apply cliffordToM2_injective
  rw [map_mul, map_mul, toM2_Ips, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_smul,
    Matrix.mul_one]

theorem Ips_mem_center : Ips ∈ Subalgebra.center ℝ Cl3 :=
  Subalgebra.mem_center_iff.mpr fun b => (Ips_central b).symm

/-- 1e. `I * I = -1`. -/
theorem Ips_sq : Ips * Ips = -1 := by
  apply cliffordToM2_injective
  rw [map_mul, toM2_Ips, map_neg, map_one]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

/-! ## 2. Matrix-free statements -/

/-- 2a. The closure is reverse-fixed. -/
theorem cl_reverse (J : Cl3) : CliffordAlgebra.reverse (cl J) = cl J := by
  rw [cl, CliffordAlgebra.reverse.map_mul, CliffordAlgebra.reverse_reverse]

/-- 2a. The closure lies in grades 0+1: `par (c J) = c J`. -/
theorem cl_par (J : Cl3) : par (cl J) = cl J := by
  rw [par, cl_reverse]
  module

/-- 2a. Equivariance of the closure. -/
theorem cl_equivariant (G J : Cl3) : cl (G * J) = G * cl J * CliffordAlgebra.reverse G := by
  simp only [cl, CliffordAlgebra.reverse.map_mul, mul_assoc]

/-- 2b. `N` is multiplicative. -/
theorem N_mul (a b : Cl3) : N (a * b) = N a * N b := by
  apply cliffordToM2_injective
  rw [map_mul, toM2_N, toM2_N, toM2_N, map_mul, det_mul, Matrix.smul_mul, Matrix.one_mul,
    smul_smul]

/-- 2b. `N q` commutes with everything. -/
theorem N_central (q x : Cl3) : N q * x = x * N q := by
  apply cliffordToM2_injective
  rw [map_mul, map_mul, toM2_N, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_smul, Matrix.mul_one]

theorem N_mem_center (q : Cl3) : N q ∈ Subalgebra.center ℝ Cl3 :=
  Subalgebra.mem_center_iff.mpr fun b => (N_central q b).symm

/-- 2b. `N q` lies in `span{1, I}`, with coordinates `Re det`, `Im det`. -/
theorem N_eq (q : Cl3) :
    N q = algebraMap ℝ Cl3 ((cliffordEquiv q).det.re) + ((cliffordEquiv q).det.im) • Ips := by
  apply cliffordToM2_injective
  rw [toM2_N, map_add, map_smul, AlgHom.commutes, toM2_Ips, cliffordEquiv_apply]
  ext i j
  fin_cases i <;> fin_cases j <;> apply Complex.ext <;>
    simp [Matrix.algebraMap_matrix_apply]

/-- 2b. `N q ∈ span{1, I}` stated without matrices. -/
theorem N_mem_span (q : Cl3) : ∃ r s : ℝ, N q = algebraMap ℝ Cl3 r + s • Ips :=
  ⟨_, _, N_eq q⟩

/-- 2c. Paravector norm. -/
theorem N_pre (t x y z : ℝ) :
    N (pre t x y z) = algebraMap ℝ Cl3 (t ^ 2 - x ^ 2 - y ^ 2 - z ^ 2) := by
  apply cliffordToM2_injective
  rw [toM2_N, F_pre, det_herm, AlgHom.commutes]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.algebraMap_matrix_apply]

theorem N_Ips : N Ips = -1 := by
  apply cliffordToM2_injective
  rw [toM2_N, toM2_Ips, map_neg, map_one]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

/-- 2d. `N (I X) = - N X`. -/
theorem N_Ips_mul (X : Cl3) : N (Ips * X) = -N X := by
  rw [N_mul, N_Ips, neg_one_mul]

theorem toM2_biv (a b c : ℝ) : cliffordToM2 (biv a b c) = I • herm 0 a b c := by
  simp only [biv, map_add, map_smul, map_mul, F_γ0, F_γ1, F_γ2]
  rw [herm_eq]
  ext i j
  fin_cases i <;> fin_cases j <;> apply Complex.ext <;>
    simp [σx, σy, σz]

/-- Bivectors are the pseudoscalar times vectors. -/
theorem biv_eq (a b c : ℝ) : biv a b c = Ips * pre 0 a b c := by
  apply cliffordToM2_injective
  rw [toM2_biv, map_mul, toM2_Ips, F_pre, Matrix.smul_mul, Matrix.one_mul]

theorem biv_reverse (a b c : ℝ) : CliffordAlgebra.reverse (biv a b c) = -biv a b c := by
  apply cliffordToM2_injective
  rw [toM2_reverse, map_neg, toM2_biv, I_herm_conjTranspose]

theorem biv_involute (a b c : ℝ) : CliffordAlgebra.involute (biv a b c) = biv a b c := by
  apply cliffordToM2_injective
  rw [toM2_involute, toM2_biv, adj_I_herm, conjTranspose_neg, I_herm_conjTranspose, neg_neg]

theorem biv_cconj (a b c : ℝ) : cconj (biv a b c) = -biv a b c := by
  rw [cconj, biv_involute, biv_reverse]

/-- 2e. On bivectors `N A = - A * A`. -/
theorem N_biv (a b c : ℝ) : N (biv a b c) = -(biv a b c * biv a b c) := by
  rw [N, biv_cconj, mul_neg]

/-- 2e. On bivectors `N A = a² + b² + c²`. -/
theorem N_biv_val (a b c : ℝ) : N (biv a b c) = algebraMap ℝ Cl3 (a ^ 2 + b ^ 2 + c ^ 2) := by
  rw [biv_eq, N_Ips_mul, N_pre, ← map_neg]
  congr 1
  ring

/-! ### 2f. The real Killing form of `Cl3` under the commutator -/

/-- `Cl3 ≃ M₂(ℂ)` as real Lie algebras. -/
def cliffordLieEquiv : Cl3 ≃ₗ⁅ℝ⁆ M2 := AlgEquiv.toLieEquiv cliffordEquiv

/-- Mathlib's `killingForm ℝ` on `Cl3` equals the one on `M₂(ℂ)` (as a real Lie algebra). -/
theorem killing_Cl3 (a b : Cl3) :
    killingForm ℝ Cl3 a b = killingForm ℝ M2 (cliffordToM2 a) (cliffordToM2 b) :=
  (LieAlgebra.killingForm_of_equiv_apply cliffordLieEquiv a b).symm

/-- Value on vectors: positive. -/
theorem killing_Cl3_vector (x y z : ℝ) :
    killingForm ℝ Cl3 (pre 0 x y z) (pre 0 x y z) = 16 * (x ^ 2 + y ^ 2 + z ^ 2) := by
  rw [killing_Cl3, F_pre, killingForm_real_boost]

theorem I_herm_eq_gen (t a b c : ℝ) : I • herm t a b c = gen t (-a) (-b) (-c) := by
  rw [gen_eq_I_smul_herm, neg_neg, neg_neg, neg_neg]

/-- Value on bivectors: negative. -/
theorem killing_Cl3_bivector (a b c : ℝ) :
    killingForm ℝ Cl3 (biv a b c) (biv a b c) = -16 * (a ^ 2 + b ^ 2 + c ^ 2) := by
  rw [killing_Cl3, toM2_biv, I_herm_eq_gen, killingForm_real_rotation]
  ring

/-- Vectors and bivectors are Killing-orthogonal. -/
theorem killing_Cl3_orthogonal (a b c x y z : ℝ) :
    killingForm ℝ Cl3 (biv a b c) (pre 0 x y z) = 0 := by
  rw [killing_Cl3, toM2_biv, I_herm_eq_gen, F_pre, killingForm_real_orthogonal]

theorem killingForm_real_gen (t a b c : ℝ) :
    killingForm ℝ M2 (gen t a b c) (gen t a b c) = -16 * (a ^ 2 + b ^ 2 + c ^ 2) := by
  have htr : (gen t a b c).trace = 2 * (t : ℂ) * I := by
    simp [gen, e1, e2, e3, σx, σy, σz, trace_fin_two]
    ring
  have key : 4 * ((gen t a b c) * (gen t a b c)).trace
      - 2 * (gen t a b c).trace * (gen t a b c).trace
      = ((-8 * (a ^ 2 + b ^ 2 + c ^ 2) : ℝ) : ℂ) := by
    rw [trace_sq_fin_two, htr, det_gen]
    push_cast
    linear_combination (8 * (t : ℂ) ^ 2) * I_sq
  rw [killingForm_real, killingForm_gl2, key, Complex.ofReal_re]
  ring

/-- The centre drops out: `κ(I X, I X)` for a paravector `X` sees only the vector part. -/
theorem killing_Cl3_Ips_pre (t x y z : ℝ) :
    killingForm ℝ Cl3 (Ips * pre t x y z) (Ips * pre t x y z)
      = -16 * (x ^ 2 + y ^ 2 + z ^ 2) := by
  rw [killing_Cl3, map_mul, toM2_Ips, F_pre, Matrix.smul_mul, Matrix.one_mul, I_herm_eq_gen,
    killingForm_real_gen]
  ring

theorem gradeZero_pre (t x y z : ℝ) : gradeZero (pre t x y z) = algebraMap ℝ Cl3 t := by
  rw [gradeZero_eq, cliffordEquiv_apply, F_pre, trace_herm, Complex.ofReal_re]
  congr 1
  ring

/-- 2g. Three-way relation with `κ` = Mathlib's real Killing form of the 8-dimensional real Lie
algebra `Cl3`.  The constant is `1/16`, not `1/8`. -/
theorem threeway_Cl3_real (t x y z : ℝ) :
    N (pre t x y z) = gradeZero (pre t x y z) * gradeZero (pre t x y z)
      + algebraMap ℝ Cl3
          ((1 / 16) * killingForm ℝ Cl3 (Ips * pre t x y z) (Ips * pre t x y z)) := by
  rw [N_pre, gradeZero_pre, killing_Cl3_Ips_pre, ← map_mul, ← map_add]
  congr 1
  ring

/-! ### 2g. The bivector Lie algebra and its own Killing form (constant `1/8`) -/

/-- The bivectors as a real Lie subalgebra of `Cl3`: reverse-odd and involute-even. -/
def spin3 : LieSubalgebra ℝ Cl3 :=
  { carrier := {X | CliffordAlgebra.reverse X = -X ∧ CliffordAlgebra.involute X = X}
    add_mem' := by
      rintro X Y ⟨h1, h2⟩ ⟨h3, h4⟩
      exact ⟨by rw [map_add, h1, h3, neg_add], by rw [map_add, h2, h4]⟩
    zero_mem' := ⟨by simp, by simp⟩
    smul_mem' := by
      rintro c X ⟨h1, h2⟩
      exact ⟨by rw [map_smul, h1, smul_neg], by rw [map_smul, h2]⟩
    lie_mem' := by
      rintro X Y ⟨h1, h2⟩ ⟨h3, h4⟩
      refine ⟨?_, ?_⟩
      · rw [Ring.lie_def, map_sub, CliffordAlgebra.reverse.map_mul,
          CliffordAlgebra.reverse.map_mul, h1, h3]
        simp
      · rw [Ring.lie_def, map_sub, map_mul, map_mul, h2, h4] }

theorem mem_spin3 {X : Cl3} :
    X ∈ spin3 ↔ CliffordAlgebra.reverse X = -X ∧ CliffordAlgebra.involute X = X := Iff.rfl

theorem biv_mem (a b c : ℝ) : biv a b c ∈ spin3 :=
  mem_spin3.mpr ⟨biv_reverse a b c, biv_involute a b c⟩

theorem su2_reconstruct (M : M2) (h1 : Mᴴ = -M) (h2 : (adjugate M)ᴴ = M) :
    I • herm 0 ((M 1 0).im) (-(M 1 0).re) ((M 0 0).im) = M := by
  have hadj : adjugate M = -M := by
    have := congrArg conjTranspose h2
    rw [conjTranspose_conjTranspose, h1] at this
    exact this
  have h11 : M 1 1 = -M 0 0 := by
    have := congrFun (congrFun hadj 0) 0
    simpa [adjugate_fin_two] using this
  have h00 : conj (M 0 0) = -M 0 0 := by
    simpa [conjTranspose_apply] using congrFun (congrFun h1 0) 0
  have h01 : conj (M 1 0) = -M 0 1 := by
    simpa [conjTranspose_apply] using congrFun (congrFun h1 0) 1
  have r00 : (M 0 0).re = 0 := by
    have := congrArg Complex.re h00
    simp at this
    linarith
  have r01 : (M 0 1).re = -(M 1 0).re := by
    have := congrArg Complex.re h01
    simp at this
    linarith
  have i01 : (M 0 1).im = (M 1 0).im := by
    have := congrArg Complex.im h01
    simp at this
    linarith
  rw [I_herm_explicit]
  ext i j
  fin_cases i <;> fin_cases j <;> apply Complex.ext <;> simp [h11, r00, r01, i01]

/-- Real coordinates on the bivectors. -/
def bcoords : spin3 ≃ₗ[ℝ] (Fin 3 → ℝ) where
  toFun X := ![(cliffordToM2 (X : Cl3) 1 0).im, -(cliffordToM2 (X : Cl3) 1 0).re,
    (cliffordToM2 (X : Cl3) 0 0).im]
  map_add' X Y := by
    ext i
    fin_cases i <;> simp <;> try ring
  map_smul' c X := by
    ext i
    fin_cases i <;> simp <;> try ring
  invFun y := ⟨biv (y 0) (y 1) (y 2), biv_mem _ _ _⟩
  left_inv := by
    rintro ⟨X, hX⟩
    apply Subtype.ext
    apply cliffordToM2_injective
    show cliffordToM2 (biv ((cliffordToM2 X 1 0).im) (-(cliffordToM2 X 1 0).re)
      ((cliffordToM2 X 0 0).im)) = cliffordToM2 X
    rw [toM2_biv]
    exact su2_reconstruct _ (by rw [← toM2_reverse, hX.1, map_neg])
      (by rw [← toM2_involute, hX.2])
  right_inv y := by
    ext i
    fin_cases i <;> simp [toM2_biv, herm_eq]

theorem bcoords_apply (X : spin3) :
    bcoords X = ![(cliffordToM2 (X : Cl3) 1 0).im, -(cliffordToM2 (X : Cl3) 1 0).re,
      (cliffordToM2 (X : Cl3) 0 0).im] := rfl

theorem bcoords_symm_coe (y : Fin 3 → ℝ) :
    ((bcoords.symm y : spin3) : Cl3) = biv (y 0) (y 1) (y 2) := rfl

/-- Basis of the bivectors. -/
def bbas : Module.Basis (Fin 3) ℝ spin3 := Module.Basis.ofEquivFun bcoords

/-- Killing form of the bivector Lie algebra from `tr (ad ∘ ad)` in a basis: `κ = -8 y·z`. -/
theorem killing_spin3 (y z : Fin 3 → ℝ) :
    killingForm ℝ spin3 (bcoords.symm y) (bcoords.symm z)
      = -8 * (y 0 * z 0 + y 1 * z 1 + y 2 * z 2) := by
  rw [killingForm_apply_apply, LinearMap.trace_eq_matrix_trace ℝ bbas]
  simp [Matrix.trace, Fin.sum_univ_three, LinearMap.toMatrix_apply, bbas, bcoords_apply,
    bcoords_symm_coe, Ring.lie_def, toM2_biv, I_herm_explicit, Pi.single_apply]
  ring

theorem killing_spin3_biv (a b c : ℝ) :
    killingForm ℝ spin3 ⟨biv a b c, biv_mem a b c⟩ ⟨biv a b c, biv_mem a b c⟩
      = -8 * (a ^ 2 + b ^ 2 + c ^ 2) := by
  have h := killing_spin3 ![a, b, c] ![a, b, c]
  have e : bcoords.symm ![a, b, c] = ⟨biv a b c, biv_mem a b c⟩ := rfl
  rw [e] at h
  rw [h]
  simp
  ring

/-- On bivectors: `N A = -(1/8) κ(A, A)` with `κ` the Killing form of the bivector algebra. -/
theorem N_biv_killing (a b c : ℝ) :
    N (biv a b c) = algebraMap ℝ Cl3
      (-(1 / 8) * killingForm ℝ spin3 ⟨biv a b c, biv_mem a b c⟩ ⟨biv a b c, biv_mem a b c⟩) := by
  rw [N_biv_val, killing_spin3_biv]
  congr 1
  ring

theorem Ips_vector_mem (x y z : ℝ) : Ips * pre 0 x y z ∈ spin3 := by
  rw [← biv_eq]
  exact biv_mem x y z

/-- 2g. Three-way relation with `κ` = Killing form of the 3-dimensional bivector Lie algebra,
evaluated on `I X₁`, `X₁` the vector part of the paravector `X`.  Constant `1/8`. -/
theorem threeway_spin3 (t x y z : ℝ) :
    N (pre t x y z) = gradeZero (pre t x y z) * gradeZero (pre t x y z)
      + algebraMap ℝ Cl3
          ((1 / 8) * killingForm ℝ spin3 ⟨Ips * pre 0 x y z, Ips_vector_mem x y z⟩
            ⟨Ips * pre 0 x y z, Ips_vector_mem x y z⟩) := by
  have e : (⟨Ips * pre 0 x y z, Ips_vector_mem x y z⟩ : spin3)
      = ⟨biv x y z, biv_mem x y z⟩ := Subtype.ext (biv_eq x y z).symm
  rw [e, killing_spin3_biv, N_pre, gradeZero_pre, ← map_mul, ← map_add]
  congr 1
  ring

/-! ## 3. PhysLean cross-check -/

/-- The images of the Clifford generators are PhysLean's Pauli matrices. -/
theorem toM2_γ0_physlib : cliffordToM2 (γ 0) = PauliMatrix.pauliMatrix (Sum.inr 0) := F_γ0
theorem toM2_γ1_physlib : cliffordToM2 (γ 1) = PauliMatrix.pauliMatrix (Sum.inr 1) := F_γ1
theorem toM2_γ2_physlib : cliffordToM2 (γ 2) = PauliMatrix.pauliMatrix (Sum.inr 2) := F_γ2

theorem cliffordEquiv_γ_physlib (i : Fin 3) :
    cliffordEquiv (γ i) = PauliMatrix.pauliMatrix (Sum.inr i) := by
  fin_cases i
  · exact toM2_γ0_physlib
  · exact toM2_γ1_physlib
  · exact toM2_γ2_physlib

end

end CalChecks

#print axioms CalChecks.dict_reverse
#print axioms CalChecks.dict_involute
#print axioms CalChecks.dict_cconj
#print axioms CalChecks.dict_N
#print axioms CalChecks.gradeZero_eq
#print axioms CalChecks.dict_scalar
#print axioms CalChecks.dict_par
#print axioms CalChecks.dict_apar
#print axioms CalChecks.grade_sum
#print axioms CalChecks.par_eq
#print axioms CalChecks.apar_eq
#print axioms CalChecks.dict_Ips
#print axioms CalChecks.Ips_central
#print axioms CalChecks.Ips_mem_center
#print axioms CalChecks.Ips_sq
#print axioms CalChecks.cl_reverse
#print axioms CalChecks.cl_par
#print axioms CalChecks.cl_equivariant
#print axioms CalChecks.N_mul
#print axioms CalChecks.N_central
#print axioms CalChecks.N_mem_center
#print axioms CalChecks.N_eq
#print axioms CalChecks.N_mem_span
#print axioms CalChecks.N_pre
#print axioms CalChecks.N_Ips
#print axioms CalChecks.N_Ips_mul
#print axioms CalChecks.biv_eq
#print axioms CalChecks.biv_reverse
#print axioms CalChecks.biv_involute
#print axioms CalChecks.biv_cconj
#print axioms CalChecks.N_biv
#print axioms CalChecks.N_biv_val
#print axioms CalChecks.killing_Cl3
#print axioms CalChecks.killing_Cl3_vector
#print axioms CalChecks.killing_Cl3_bivector
#print axioms CalChecks.killing_Cl3_orthogonal
#print axioms CalChecks.killing_Cl3_Ips_pre
#print axioms CalChecks.gradeZero_pre
#print axioms CalChecks.threeway_Cl3_real
#print axioms CalChecks.biv_mem
#print axioms CalChecks.killing_spin3
#print axioms CalChecks.killing_spin3_biv
#print axioms CalChecks.N_biv_killing
#print axioms CalChecks.threeway_spin3
#print axioms CalChecks.toM2_γ0_physlib
#print axioms CalChecks.toM2_γ1_physlib
#print axioms CalChecks.toM2_γ2_physlib
#print axioms CalChecks.cliffordEquiv_γ_physlib
