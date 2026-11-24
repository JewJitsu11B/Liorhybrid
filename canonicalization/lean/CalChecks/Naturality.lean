/-
The commuting squares, stated in Mathlib's category-theory types.

A group `G` is a category with one object (`SingleObj G`); a `G`-set is a functor
`actionAsFunctor G X : SingleObj G ⥤ Type`; an equivariant map is a natural transformation.

* `closureNat`     : the closure `c(J) = J J†` is a natural transformation
                     (carriers, left multiplication) ⟶ (Hermitian slice, congruence), for `GL(2,ℂ)`.
* `closureAction`  : the same map as a morphism in `Action (Type) GL(2,ℂ)`.
* `bornNat`        : the trace is natural into the constant functor, for `U(2)`.
* `lorentzNat`     : the determinant is natural into the constant functor, for `SL(2,ℂ)`.
* `killingNat`     : `A ↦ κ(A,A)` (Mathlib's `killingForm`) is natural into the constant functor,
                     for `GL(2,ℂ)` acting by `Ad`.

The content of each naturality square is a theorem already in the library
(`born_left_congruence`, `det_congruence_SL`, `killingForm_gl2`); this file only restates them.
-/
import CalChecks.Structure

open Matrix Complex CategoryTheory

namespace CalChecks

attribute [local instance] LieRing.ofAssociativeRing LieAlgebra.ofAssociativeAlgebra

/-- `GL(2,ℂ)` as the units of `M₂(ℂ)`. -/
abbrev GL2 := (M2)ˣ
/-- `SL(2,ℂ)`. -/
abbrev SL2 := Matrix.SpecialLinearGroup (Fin 2) ℂ
/-- `U(2)`. -/
abbrev U2 := Matrix.unitaryGroup (Fin 2) ℂ

/-- Carriers, with `GL(2,ℂ)` acting by left multiplication. -/
structure CarrierL where
  /-- the matrix -/
  val : M2

/-- The Hermitian slice, acted on by congruence `X ↦ G X G†`. -/
structure Herm2 where
  /-- the matrix -/
  val : M2
  /-- it is Hermitian -/
  herm : valᴴ = val

/-- `M₂(ℂ)` with `GL(2,ℂ)` acting by `Ad`: `A ↦ G A G⁻¹`. -/
structure AdM2 where
  /-- the matrix -/
  val : M2

theorem Herm2.ext' {X Y : Herm2} (h : X.val = Y.val) : X = Y := by
  cases X; cases Y; cases h; rfl

theorem congruence_herm (G X : M2) (hX : Xᴴ = X) : (G * X * Gᴴ)ᴴ = G * X * Gᴴ := by
  simp [conjTranspose_mul, hX, Matrix.mul_assoc]

instance : MulAction GL2 CarrierL where
  smul g J := ⟨(g : M2) * J.val⟩
  one_smul J := by cases J; exact congrArg CarrierL.mk (Matrix.one_mul _)
  mul_smul g h J := by cases J; exact congrArg CarrierL.mk (Matrix.mul_assoc _ _ _)

instance : MulAction GL2 Herm2 where
  smul g X := ⟨(g : M2) * X.val * (g : M2)ᴴ, congruence_herm _ _ X.herm⟩
  one_smul X := Herm2.ext' (by show ((1 : GL2) : M2) * X.val * ((1 : GL2) : M2)ᴴ = X.val; simp)
  mul_smul g h X := Herm2.ext' (by
    show ((g * h : GL2) : M2) * X.val * ((g * h : GL2) : M2)ᴴ
      = (g : M2) * ((h : M2) * X.val * (h : M2)ᴴ) * (g : M2)ᴴ
    simp [Units.val_mul, conjTranspose_mul, Matrix.mul_assoc])

instance : MulAction SL2 Herm2 where
  smul g X := ⟨(g : M2) * X.val * (g : M2)ᴴ, congruence_herm _ _ X.herm⟩
  one_smul X := Herm2.ext' (by show ((1 : SL2) : M2) * X.val * ((1 : SL2) : M2)ᴴ = X.val; simp)
  mul_smul g h X := Herm2.ext' (by
    show ((g * h : SL2) : M2) * X.val * ((g * h : SL2) : M2)ᴴ
      = (g : M2) * ((h : M2) * X.val * (h : M2)ᴴ) * (g : M2)ᴴ
    simp [conjTranspose_mul, Matrix.mul_assoc])

instance : MulAction U2 Herm2 where
  smul g X := ⟨(g : M2) * X.val * (g : M2)ᴴ, congruence_herm _ _ X.herm⟩
  one_smul X := Herm2.ext' (by show ((1 : U2) : M2) * X.val * ((1 : U2) : M2)ᴴ = X.val; simp)
  mul_smul g h X := Herm2.ext' (by
    show ((g * h : U2) : M2) * X.val * ((g * h : U2) : M2)ᴴ
      = (g : M2) * ((h : M2) * X.val * (h : M2)ᴴ) * (g : M2)ᴴ
    simp [conjTranspose_mul, Matrix.mul_assoc])

instance : MulAction GL2 AdM2 where
  smul g A := ⟨(g : M2) * A.val * ((g⁻¹ : GL2) : M2)⟩
  one_smul A := by cases A; exact congrArg AdM2.mk (by simp)
  mul_smul g h A := by
    cases A with | mk a =>
    refine congrArg AdM2.mk ?_
    show ((g * h : GL2) : M2) * a * (((g * h)⁻¹ : GL2) : M2)
      = (g : M2) * ((h : M2) * a * ((h⁻¹ : GL2) : M2)) * ((g⁻¹ : GL2) : M2)
    rw [_root_.mul_inv_rev, Units.val_mul, Units.val_mul]
    simp only [Matrix.mul_assoc]

theorem smul_carrier_val (g : GL2) (J : CarrierL) : (g • J).val = (g : M2) * J.val := rfl
theorem smul_herm_val (g : GL2) (X : Herm2) : (g • X).val = (g : M2) * X.val * (g : M2)ᴴ := rfl
theorem smul_herm_val_SL (g : SL2) (X : Herm2) :
    (g • X).val = (g : M2) * X.val * (g : M2)ᴴ := rfl
theorem smul_herm_val_U (g : U2) (X : Herm2) : (g • X).val = (g : M2) * X.val * (g : M2)ᴴ := rfl
theorem smul_ad_val (g : GL2) (A : AdM2) :
    (g • A).val = (g : M2) * A.val * ((g⁻¹ : GL2) : M2) := rfl

/-- The closure `c(J) = J J†`, landing in the Hermitian slice. -/
def closure (J : CarrierL) : Herm2 := ⟨J.val * J.valᴴ, by simp [conjTranspose_mul]⟩

/-- The square `c(GJ) = G c(J) G†`. -/
theorem closure_smul (g : GL2) (J : CarrierL) : closure (g • J) = g • closure J :=
  Herm2.ext' (born_left_congruence (g : M2) J.val)

/-- **The closure is a natural transformation** between the two `GL(2,ℂ)`-sets, seen as functors
out of the one-object category of the group. -/
def closureNat : actionAsFunctor GL2 CarrierL ⟶ actionAsFunctor GL2 Herm2 where
  app _ := ↾closure
  naturality _ _ g := by
    ext J
    exact closure_smul g J

/-- The same square as a morphism in Mathlib's category `Action (Type) GL(2,ℂ)`. -/
def closureAction : Action.ofMulAction GL2 CarrierL ⟶ Action.ofMulAction GL2 Herm2 where
  hom := ↾closure
  comm g := by
    ext J
    exact closure_smul g J

/-! ## The three readouts: natural into the constant functor -/

theorem trace_smul_U (g : U2) (X : Herm2) : (g • X).val.trace = X.val.trace := by
  have h : (g : M2)ᴴ * (g : M2) = 1 := by
    have := Matrix.mem_unitaryGroup_iff'.mp g.2
    simpa [Matrix.star_eq_conjTranspose] using this
  rw [smul_herm_val_U, Matrix.trace_mul_cycle, h, Matrix.one_mul]

theorem det_smul_SL (g : SL2) (X : Herm2) : (g • X).val.det = X.val.det := by
  rw [smul_herm_val_SL]
  exact det_congruence_SL _ _ g.2

theorem killing_smul (g : GL2) (A : AdM2) :
    killingForm ℂ M2 (g • A).val (g • A).val = killingForm ℂ M2 A.val A.val := by
  have h : ((g⁻¹ : GL2) : M2) * (g : M2) = 1 := by simp
  have t1 : ∀ X : M2, ((g : M2) * X * ((g⁻¹ : GL2) : M2)).trace = X.trace := fun X => by
    rw [Matrix.trace_mul_cycle, h, Matrix.one_mul]
  have t2 : (g : M2) * A.val * ((g⁻¹ : GL2) : M2) * ((g : M2) * A.val * ((g⁻¹ : GL2) : M2))
      = (g : M2) * (A.val * A.val) * ((g⁻¹ : GL2) : M2) := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc ((g⁻¹ : GL2) : M2) (g : M2), h, Matrix.one_mul]
  rw [killingForm_gl2, killingForm_gl2, smul_ad_val, t2, t1, t1]

/-- Born: the trace is natural for `U(2)`. -/
def bornNat : actionAsFunctor U2 Herm2 ⟶ (Functor.const (SingleObj U2)).obj ℂ where
  app _ := ↾fun X => X.val.trace
  naturality _ _ g := by
    ext X
    exact trace_smul_U g X

/-- Lorentz: the determinant is natural for `SL(2,ℂ)`. -/
def lorentzNat : actionAsFunctor SL2 Herm2 ⟶ (Functor.const (SingleObj SL2)).obj ℂ where
  app _ := ↾fun X => X.val.det
  naturality _ _ g := by
    ext X
    exact det_smul_SL g X

/-- Killing: `A ↦ κ(A,A)` is natural for `GL(2,ℂ)` acting by `Ad`. -/
noncomputable def killingNat : actionAsFunctor GL2 AdM2 ⟶ (Functor.const (SingleObj GL2)).obj ℂ where
  app _ := ↾fun A => killingForm ℂ M2 A.val A.val
  naturality _ _ g := by
    ext A
    exact killing_smul g A

end CalChecks

section AxiomCheck
#print axioms CalChecks.closureNat
#print axioms CalChecks.closureAction
#print axioms CalChecks.bornNat
#print axioms CalChecks.lorentzNat
#print axioms CalChecks.killingNat
end AxiomCheck
