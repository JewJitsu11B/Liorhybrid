/-
Cross-check against PhysLean (Physlib), and an axiom audit of every theorem in this project.
PhysLean supplies the machine-checked statement SO⁺(1,3) ≅ SL(2,ℂ)/{±1}:
a homomorphism, orthochronous with det 1, surjective onto the restricted group, kernel {±1}.
-/
import Physlib.Relativity.PauliMatrices.Basic
import Physlib.Relativity.SL2C.Basic
import Physlib.Relativity.SL2C.Cover
import CalChecks.SO4

namespace CalChecks

theorem σx_eq_physlib : σx = PauliMatrix.pauliMatrix (Sum.inr 0) := rfl
theorem σy_eq_physlib : σy = PauliMatrix.pauliMatrix (Sum.inr 1) := rfl
theorem σz_eq_physlib : σz = PauliMatrix.pauliMatrix (Sum.inr 2) := rfl

#check @Lorentz.SL2C.toLorentzGroup
#check @Lorentz.SL2C.toLorentzGroup_det_one
#check @Lorentz.SL2C.toLorentzGroup_isOrthochronous
#check @Lorentz.SL2C.toLorentzGroup_eq_one_iff
#check @Lorentz.SL2C.toRestrictedLorentzGroup_surjective
#print axioms Lorentz.SL2C.toLorentzGroup_eq_one_iff
#print axioms Lorentz.SL2C.toRestrictedLorentzGroup_surjective

#print axioms carrier_eq
#print axioms e1_sq
#print axioms e2_sq
#print axioms e3_sq
#print axioms e1_e2
#print axioms e2_e3
#print axioms e3_e1
#print axioms e2_e1
#print axioms det_carrier
#print axioms trace_carrier_mul_conjTranspose
#print axioms born_right_unitary
#print axioms born_left_congruence
#print axioms hs_invariant
#print axioms povm_pairing
#print axioms H_add_A
#print axioms H_isHermitian
#print axioms A_antiHermitian
#print axioms H_idem
#print axioms H_intertwines
#print axioms A_intertwines
#print axioms herm_eq
#print axioms herm_isHermitian
#print axioms det_herm
#print axioms det_congruence
#print axioms det_congruence_SL
#print axioms trace_sq_fin_two
#print axioms purity_unit_trace
#print axioms purity_velocity
#print axioms lagrange_channels
#print axioms normSq_sub_re_sq
#print axioms det_born
#print axioms born_real_channels
#print axioms gen_eq_I_smul_herm
#print axioms det_gen
#print axioms det_I_smul
#print axioms det_su2
#print axioms chsh_local
#print axioms chsh_quantum_bound
#print axioms chsh_quantum_attained
#print axioms uniq_expand
#print axioms uniq_alpha
#print axioms uniq_projector
#print axioms dotσ_eq
#print axioms jointClosure_trace
#print axioms bell_correlation
#print axioms bell_marginal_left
#print axioms adM_apply
#print axioms stdBasis_repr
#print axioms killing_gl2
#print axioms killing_centre
#print axioms det_su2_killing
#print axioms pauliVec_sq
#print axioms cliffordToM2_ι
#print axioms paravector_norm
#print axioms char_half_sq
#print axioms char_vector_sq
#print axioms char_vector_sym
#print axioms char_vector_alt
#print axioms weights_vector_sq
#print axioms weights_weyl_tensor
#print axioms weights_half_sq
#print axioms carrier_left_mul
#print axioms carrier_right_mul
#print axioms Lmat_orthogonal
#print axioms Rmat_orthogonal
#print axioms Lmat_Rmat_comm
#print axioms Lmat_det
#print axioms Rmat_det
#print axioms sumSq_left
#print axioms sumSq_right
#print axioms normSq_left
#print axioms normSq_right
#print axioms killingR_rotation
#print axioms killingR_boost
#print axioms killingR_orthogonal
#print axioms invariant_form_unique
#print axioms frac_integral_exp
#print axioms cal_alpha_two
#print axioms half_turn
#print axioms channel_half_turn
#print axioms sign_character
#print axioms carrier_coordsOf
#print axioms coordsOf_carrier
#print axioms carrier_injective
#print axioms left_action_columns
#print axioms I_smul_e
#print axioms σ_sq
#print axioms det_two_sided
#print axioms born_posSemidef
#print axioms cone_stabilizer
#print axioms rank_invariant
#print axioms born_singular_iff
#print axioms congruence_hermitian
#print axioms exists_herm
#print axioms herm_injective
#print axioms A_eq_I_smul_H
#print axioms exists_gen
#print axioms sl2_decomposition
#print axioms povm_sum
#print axioms povm_sum_survives_iff
#print axioms born_weights_sum
#print axioms born_additive
#print axioms purity_gamma
#print axioms purity_unitary_invariant
#print axioms boost_changes_trace
#print axioms purity_channels
#print axioms vsq_lagrange
#print axioms vsq_bounds
#print axioms vsq_eq_one_iff
#print axioms vsq_real
#print axioms vsq_scale
#print axioms vsq_rotation
#print axioms boltzmann_limit
#print axioms killing_ad_invariant
#print axioms su2_acts_by_rotation
#print axioms Rot_orthogonal
#print axioms Rot_det
#print axioms dotσ_hermitian
#print axioms dotσ_sq
#print axioms dotσ_trace
#print axioms indep_of_trace_zero
#print axioms uniq_alpha_dotσ
#print axioms jointClosure_idem
#print axioms jointClosure_hermitian
#print axioms bell_marginal_right
#print axioms chsh_tsirelson
#print axioms spinor_kernel
#print axioms spinor_kernel_converse
#print axioms ad_eq_adM
#print axioms killingForm_gl2
#print axioms trace_real
#print axioms killingForm_real
#print axioms killingForm_real_rotation
#print axioms killingForm_real_boost
#print axioms killingForm_real_orthogonal
#print axioms cone_transitive
#print axioms cone_orbit_posDef
#print axioms Q3_single
#print axioms γ_sq
#print axioms Q3_polar
#print axioms γ_anti
#print axioms s10
#print axioms s20
#print axioms s21
#print axioms γ_sq'
#print axioms s10'
#print axioms s20'
#print axioms s21'
#print axioms w_mem
#print axioms γ_mul_w_mem
#print axioms γ_mul_mem
#print axioms ι_eq_sum
#print axioms mem_S
#print axioms S_eq_top
#print axioms finrank_Cl3_le
#print axioms finrank_M2
#print axioms F_γ0
#print axioms F_γ1
#print axioms F_γ2
#print axioms F_vol
#print axioms F_pre
#print axioms cliffordToM2_surjective
#print axioms cliffordToM2_injective
#print axioms cliffordEquiv_ι
#print axioms finrank_Cl3
#print axioms trace_herm
#print axioms conj_splits_one_three
#print axioms purity_boost_dependent
#print axioms closure_trace_ne_zero
#print axioms born_weight_nonneg
#print axioms born_kinematic
#print axioms quat_sandwich
#print axioms IsLR.comp
#print axioms reflection_isLR
#print axioms isometry_isLR
#print axioms qc_qv
#print axioms qv_qc
#print axioms qc_add
#print axioms qc_smul
#print axioms qv_add
#print axioms qv_smul
#print axioms Lmat_mulVec
#print axioms Rmat_mulVec
#print axioms Cmat_mulVec
#print axioms Cmat_det
#print axioms Lmat_mul
#print axioms Rmat_mul
#print axioms normSq_eq_dot
#print axioms unit_qc
#print axioms dot_orth
#print axioms isoOf_apply
#print axioms so4_surjective
#print axioms so4_kernel
#print axioms so4_kernel_converse
#print axioms eq_one_of_mul_self
#print axioms normSq_unitary
#print axioms norm_unitary
#print axioms mem_unitary_of_norm
#print axioms norm_qv
#print axioms LR_mem_SO4
#print axioms so4Hom_apply
#print axioms so4Hom_surjective
#print axioms so4Hom_ker
#print axioms so4Hom
#print axioms so4Equiv

end CalChecks
