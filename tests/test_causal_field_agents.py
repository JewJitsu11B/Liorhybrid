"""
Tests for the causal field agent team.

Validates:
  - CoordinatorScribeAgent scans repo and discovers causal field files
  - AbstractAlgebraAgent detects sign errors and alpha-parameter issues
  - GeometryAgent detects missing parallel propagators / fiber holonomy
  - ValidationAgent runs numerical checks on a live CausalFieldLayer
  - Transport operator classes have correct flat-limit behaviour
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import math
import torch
import pytest
from pathlib import Path

# ── Locate repo root so agents can be constructed with an absolute path ──────
REPO_ROOT = str(Path(__file__).resolve().parent.parent)


# =============================================================================
# Helpers
# =============================================================================

def _make_agent_location(path: str, role: str = "core implementation"):
    from utils.causal_field_agents import CausalFieldLocation
    return CausalFieldLocation(path=path, role=role, relevant_lines=[])


# =============================================================================
# CoordinatorScribeAgent – repo scanning
# =============================================================================

class TestCoordinatorScribeAgent:
    """Tests that the coordinator correctly discovers causal field files."""

    def test_scan_finds_causal_field_py(self):
        from utils.causal_field_agents import CoordinatorScribeAgent
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        locations = coord._scan_repo()
        paths = [loc.path for loc in locations]
        # The core implementation file must always be found
        assert any("causal_field" in p for p in paths), (
            f"Expected 'causal_field' in discovered paths, got: {paths}"
        )

    def test_scan_classifies_core_implementation(self):
        from utils.causal_field_agents import CoordinatorScribeAgent
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        locations = coord._scan_repo()
        core = [loc for loc in locations if "core implementation" in loc.role]
        assert core, "At least one file should be classified as 'core implementation'"

    def test_run_produces_report(self):
        """Full coordinator run returns a CausalFieldReport."""
        from utils.causal_field_agents import CoordinatorScribeAgent, CausalFieldReport
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        assert isinstance(report, CausalFieldReport)
        assert report.locations
        assert report.findings
        assert report.action_log
        assert report.transport_plan

    def test_action_log_contains_summary(self):
        from utils.causal_field_agents import CoordinatorScribeAgent
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        log_text = "\n".join(report.action_log)
        assert "Summary" in log_text
        assert "Transport Operator Plan" in log_text

    def test_transport_plan_covers_four_phases(self):
        from utils.causal_field_agents import CoordinatorScribeAgent
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        plan_text = "\n".join(report.transport_plan)
        for phase in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
            assert phase in plan_text, f"Transport plan missing {phase}"


# =============================================================================
# AbstractAlgebraAgent – source-text analysis
# =============================================================================

class TestAbstractAlgebraAgent:
    """Tests the algebraic structure checks on source text fragments."""

    def _agent(self):
        from utils.causal_field_agents import AbstractAlgebraAgent
        return AbstractAlgebraAgent()

    def _loc(self, path="models/causal_field.py"):
        return _make_agent_location(path)

    def test_detects_correct_minus_sign(self):
        agent = self._agent()
        text = "T_flat = alpha * J_flat - (1 - alpha) * memory_out"
        findings = agent._check_sign(text, "models/causal_field.py")
        sign_f = [f for f in findings if "sign" in f.check.lower()][0]
        assert sign_f.passed

    def test_detects_wrong_plus_sign(self):
        agent = self._agent()
        text = "T_flat = alpha * J_flat + (1 - alpha) * memory_out"
        findings = agent._check_sign(text, "models/causal_field.py")
        sign_f = [f for f in findings if "sign" in f.check.lower()][0]
        assert not sign_f.passed
        assert sign_f.severity == "critical"

    def test_detects_missing_alpha_param(self):
        agent = self._agent()
        text = "alpha = self.memory.kernel.weights[0]"
        findings = agent._check_alpha_parameter(text, "models/causal_field.py")
        alpha_f = findings[0]
        assert not alpha_f.passed
        assert alpha_f.severity == "major"

    def test_detects_dedicated_alpha_param(self):
        agent = self._agent()
        text = "self.alpha = nn.Parameter(torch.tensor(0.5))"
        findings = agent._check_alpha_parameter(text, "models/causal_field.py")
        alpha_f = findings[0]
        assert alpha_f.passed

    def test_detects_phi_antisymmetry(self):
        agent = self._agent()
        text = "self.Phi.data = self.Phi.data - self.Phi.data.T"
        findings = agent._check_phi_antisymmetry(text, "models/causal_field.py")
        assert findings[0].passed

    def test_detects_j_antisymmetry(self):
        agent = self._agent()
        text = "J_tensor = J_tensor - J_tensor.transpose(-1, -2)"
        findings = agent._check_j_antisymmetry(text, "models/causal_field.py")
        assert findings[0].passed

    def test_audit_on_fixed_causal_field_passes_sign_and_alpha(self):
        from utils.causal_field_agents import AbstractAlgebraAgent, CoordinatorScribeAgent
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        locations = coord._scan_repo()
        source_text = coord._load_source(locations)
        agent = AbstractAlgebraAgent()
        findings = agent.audit(locations, source_text)
        sign_findings = [f for f in findings if "sign" in f.check.lower()]
        alpha_findings = [f for f in findings if "alpha" in f.check.lower()]
        assert sign_findings, "Should have at least one sign check"
        assert all(f.passed for f in sign_findings), (
            f"Sign check(s) failed: {[(f.check, f.details) for f in sign_findings if not f.passed]}"
        )
        assert alpha_findings, "Should have at least one alpha check"
        assert all(f.passed for f in alpha_findings), (
            f"Alpha check(s) failed: {[(f.check, f.details) for f in alpha_findings if not f.passed]}"
        )


# =============================================================================
# GeometryAgent – source-text analysis
# =============================================================================

class TestGeometryAgent:
    """Tests the geometric structure checks on source text fragments."""

    def _agent(self):
        from utils.causal_field_agents import GeometryAgent
        return GeometryAgent()

    def test_detects_parallel_propagator_import(self):
        agent = self._agent()
        text = "from operators.parallel_transport import ParallelPropagator"
        findings = agent._check_parallel_propagators(text, "models/causal_field.py")
        assert findings[0].passed

    def test_detects_missing_parallel_propagator(self):
        agent = self._agent()
        text = "class ParallelTransport(nn.Module): pass"
        # No 'ParallelPropagator' keyword → should flag as missing
        findings = agent._check_parallel_propagators(text, "models/causal_field.py")
        # 'ParallelTransport' is not 'ParallelPropagator' so it should not pass
        assert not findings[0].passed

    def test_detects_fiber_holonomy(self):
        agent = self._agent()
        text = "from operators.parallel_transport import FiberHolonomy"
        findings = agent._check_fiber_holonomy(text, "models/causal_field.py")
        assert findings[0].passed

    def test_detects_causal_support_via_lior(self):
        agent = self._agent()
        text = "memory_out, new_memory = self.memory(transported_flat, memory)"
        findings = agent._check_causal_support(text, "models/causal_field.py")
        assert findings[0].passed

    def test_detects_kernel_normalisation(self):
        agent = self._agent()
        text = "from models.lior_kernel import LiorMemoryState"
        findings = agent._check_kernel_normalisation(text, "models/causal_field.py")
        assert findings[0].passed


# =============================================================================
# ValidationAgent – numerical checks on live CausalFieldLayer
# =============================================================================

class TestValidationAgent:
    """Numerical runtime checks for CausalFieldLayer."""

    def test_alpha_is_parameter(self):
        from utils.causal_field_agents import ValidationAgent
        agent = ValidationAgent()
        findings = agent.audit(d_model=16, d_field=4, d_spinor=4)
        alpha_f = [f for f in findings if "alpha" in f.check.lower()][0]
        assert alpha_f.passed, f"alpha check failed: {alpha_f.details}"

    def test_forward_sign_check(self):
        from utils.causal_field_agents import ValidationAgent
        agent = ValidationAgent()
        findings = agent.audit(d_model=16, d_field=4, d_spinor=4)
        sign_f = [f for f in findings if "sign" in f.check.lower()][0]
        assert sign_f.passed, f"sign check failed: {sign_f.details}"

    def test_phi_antisymmetry_check(self):
        from utils.causal_field_agents import ValidationAgent
        agent = ValidationAgent()
        findings = agent.audit(d_model=16, d_field=4, d_spinor=4)
        phi_f = [f for f in findings if "phi" in f.check.lower()][0]
        assert phi_f.passed, f"Phi antisymmetry check failed: {phi_f.details}"

    def test_output_shape_check(self):
        from utils.causal_field_agents import ValidationAgent
        agent = ValidationAgent()
        findings = agent.audit(d_model=16, d_field=4, d_spinor=4)
        shape_f = [f for f in findings if "shape" in f.check.lower()][0]
        assert shape_f.passed, f"Output shape check failed: {shape_f.details}"


# =============================================================================
# ParallelPropagator – unit tests
# =============================================================================

class TestParallelPropagator:
    """Unit tests for the ParallelPropagator transport operator."""

    def test_flat_limit_is_identity(self):
        from operators.parallel_transport import ParallelPropagator
        prop = ParallelPropagator(d_field=8)
        assert prop.is_flat_limit(), "Fresh ParallelPropagator should be at identity"

    def test_forward_preserves_shape(self):
        from operators.parallel_transport import ParallelPropagator
        prop = ParallelPropagator(d_field=8)
        v = torch.randn(3, 5, 8)
        out = prop(v)
        assert out.shape == v.shape

    def test_identity_transport_is_no_op(self):
        from operators.parallel_transport import ParallelPropagator
        prop = ParallelPropagator(d_field=8)
        v = torch.randn(2, 4, 8)
        out = prop(v)
        assert torch.allclose(out, v, atol=1e-6), "Identity propagator must be a no-op"

    def test_learnable_after_flat_init(self):
        from operators.parallel_transport import ParallelPropagator
        prop = ParallelPropagator(d_field=8)
        assert prop.transport.requires_grad, "transport parameter must require grad"


# =============================================================================
# FiberHolonomy – unit tests
# =============================================================================

class TestFiberHolonomy:
    """Unit tests for the FiberHolonomy transport operator."""

    def test_forward_shape(self):
        from operators.parallel_transport import FiberHolonomy
        hol = FiberHolonomy(d_internal=8)
        field = torch.randn(2, 6, 8)
        out = hol(field)
        assert out.shape == field.shape

    def test_phi_is_antisymmetric(self):
        from operators.parallel_transport import FiberHolonomy
        hol = FiberHolonomy(d_internal=8)
        Phi = hol.antisymmetric_phi
        err = (Phi + Phi.T).abs().max().item()
        assert err < 1e-6, f"Phi is not antisymmetric; max|Phi+Phi.T| = {err:.2e}"

    def test_holonomy_matrix_shape(self):
        from operators.parallel_transport import FiberHolonomy
        hol = FiberHolonomy(d_internal=8)
        H = hol.holonomy_matrix()
        assert H.shape == (8, 8)

    def test_phi_fiber_composition(self):
        """H^a_c = Phi^a_b P^b_c; verify by manual contraction."""
        from operators.parallel_transport import FiberHolonomy
        hol = FiberHolonomy(d_internal=4)
        Phi = hol.antisymmetric_phi
        P = hol.P_fiber
        expected = Phi @ P
        actual = hol.holonomy_matrix()
        assert torch.allclose(expected, actual, atol=1e-6)


# =============================================================================
# Integration – full coordinator run including numerical validation
# =============================================================================

class TestCoordinatorFullRun:
    """Integration test: full coordinator run with numerical validation."""

    def test_full_run_all_checks_pass(self):
        from utils.causal_field_agents import CoordinatorScribeAgent
        coord = CoordinatorScribeAgent(repo_root=REPO_ROOT)
        report = coord.run(
            run_numerical=True,
            d_model=16,
            d_field=4,
            d_spinor=4,
        )
        failures = [f for f in report.findings if not f.passed]
        assert not failures, (
            "Expected all checks to pass after the law fix.\n"
            + "\n".join(
                f"  [{f.severity.upper()}] {f.agent}: {f.check}\n"
                f"    {f.details}"
                for f in failures
            )
        )
