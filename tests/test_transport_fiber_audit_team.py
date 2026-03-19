"""
Tests for the seven-agent Transport & Fiber Bundle Audit Team.

Mirrors the pattern of tests/test_math_validation_team.py.
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

import pytest

from utils.transport_fiber_audit_team import (
    AuditReport,
    CoordinatorAgent,
    PhysicsAgent,
    GeometryAgent,
    CodingAgent,
    ValidationAgent,
    MoraleAgent,
    ScribeAgent,
    TransportFiberAuditTeam,
    _check_pipeline_wiring,
    SEVERITY_INFO,
    SEVERITY_CRITICAL,
)
from utils.pipeline_audit import reset_audit, write_transport_fiber_audit_report


# ---------------------------------------------------------------------------
# TransportFiberAuditTeam — integration
# ---------------------------------------------------------------------------

def test_audit_team_returns_report():
    """Full team run returns a populated AuditReport."""
    team = TransportFiberAuditTeam()
    report = team.run()

    assert isinstance(report, AuditReport)
    assert report.approval_status == "AWAITING APPROVAL TO EXECUTE"
    assert len(report.findings) >= 10
    assert len(report.wiring_checks) >= 4
    assert report.morale_notes
    assert report.scribe_log is not None


def test_audit_team_coordinator_scope():
    """Coordinator scope is non-empty and mentions key operators."""
    coord = CoordinatorAgent()
    scope = coord.scope()
    assert "ParallelTransport" in scope or "transport" in scope.lower()
    assert "fiber bundle" in scope.lower() or "Tetrad" in scope or "tetrad" in scope.lower()
    assert len(coord.task_queue()) == 6


# ---------------------------------------------------------------------------
# Physics agent
# ---------------------------------------------------------------------------

def test_physics_agent_phi_antisymmetry():
    """Physics agent reports on Phi antisymmetry."""
    agent = PhysicsAgent()
    findings = agent.audit()
    phi_check = next(
        (f for f in findings if "antisymmetric" in f.check.lower()), None
    )
    assert phi_check is not None, "Expected Phi antisymmetry check"
    assert phi_check.passed, "Phi should be antisymmetric"


def test_physics_agent_phi_not_in_forward():
    """Physics agent detects that Phi is missing from forward()."""
    agent = PhysicsAgent()
    findings = agent.audit()
    phi_forward = next(
        (f for f in findings if "field equation" in f.check.lower()), None
    )
    assert phi_forward is not None, "Expected Phi-in-forward check"
    assert not phi_forward.passed, "Phi should NOT be wired into forward (known gap)"


# ---------------------------------------------------------------------------
# Geometry agent
# ---------------------------------------------------------------------------

def test_geometry_agent_tetrad_orthonormality():
    """Geometry agent verifies Tetrad orthonormality."""
    agent = GeometryAgent()
    findings = agent.audit()
    ortho = next((f for f in findings if "orthonormality" in f.check.lower()), None)
    assert ortho is not None
    assert ortho.passed, "Tetrad should be orthonormal"


def test_geometry_agent_detects_tetrad_not_wired():
    """Geometry agent detects that kernels/tetrad.Tetrad is not wired into CliffordConnection."""
    agent = GeometryAgent()
    findings = agent.audit()
    wiring = next(
        (f for f in findings if "CliffordConnection" in f.operator and "Tetrad" in f.operator),
        None,
    )
    assert wiring is not None
    assert not wiring.passed, "Shared Tetrad should be detected as NOT wired"


def test_geometry_agent_pi_memory_unused():
    """Geometry agent flags Pi_memory as unused."""
    agent = GeometryAgent()
    findings = agent.audit()
    pi_mem = next((f for f in findings if "Pi_memory" in f.operator), None)
    assert pi_mem is not None
    assert not pi_mem.passed, "Pi_memory should be flagged as unused"


# ---------------------------------------------------------------------------
# Coding agent
# ---------------------------------------------------------------------------

def test_coding_agent_shape_contract():
    """Coding agent verifies CausalFieldLayer output shape."""
    agent = CodingAgent()
    findings = agent.audit()
    shape = next(
        (f for f in findings if "output shape matches input" in f.check.lower()), None
    )
    assert shape is not None
    assert shape.passed, "CausalFieldLayer output shape should match input"


def test_coding_agent_device_safety():
    """Coding agent confirms no forced CPU transfers."""
    agent = CodingAgent()
    findings = agent.audit()
    gpu = next((f for f in findings if ".cpu()" in f.check or "GPU-safe" in f.check), None)
    assert gpu is not None
    assert gpu.passed, "models/causal_field.py should be device-safe (no .cpu()/.numpy())"


# ---------------------------------------------------------------------------
# Validation agent
# ---------------------------------------------------------------------------

def test_validation_agent_pipeline_nan_free():
    """Validation agent confirms full pipeline forward is NaN/Inf-free."""
    agent = ValidationAgent()
    findings = agent.audit()
    nan_check = next(
        (f for f in findings if "NaN/Inf-free" in f.check), None
    )
    assert nan_check is not None
    assert nan_check.passed, "CausalFieldLayer forward should be NaN-free"


def test_validation_agent_transport_finite():
    """Validation agent confirms ParallelTransport output is finite."""
    agent = ValidationAgent()
    findings = agent.audit()
    finite = next(
        (f for f in findings if "ParallelTransport" in f.operator and "finite" in f.check.lower()),
        None,
    )
    assert finite is not None
    assert finite.passed


# ---------------------------------------------------------------------------
# Morale agent
# ---------------------------------------------------------------------------

def test_morale_agent_produces_notes():
    """Morale agent produces non-empty notes."""
    physics = PhysicsAgent()
    geometry = GeometryAgent()
    coding = CodingAgent()
    validation = ValidationAgent()
    all_findings = (
        physics.audit() + geometry.audit() + coding.audit() + validation.audit()
    )
    agent = MoraleAgent()
    notes = agent.audit(all_findings)
    assert len(notes) >= 2
    assert any("AWAITING APPROVAL" in n or "approval" in n.lower() for n in notes)


# ---------------------------------------------------------------------------
# Scribe agent
# ---------------------------------------------------------------------------

def test_scribe_consolidates_findings():
    """Scribe produces a log with summary and action items."""
    physics = PhysicsAgent()
    findings = physics.audit()
    wiring = _check_pipeline_wiring()
    scribe = ScribeAgent()
    log = scribe.consolidate(findings, wiring)

    assert log.summary
    assert log.action_items
    assert len(log.findings) == len(findings)


def test_scribe_action_items_for_failures():
    """Scribe generates action items for every failed finding."""
    agent = GeometryAgent()
    findings = agent.audit()
    wiring = _check_pipeline_wiring()
    scribe = ScribeAgent()
    log = scribe.consolidate(findings, wiring)

    failed = [f for f in findings if not f.passed]
    if failed:
        assert len(log.action_items) >= len(failed)


# ---------------------------------------------------------------------------
# Pipeline wiring checker
# ---------------------------------------------------------------------------

def test_pipeline_wiring_parallel_transport_wired():
    """ParallelTransport is wired into CausalFieldLayer."""
    checks = _check_pipeline_wiring()
    pi = next((c for c in checks if c.operator == "ParallelTransport"), None)
    assert pi is not None
    assert pi.wired, "ParallelTransport should be wired into CausalFieldLayer"


def test_pipeline_wiring_clifford_connection_wired():
    """CliffordConnection is wired into CausalFieldLayer."""
    checks = _check_pipeline_wiring()
    gamma = next((c for c in checks if c.operator == "CliffordConnection"), None)
    assert gamma is not None
    assert gamma.wired, "CliffordConnection should be wired into CausalFieldLayer"


def test_pipeline_wiring_tetrad_not_wired_into_causal_field():
    """kernels/tetrad.Tetrad (fiber bundle) is NOT wired into models/causal_field.py."""
    checks = _check_pipeline_wiring()
    tetrad = next(
        (c for c in checks if "fiber bundle" in c.operator.lower() or "kernels.tetrad" in c.operator),
        None,
    )
    assert tetrad is not None
    assert not tetrad.wired, "kernels.tetrad.Tetrad should NOT be wired yet (known gap)"


def test_pipeline_wiring_tetrad_exported_from_kernels():
    """Tetrad is exported from kernels/__init__.py."""
    checks = _check_pipeline_wiring()
    export = next((c for c in checks if "export" in c.operator.lower()), None)
    assert export is not None
    assert export.wired


# ---------------------------------------------------------------------------
# pipeline_audit integration
# ---------------------------------------------------------------------------

def test_write_transport_fiber_audit_report(monkeypatch, tmp_path):
    """write_transport_fiber_audit_report appends findings to the audit markdown."""
    audit_file = tmp_path / "pipeline_audit.md"
    monkeypatch.setenv("BCF_PIPELINE_AUDIT_PATH", str(audit_file))

    reset_audit()
    write_transport_fiber_audit_report()
    text = audit_file.read_text(encoding="utf-8")

    assert "Transport & Fiber Bundle Audit Team" in text
    assert "transport_fiber_bundle_audit" in text
    assert "AWAITING APPROVAL" in text
    assert "Pipeline Wiring Checks" in text
    assert "Scribe: Action Items" in text
    assert "Morale Notes" in text
