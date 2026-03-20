"""
Tests for telemetry pipeline audit team (lead + specialists + support + checkpointing).
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

from utils.telemetry_pipeline_audit_team import (
    TelemetryAuditReport,
    TelemetryLeadAgent,
    AbstractAlgebraTelemetryAgent,
    DifferentialGeometryTelemetryAgent,
    GRTelemetryAgent,
    CheckpointingAgent,
    TelemetryPipelineAuditTeam,
    _check_pipeline_wiring,
)


def test_audit_team_returns_report():
    team = TelemetryPipelineAuditTeam()
    report = team.run()
    assert isinstance(report, TelemetryAuditReport)
    assert report.approval_status == "AWAITING APPROVAL TO EXECUTE"
    assert report.findings
    assert report.support_checks
    assert report.wiring_checks
    assert report.telemetry_notes
    assert report.scribe_log.summary if hasattr(report.scribe_log, "summary") else True


def test_lead_agent_queue_contains_specialists_and_support_and_checkpointing():
    lead = TelemetryLeadAgent()
    roles = {name for name, _ in lead.task_queue()}
    assert "AbstractAlgebraist" in roles
    assert "AlgebraSupport" in roles
    assert "DiffGeometer" in roles
    assert "DiffGeomSupport" in roles
    assert "GRAgent" in roles
    assert "GRSupport" in roles
    assert "Checkpointing" in roles
    assert "Scribe" in roles


def test_checkpointing_agent_checks_manual_quit_and_end_epoch_and_periodic():
    findings = CheckpointingAgent().audit()
    ids = {f.finding_id for f in findings}
    assert "CKPT-1" in ids  # periodic
    assert "CKPT-2" in ids  # manual quit forced
    assert "CKPT-3" in ids  # end epoch forced
    assert "CKPT-4" in ids  # entropy-softmax compatibility


def test_specialists_produce_telemetry_semantics_findings():
    alg = AbstractAlgebraTelemetryAgent().audit()
    dg = DifferentialGeometryTelemetryAgent().audit()
    gr = GRTelemetryAgent().audit()
    assert any(f.finding_id.startswith("ALG-") for f in alg)
    assert any(f.finding_id.startswith("DG-") for f in dg)
    assert any(f.finding_id.startswith("GR-") for f in gr)


def test_wiring_checks_cover_forced_checkpoint_paths():
    checks = _check_pipeline_wiring()
    by_feature = {c.feature: c for c in checks}
    assert "Forced checkpoint on manual quit" in by_feature
    assert "Forced checkpoint on end of epoch" in by_feature
    assert by_feature["Forced checkpoint on manual quit"].entry_point
    assert by_feature["Forced checkpoint on end of epoch"].entry_point
