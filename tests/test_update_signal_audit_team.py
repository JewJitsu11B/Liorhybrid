"""
Tests for the Update-Signal Audit Team.

Mirrors the pattern of tests/test_mem_management_audit_team.py and
tests/test_telemetry_pipeline_audit_team.py.
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

from utils.update_signal_audit_team import (
    UpdateSignalReport,
    UpdateFinding,
    WiringCheck,
    ScribeLog,
    CoordinatorAgent,
    ScalingAnalystAgent,
    SignalChainAgent,
    NudgeScaleAgent,
    PrintDiagnosticsAgent,
    MoraleAgent,
    ScribeAgent,
    UpdateSignalAuditTeam,
    _check_pipeline_wiring,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)


# ---------------------------------------------------------------------------
# UpdateSignalAuditTeam — integration
# ---------------------------------------------------------------------------

def test_audit_team_returns_report():
    """Full team run returns a populated UpdateSignalReport."""
    team = UpdateSignalAuditTeam()
    report = team.run()

    assert isinstance(report, UpdateSignalReport)
    assert report.approval_status == "AWAITING APPROVAL TO EXECUTE"
    assert len(report.findings) >= 10
    assert len(report.wiring_checks) >= 4
    assert report.morale_notes
    assert report.scribe_log is not None


def test_audit_team_coordinator_scope_mentions_key_topics():
    """Coordinator scope mentions the two primary audit questions."""
    coord = CoordinatorAgent()
    scope = coord.scope()
    assert "eta_update" in scope or "eta" in scope
    assert "beta_nudge" in scope or "beta" in scope
    assert "lior_diff" in scope.lower() or "LIoR" in scope
    assert "attenuat" in scope.lower() or "signal" in scope.lower()


def test_audit_team_coordinator_task_queue_roles():
    """Task queue covers all specialist roles."""
    coord = CoordinatorAgent()
    roles = {role for role, _ in coord.task_queue()}
    assert "ScalingAnalyst" in roles
    assert "SignalChain" in roles
    assert "NudgeScale" in roles
    assert "PrintDiagnostics" in roles
    assert "Morale" in roles
    assert "Scribe" in roles


# ---------------------------------------------------------------------------
# ScalingAnalystAgent
# ---------------------------------------------------------------------------

def test_scaling_analyst_produces_scale_findings():
    """ScalingAnalystAgent produces SCALE-* findings."""
    findings = ScalingAnalystAgent().audit()
    ids = {f.finding_id for f in findings}
    assert "SCALE-1" in ids  # eta_update default
    assert "SCALE-2" in ids  # beta_nudge default
    assert "SCALE-3" in ids  # effective eta and expected delta
    assert "SCALE-4" in ids  # division wired in apply_manual_update


def test_scaling_analyst_eta_division_wired():
    """SCALE-4 must pass — eta = eta_update / beta_nudge is in source."""
    findings = {f.finding_id: f for f in ScalingAnalystAgent().audit()}
    assert findings["SCALE-4"].passed, (
        "eta = cfg.eta_update / cfg.beta_nudge not found in apply_manual_update"
    )


def test_scaling_analyst_defaults_are_sensible():
    """SCALE-1 and SCALE-2 pass with current default parameter values."""
    findings = {f.finding_id: f for f in ScalingAnalystAgent().audit()}
    assert findings["SCALE-1"].passed, "eta_update default out of range"
    assert findings["SCALE-2"].passed, "beta_nudge default out of range"


# ---------------------------------------------------------------------------
# SignalChainAgent
# ---------------------------------------------------------------------------

def test_signal_chain_produces_sig_findings():
    """SignalChainAgent produces SIG-* findings."""
    findings = SignalChainAgent().audit()
    ids = {f.finding_id for f in findings}
    assert "SIG-1" in ids  # lior_diff sign
    assert "SIG-2" in ids  # NaN/Inf guard
    assert "SIG-3" in ids  # velocity forwarded
    assert "SIG-4" in ids  # apply_manual_update called
    assert "SIG-5" in ids  # no artificial zero-clamp


def test_signal_chain_lior_diff_sign_correct():
    """SIG-1 passes — lior_diff computed as nudged - free."""
    findings = {f.finding_id: f for f in SignalChainAgent().audit()}
    assert findings["SIG-1"].passed, "lior_diff sign may be wrong"


def test_signal_chain_nan_guard_present():
    """SIG-2 passes — NaN/Inf guard present in apply_manual_update."""
    findings = {f.finding_id: f for f in SignalChainAgent().audit()}
    assert findings["SIG-2"].passed, "NaN/Inf guard missing"


def test_signal_chain_apply_manual_update_called():
    """SIG-4 passes — apply_manual_update is called in run_two_phase_and_update."""
    findings = {f.finding_id: f for f in SignalChainAgent().audit()}
    assert findings["SIG-4"].passed, "apply_manual_update not wired"


def test_signal_chain_no_zero_clamp():
    """SIG-5 passes — lior_diff_val is not artificially zeroed."""
    findings = {f.finding_id: f for f in SignalChainAgent().audit()}
    assert findings["SIG-5"].passed, "lior_diff_val is being zeroed before delta_g"


# ---------------------------------------------------------------------------
# NudgeScaleAgent
# ---------------------------------------------------------------------------

def test_nudge_scale_produces_nudge_findings():
    """NudgeScaleAgent produces NUDGE-* findings."""
    findings = NudgeScaleAgent().audit()
    ids = {f.finding_id for f in findings}
    assert "NUDGE-1" in ids  # nudge_scale default
    assert "NUDGE-2" in ids  # nudge_mode default
    assert "NUDGE-3" in ids  # nudge_every_windows
    assert "NUDGE-4" in ids  # lior_diff scale analysis


def test_nudge_scale_mode_is_target_embedding():
    """NUDGE-2 passes — nudge_mode default is 'target_embedding'."""
    findings = {f.finding_id: f for f in NudgeScaleAgent().audit()}
    assert findings["NUDGE-2"].passed, "nudge_mode default is not 'target_embedding'"


def test_nudge_scale_fires_every_window():
    """NUDGE-3 passes — nudge_every_windows default is 1."""
    findings = {f.finding_id: f for f in NudgeScaleAgent().audit()}
    assert findings["NUDGE-3"].passed, "nudge_every_windows default is not 1"


def test_nudge_4_answers_tiny_updates_question():
    """NUDGE-4 provides a direct answer about whether tiny updates are a problem."""
    findings = {f.finding_id: f for f in NudgeScaleAgent().audit()}
    f4 = findings["NUDGE-4"]
    # Should always pass (informational)
    assert f4.passed
    # Evidence must address the question directly
    ev = f4.evidence.lower()
    assert "not" in ev or "no" in ev or "slow" in ev  # "not a bug" / "not broken" / "slow"
    assert "eta_update" in f4.recommendation or "nudge_scale" in f4.recommendation


# ---------------------------------------------------------------------------
# PrintDiagnosticsAgent
# ---------------------------------------------------------------------------

def test_print_diagnostics_produces_print_findings():
    """PrintDiagnosticsAgent produces PRINT-* findings."""
    findings = PrintDiagnosticsAgent().audit()
    ids = {f.finding_id for f in findings}
    assert "PRINT-1" in ids  # |Δ| format
    assert "PRINT-2" in ids  # |Δθ| format
    assert "PRINT-3" in ids  # LIoR diff format


def test_print_diagnostics_delta_format_is_scientific():
    """PRINT-1 passes — |Δ| uses :.4e format after the fix."""
    findings = {f.finding_id: f for f in PrintDiagnosticsAgent().audit()}
    assert findings["PRINT-1"].passed, (
        "|Δ| still uses :.6f format — tiny updates will appear as 0.000000"
    )


def test_print_diagnostics_theta_format_is_scientific():
    """PRINT-2 passes — |Δθ| uses :.4e format after the fix."""
    findings = {f.finding_id: f for f in PrintDiagnosticsAgent().audit()}
    assert findings["PRINT-2"].passed, (
        "|Δθ| still uses :.6f format — tiny rotor updates will appear as 0.000000"
    )


# ---------------------------------------------------------------------------
# Pipeline wiring checks
# ---------------------------------------------------------------------------

def test_wiring_checks_cover_key_features():
    """_check_pipeline_wiring covers apply_manual_update and print formats."""
    checks = _check_pipeline_wiring()
    features = {c.feature for c in checks}
    assert any("apply_manual_update" in f for f in features)
    assert any("Δ" in f or "delta" in f.lower() for f in features)
    assert any("nudge_mode" in f or "nudge" in f.lower() for f in features)


def test_wiring_apply_manual_update_is_wired():
    """apply_manual_update must be wired into run_two_phase_and_update."""
    checks = {c.feature: c for c in _check_pipeline_wiring()}
    target = next(
        (c for f, c in checks.items() if "apply_manual_update" in f), None
    )
    assert target is not None, "No wiring check for apply_manual_update"
    assert target.wired, "apply_manual_update is not wired"


def test_wiring_print_format_delta_is_scientific():
    """|Δ| print wiring check passes after the format fix."""
    checks = {c.feature: c for c in _check_pipeline_wiring()}
    target = next((c for f, c in checks.items() if "|Δ|" in f), None)
    assert target is not None, "No wiring check for |Δ| format"
    assert target.wired, "|Δ| print format is not using :.4e"


def test_wiring_startup_summary_is_present():
    """Startup param count + window structure summary is wired."""
    checks = {c.feature: c for c in _check_pipeline_wiring()}
    target = next((c for f, c in checks.items() if "param" in f.lower() or "steps_per_window" in f), None)
    assert target is not None, "No wiring check for startup summary"
    assert target.wired, "Startup param/window summary not wired into trainer2_entrypoint"


def test_wiring_telemetry_glossary_is_present():
    """Telemetry glossary wiring check passes."""
    checks = {c.feature: c for c in _check_pipeline_wiring()}
    target = next((c for f, c in checks.items() if "glossary" in f.lower()), None)
    assert target is not None, "No wiring check for telemetry glossary"
    assert target.wired, "Telemetry glossary not printed at startup"


def test_wiring_gpu_ram_per_window():
    """Per-window GPU RAM reporting is wired into the window loop."""
    checks = {c.feature: c for c in _check_pipeline_wiring()}
    target = next((c for f, c in checks.items() if "GPU RAM" in f or "gpu" in f.lower()), None)
    assert target is not None, "No wiring check for GPU RAM reporting"
    assert target.wired, "GPU RAM not reported per window"


# ---------------------------------------------------------------------------
# ScribeAgent
# ---------------------------------------------------------------------------

def test_scribe_consolidates_all_findings():
    """ScribeAgent consolidates findings and wiring checks into a ScribeLog."""
    team = UpdateSignalAuditTeam()
    report = team.run()
    log = report.scribe_log
    assert isinstance(log, ScribeLog)
    assert len(log.findings) == len(report.findings)
    assert log.summary != ""
    assert log.action_items


def test_scribe_action_items_nonempty():
    """ScribeLog action_items is always nonempty (fallback message if all pass)."""
    team = UpdateSignalAuditTeam()
    report = team.run()
    assert len(report.scribe_log.action_items) >= 1


# ---------------------------------------------------------------------------
# MoraleAgent
# ---------------------------------------------------------------------------

def test_morale_notes_mention_loss_and_perplexity():
    """MoraleAgent notes mention loss/perplexity context."""
    team = UpdateSignalAuditTeam()
    report = team.run()
    combined = " ".join(report.morale_notes).lower()
    assert "loss" in combined or "perplexity" in combined or "metric" in combined


def test_morale_notes_answer_tiny_updates_question():
    """MoraleAgent gives a direct answer to 'are tiny updates a problem?'."""
    team = UpdateSignalAuditTeam()
    report = team.run()
    combined = " ".join(report.morale_notes)
    assert "tiny" in combined.lower() or "small" in combined.lower() or "|Δ|" in combined
    # Must give a recommendation (eta_update or nudge_scale)
    assert "eta_update" in combined or "nudge_scale" in combined
