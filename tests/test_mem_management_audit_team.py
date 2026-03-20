"""
Tests for the eight-agent GPU Memory Management Audit Team.

Mirrors the pattern of tests/test_transport_fiber_audit_team.py.
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

import pytest

from utils.mem_management_audit_team import (
    MemAuditReport,
    MemFinding,
    WiringCheck,
    ScribeLog,
    CoordinatorAgent,
    MemoryProfilerAgent,
    StreamingAuditorAgent,
    CleanerInventoryAgent,
    AllocationPatternAgent,
    OOMPreventionAgent,
    MoraleAgent,
    ScribeAgent,
    MemManagementAuditTeam,
    _check_pipeline_wiring,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)


# ---------------------------------------------------------------------------
# MemManagementAuditTeam — integration
# ---------------------------------------------------------------------------

def test_audit_team_returns_report():
    """Full team run returns a populated MemAuditReport."""
    team = MemManagementAuditTeam()
    report = team.run()

    assert isinstance(report, MemAuditReport)
    assert report.approval_status == "AWAITING APPROVAL TO EXECUTE"
    assert len(report.findings) >= 15
    assert len(report.wiring_checks) >= 4
    assert report.morale_notes
    assert report.scribe_log is not None


def test_audit_team_coordinator_scope():
    """Coordinator scope mentions key OOM-related topics."""
    coord = CoordinatorAgent()
    scope = coord.scope()
    assert "memory" in scope.lower() or "OOM" in scope
    assert "biquaternion" in scope.lower() or "BiQuat" in scope or "LayerNorm" in scope
    assert len(coord.task_queue()) == 7


def test_audit_team_coordinator_task_queue_roles():
    """Task queue covers all seven specialist roles."""
    coord = CoordinatorAgent()
    roles = {role for role, _ in coord.task_queue()}
    assert "MemoryProfiler" in roles
    assert "StreamingAuditor" in roles
    assert "CleanerInventory" in roles
    assert "AllocationPattern" in roles
    assert "OOMPrevention" in roles
    assert "Morale" in roles
    assert "Scribe" in roles


# ---------------------------------------------------------------------------
# MemoryProfilerAgent
# ---------------------------------------------------------------------------

def test_profiler_checks_expandable_segments():
    """MemoryProfilerAgent reports whether expandable_segments is configured."""
    agent = MemoryProfilerAgent()
    findings = agent.audit()
    exp_seg = next(
        (f for f in findings if "expandable_segments" in f.check.lower()), None
    )
    assert exp_seg is not None, "Expected expandable_segments check"
    # In this repo the env var is NOT set at the entry point, so it should fail
    assert not exp_seg.passed
    assert exp_seg.severity == SEVERITY_HIGH


def test_profiler_checks_get_stats_and_memory_info():
    """MemoryProfilerAgent reports that get_stats/get_memory_info exist."""
    agent = MemoryProfilerAgent()
    findings = agent.audit()
    stats_check = next(
        (f for f in findings if "get_stats" in f.check or "get_memory_info" in f.check), None
    )
    assert stats_check is not None
    assert stats_check.passed, "GPUCleanupThread should expose monitoring hooks"


def test_profiler_checks_enable_expandable_segments_fn():
    """MemoryProfilerAgent verifies enable_expandable_segments() helper exists."""
    agent = MemoryProfilerAgent()
    findings = agent.audit()
    helper_check = next(
        (f for f in findings if "enable_expandable_segments" in f.check), None
    )
    assert helper_check is not None
    assert helper_check.passed


# ---------------------------------------------------------------------------
# StreamingAuditorAgent
# ---------------------------------------------------------------------------

def test_streaming_flush_stream_present():
    """StreamingAuditorAgent detects flush_stream side-stream in TelemetryState."""
    agent = StreamingAuditorAgent()
    findings = agent.audit()
    flush = next(
        (f for f in findings if "flush_stream" in f.check.lower() or "side-stream" in f.check.lower()), None
    )
    assert flush is not None, "Expected flush_stream check"
    assert flush.passed, "flush_stream should be present and used"


def test_streaming_non_blocking_present():
    """StreamingAuditorAgent detects non_blocking=True transfers."""
    agent = StreamingAuditorAgent()
    findings = agent.audit()
    nb = next(
        (f for f in findings if "non_blocking" in f.check), None
    )
    assert nb is not None
    assert nb.passed, "non_blocking=True should be used"


def test_streaming_cleanup_not_in_stream_ctx():
    """StreamingAuditorAgent reports empty_cache() is NOT in a CUDA stream context."""
    agent = StreamingAuditorAgent()
    findings = agent.audit()
    cleanup_stream = next(
        (f for f in findings if "empty_cache" in f.check.lower()), None
    )
    assert cleanup_stream is not None
    assert not cleanup_stream.passed, "empty_cache should NOT be stream-scoped currently"


# ---------------------------------------------------------------------------
# CleanerInventoryAgent
# ---------------------------------------------------------------------------

def test_cleaner_gpu_cleanup_thread_exists():
    """CleanerInventoryAgent confirms GPUCleanupThread exists."""
    agent = CleanerInventoryAgent()
    findings = agent.audit()
    thread_check = next(
        (f for f in findings if "GPUCleanupThread" in f.check and "daemon" in f.check.lower()),
        None,
    )
    assert thread_check is not None
    assert thread_check.passed, "GPUCleanupThread should exist in gpu_cleanup.py"


def test_cleaner_standalone_fn_exists():
    """CleanerInventoryAgent confirms cleanup_gpu_memory() exists."""
    agent = CleanerInventoryAgent()
    findings = agent.audit()
    fn_check = next(
        (f for f in findings if "cleanup_gpu_memory" in f.check and "one-shot" in f.check.lower()),
        None,
    )
    assert fn_check is not None
    assert fn_check.passed


def test_cleaner_thread_not_wired_in_trainer2():
    """CleanerInventoryAgent detects that GPUCleanupThread is NOT started in trainer2."""
    agent = CleanerInventoryAgent()
    findings = agent.audit()
    wiring = next(
        (f for f in findings if "trainer2_entrypoint" in f.component), None
    )
    assert wiring is not None
    assert not wiring.passed, "GPUCleanupThread should NOT yet be wired in trainer2"
    assert wiring.severity == SEVERITY_HIGH


def test_cleaner_inline_empty_cache_present():
    """CleanerInventoryAgent detects inline empty_cache() in trainer2."""
    agent = CleanerInventoryAgent()
    findings = agent.audit()
    inline = next(
        (f for f in findings if "2493" in f.component or "run_two_phase" in f.component.lower()),
        None,
    )
    assert inline is not None
    assert inline.passed, "Inline empty_cache() should be present in trainer2"


# ---------------------------------------------------------------------------
# AllocationPatternAgent
# ---------------------------------------------------------------------------

def test_allocation_oom_site_detected():
    """AllocationPatternAgent identifies the OOM trigger site in biquaternion.py."""
    agent = AllocationPatternAgent()
    findings = agent.audit()
    oom_site = next(
        (f for f in findings if "biquaternion" in f.component.lower() and "382" in f.component),
        None,
    )
    assert oom_site is not None, "Expected OOM site check at biquaternion.py:382"
    assert not oom_site.passed, "residual + output should be flagged as OOM pattern"
    assert oom_site.severity == SEVERITY_HIGH


def test_allocation_inplace_ops_present():
    """AllocationPatternAgent confirms in-place ops are used in trainer2."""
    agent = AllocationPatternAgent()
    findings = agent.audit()
    inplace = next(
        (f for f in findings if ".add_()" in f.check or "in-place" in f.check.lower()), None
    )
    assert inplace is not None
    assert inplace.passed, "In-place accumulation ops should be present"


def test_allocation_preallocated_buffer():
    """AllocationPatternAgent confirms pre-allocated GPU metric buffer."""
    agent = AllocationPatternAgent()
    findings = agent.audit()
    prebuf = next(
        (f for f in findings if "pre-allocated" in f.check.lower()), None
    )
    assert prebuf is not None
    assert prebuf.passed


# ---------------------------------------------------------------------------
# OOMPreventionAgent
# ---------------------------------------------------------------------------

def test_oom_prevention_gradient_checkpointing_missing():
    """OOMPreventionAgent detects that gradient checkpointing is missing in geometric_stack."""
    agent = OOMPreventionAgent()
    findings = agent.audit()
    gc_check = next(
        (f for f in findings if "checkpointing" in f.check.lower() or "GeometricStack" in f.component),
        None,
    )
    assert gc_check is not None, "Expected gradient checkpointing check"
    assert not gc_check.passed, "Gradient checkpointing should NOT be wired yet"
    assert gc_check.severity == SEVERITY_HIGH


def test_oom_prevention_amp_missing_in_trainer2():
    """OOMPreventionAgent detects AMP is missing from trainer2."""
    agent = OOMPreventionAgent()
    findings = agent.audit()
    amp_check = next(
        (f for f in findings if "autocast" in f.check.lower() or "AMP" in f.check), None
    )
    assert amp_check is not None
    assert not amp_check.passed, "AMP should NOT be wired in trainer2 yet"
    assert amp_check.severity == SEVERITY_HIGH


def test_oom_prevention_bptt_detach_present():
    """OOMPreventionAgent confirms BPTT detach is implemented in BiQuatCausalLayer."""
    agent = OOMPreventionAgent()
    findings = agent.audit()
    bptt = next(
        (f for f in findings if "bptt" in f.check.lower() or "BPTT" in f.check), None
    )
    assert bptt is not None
    assert bptt.passed, "BPTT detach should be implemented"


def test_oom_prevention_inference_mode_present():
    """OOMPreventionAgent confirms inference_mode/no_grad is used in trainer2."""
    agent = OOMPreventionAgent()
    findings = agent.audit()
    inf_mode = next(
        (f for f in findings if "inference_mode" in f.check or "no_grad" in f.check), None
    )
    assert inf_mode is not None
    assert inf_mode.passed, "inference_mode/no_grad should be used in trainer2"


def test_oom_prevention_expandable_segments_missing():
    """OOMPreventionAgent detects expandable_segments not set at entry point."""
    agent = OOMPreventionAgent()
    findings = agent.audit()
    exp_check = next(
        (f for f in findings if "expandable_segments" in f.check.lower() and "main.py" in f.component),
        None,
    )
    assert exp_check is not None
    assert not exp_check.passed
    assert exp_check.severity == SEVERITY_HIGH


# ---------------------------------------------------------------------------
# MoraleAgent
# ---------------------------------------------------------------------------

def test_morale_returns_notes():
    """MoraleAgent returns non-empty notes."""
    team = MemManagementAuditTeam()
    report = team.run()
    assert len(report.morale_notes) >= 2


def test_morale_mentions_approval():
    """MoraleAgent always includes plan-only / approval note."""
    team = MemManagementAuditTeam()
    report = team.run()
    assert any("approval" in n.lower() or "plan-only" in n.lower() for n in report.morale_notes)


# ---------------------------------------------------------------------------
# ScribeAgent
# ---------------------------------------------------------------------------

def test_scribe_consolidates_all_findings():
    """ScribeAgent consolidates findings from all specialist agents."""
    team = MemManagementAuditTeam()
    report = team.run()
    log = report.scribe_log
    assert isinstance(log, ScribeLog)
    assert len(log.findings) == len(report.findings)
    assert log.summary != ""


def test_scribe_action_items_ordered_by_severity():
    """ScribeAgent orders action items CRITICAL → HIGH → MEDIUM → LOW."""
    team = MemManagementAuditTeam()
    report = team.run()
    log = report.scribe_log
    items = log.action_items
    # Verify no MEDIUM item appears before a HIGH item
    seen_high = False
    for item in items:
        if "[HIGH]" in item:
            seen_high = True
        if "[MEDIUM]" in item and not seen_high:
            # MEDIUM appeared before any HIGH — ordering violated
            # Only flag if there ARE high items
            high_items = [i for i in items if "[HIGH]" in i]
            if high_items:
                pytest.fail(f"MEDIUM item '{item}' appears before HIGH items")
            break


def test_scribe_failed_findings_in_action_items():
    """Every failed finding appears in action items."""
    team = MemManagementAuditTeam()
    report = team.run()
    log = report.scribe_log
    failed = log.failed_findings
    for f in failed:
        matched = any(f.component in item or f.check[:30] in item for item in log.action_items)
        assert matched, f"Failed finding not in action items: {f.check}"


# ---------------------------------------------------------------------------
# Pipeline wiring checks
# ---------------------------------------------------------------------------

def test_pipeline_wiring_returns_checks():
    """_check_pipeline_wiring returns at least 5 checks."""
    checks = _check_pipeline_wiring()
    assert len(checks) >= 5


def test_pipeline_wiring_gpu_cleanup_thread_not_wired():
    """GPUCleanupThread is not yet wired in trainer2."""
    checks = _check_pipeline_wiring()
    cleanup_thread = next(
        (c for c in checks if "GPUCleanupThread" in c.feature), None
    )
    assert cleanup_thread is not None
    assert not cleanup_thread.wired


def test_pipeline_wiring_amp_not_wired():
    """AMP autocast is not yet wired in trainer2."""
    checks = _check_pipeline_wiring()
    amp = next((c for c in checks if "autocast" in c.feature.lower() or "AMP" in c.feature), None)
    assert amp is not None
    assert not amp.wired


def test_pipeline_wiring_expandable_segments_not_set():
    """expandable_segments is not set at entry point."""
    checks = _check_pipeline_wiring()
    exp = next(
        (c for c in checks if "expandable_segments" in c.feature), None
    )
    assert exp is not None
    assert not exp.wired


def test_pipeline_wiring_inline_empty_cache_present():
    """Inline empty_cache() is wired in run_two_phase_and_update."""
    checks = _check_pipeline_wiring()
    ec = next(
        (c for c in checks if "empty_cache" in c.feature.lower()), None
    )
    assert ec is not None
    assert ec.wired


# ---------------------------------------------------------------------------
# MemAuditReport properties
# ---------------------------------------------------------------------------

def test_report_passed_property_reflects_findings():
    """MemAuditReport.passed is False when any finding fails."""
    team = MemManagementAuditTeam()
    report = team.run()
    # We expect failures (OOM gaps not yet fixed)
    assert not report.passed


def test_report_critical_findings_list():
    """ScribeLog.critical_findings returns only CRITICAL+failed findings."""
    team = MemManagementAuditTeam()
    report = team.run()
    crits = report.scribe_log.critical_findings
    for f in crits:
        assert not f.passed
        assert f.severity == SEVERITY_CRITICAL


def test_report_failed_findings_list():
    """ScribeLog.failed_findings returns only failed findings."""
    team = MemManagementAuditTeam()
    report = team.run()
    failed = report.scribe_log.failed_findings
    for f in failed:
        assert not f.passed


# ---------------------------------------------------------------------------
# Severity constant values
# ---------------------------------------------------------------------------

def test_severity_constants_are_strings():
    """Severity constants are non-empty strings."""
    for sev in (SEVERITY_INFO, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL):
        assert isinstance(sev, str) and sev != ""


def test_severity_constants_distinct():
    """All severity constants have distinct values."""
    sevs = [SEVERITY_INFO, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL]
    assert len(set(sevs)) == 5
