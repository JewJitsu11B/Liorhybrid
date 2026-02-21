"""
Tests for multi-agent full-pipeline math validation.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from utils.math_validation_team import BRIDGE_AGENT_NAMES, DefaultAddressLayout, MathValidationTeam
from utils.pipeline_audit import reset_audit, write_math_validation_report


def test_math_validation_team_passes_default_pipeline():
    team = MathValidationTeam()
    report = team.validate_all()

    assert report.passed
    assert len(report.findings) >= 8
    assert all(f.literature for f in report.findings)
    assert report.logic_audit_comments
    assert report.stub_output.pseudocode
    assert report.stub_output.formalisms
    assert report.bridge_plan == []
    assert all("path=" in c and "why=" in c for c in report.logic_audit_comments)


def test_math_validation_team_detects_precompute_contract_break():
    team = MathValidationTeam()
    broken_cfg = DefaultAddressLayout(n_nearest=30, n_high_sim=10, n_low_sim=10, m=5)

    report = team.validate_all(address_config=broken_cfg)
    failed = [f for f in report.failed_checks if "Precompute dimensions satisfy Option-6 contracts" in f.check]

    assert failed, "Expected precompute contract failure was not detected."
    assert set(BRIDGE_AGENT_NAMES) == {
        s.owner_agent for s in report.bridge_plan
    }


def test_math_validation_report_written_to_pipeline_audit(monkeypatch, tmp_path):
    audit_file = tmp_path / "pipeline_audit.md"
    monkeypatch.setenv("BCF_PIPELINE_AUDIT_PATH", str(audit_file))

    reset_audit()
    write_math_validation_report()
    text = audit_file.read_text(encoding="utf-8")

    assert "Math Validation Team" in text
    assert "full_pipeline_math_validation" in text
    assert "Logic Audit Comments" in text
    assert "Stub Team: Pseudocode" in text
    assert "Stub Team: Formalisms" in text
