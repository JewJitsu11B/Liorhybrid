"""
Pipeline Audit Logger

Purpose:
- Print a single-line "flag" the first time a given file participates in the pipeline.
- Append file+folder call records to a markdown audit file.

Design constraints:
- Must not read GPU tensors or call `.item()`; this module only deals with strings/paths.
- Safe to call inside compiled/AMP regions (no device sync).
"""
from __future__ import annotations
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import threading
from typing import Optional, Set, List


_LOCK = threading.Lock()
_SEEN_FILES: Set[str] = set()
_EVENTS: List["AuditEvent"] = []


@dataclass(frozen=True)
class AuditEvent:
    ts_utc: str
    label: str
    file: str
    folder: str


def _repo_root() -> Path:
    # utils/ -> Liorhybrid/
    return Path(__file__).resolve().parents[1]


def _audit_path() -> Path:
    p = os.environ.get("BCF_PIPELINE_AUDIT_PATH")
    if p:
        return Path(p).expanduser().resolve()
    return (_repo_root() / "pipeline_audit.md").resolve()


def reset_audit() -> Path:
    """
    Clear in-memory registry and overwrite the markdown audit file header.
    """
    with _LOCK:
        _SEEN_FILES.clear()
        _EVENTS.clear()

        out = _audit_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        out.write_text(f"# Pipeline Audit\n\nRun started (UTC): `{now}`\n\n## Events\n", encoding="utf-8")
        return out


def audit_file_once(label: str, file_path: str) -> None:
    """
    Log a one-line CLI flag and append to audit markdown on first observation of file_path.
    """
    try:
        file_abs = Path(file_path).resolve()
    except Exception:
        file_abs = Path(str(file_path))

    try:
        rel = file_abs.relative_to(_repo_root())
        rel_str = str(rel)
        folder_str = str(rel.parent)
    except Exception:
        rel_str = str(file_abs)
        folder_str = str(file_abs.parent)

    with _LOCK:
        if rel_str in _SEEN_FILES:
            return
        _SEEN_FILES.add(rel_str)

        ts = datetime.now(timezone.utc).isoformat()
        ev = AuditEvent(ts_utc=ts, label=str(label), file=rel_str, folder=folder_str)
        _EVENTS.append(ev)

        # CLI one-line flag (human scannable)
        print(f"[PIPELINE] {ev.label} :: {ev.file}", flush=True)

        # Append to markdown
        out = _audit_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(f"- `{ev.ts_utc}` | `{ev.label}` | `{ev.file}` | `{ev.folder}`\n")


def write_summary() -> Path:
    """
    Append a summary section (unique files) to the audit markdown.
    """
    with _LOCK:
        out = _audit_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write("\n## Unique Files\n")
            for fp in sorted(_SEEN_FILES):
                f.write(f"- `{fp}`\n")
        return out


def write_transport_fiber_audit_report(
    label: str = "transport_fiber_bundle_audit",
) -> Path:
    """
    Run the seven-agent transport & fiber bundle audit and append findings to
    the audit markdown.  No code changes are made — findings are plan-only.
    """
    from .transport_fiber_audit_team import TransportFiberAuditTeam

    report = TransportFiberAuditTeam().run()
    out = _audit_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        with out.open("a", encoding="utf-8") as f:
            f.write(f"\n## Transport & Fiber Bundle Audit Team ({label})\n")
            f.write(f"- Approval status: `{report.approval_status}`\n")
            f.write(f"- Coordinator scope: {report.coordinator_scope}\n")
            f.write(f"- Overall: `{'PASS' if report.passed else 'FAIL'}`\n\n")

            f.write("### Agent Findings\n")
            for finding in report.findings:
                status = "PASS" if finding.passed else "FAIL"
                f.write(
                    f"- `{status}` | `{finding.severity}` | `{finding.role}` | "
                    f"`{finding.operator}` | {finding.check} | "
                    f"evidence: {finding.evidence}\n"
                )

            f.write("\n### Pipeline Wiring Checks\n")
            for wc in report.wiring_checks:
                wired_str = "WIRED" if wc.wired else "NOT WIRED"
                f.write(
                    f"- `{wired_str}` | `{wc.operator}` | "
                    f"entry: {wc.entry_point} | {wc.notes}\n"
                )

            f.write("\n### Morale Notes\n")
            for note in report.morale_notes:
                f.write(f"- {note}\n")

            f.write("\n### Scribe: Summary\n")
            f.write(f"- {report.scribe_log.summary}\n")

            f.write("\n### Scribe: Action Items (Pending Approval)\n")
            for item in report.scribe_log.action_items:
                f.write(f"- {item}\n")

    return out


def write_math_validation_report(label: str = "full_pipeline_math_validation") -> Path:
    """
    Run multi-agent math validation and append findings to the audit markdown.
    """
    from .math_validation_team import MathValidationTeam

    report = MathValidationTeam().validate_all()
    out = _audit_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        with out.open("a", encoding="utf-8") as f:
            f.write(f"\n## Math Validation Team ({label})\n")
            f.write(f"- Overall: `{'PASS' if report.passed else 'FAIL'}`\n")
            for finding in report.findings:
                status = "PASS" if finding.passed else "FAIL"
                f.write(
                    f"- `{status}` | `{finding.agent}` | `{finding.check}` | "
                    f"{finding.details} | path: {finding.logic_path} | "
                    f"why: {finding.rationale} | ref: {finding.literature}\n"
                )
            f.write("\n### Logic Audit Comments\n")
            for comment in report.logic_audit_comments:
                f.write(f"- {comment}\n")
            f.write("\n### Stub Team: Pseudocode\n")
            for line in report.stub_output.pseudocode:
                f.write(f"- {line}\n")
            f.write("\n### Stub Team: Formalisms\n")
            for line in report.stub_output.formalisms:
                f.write(f"- {line}\n")
            f.write("\n### Bridge Plan\n")
            if report.bridge_plan:
                for step in report.bridge_plan:
                    f.write(
                        f"- `{step.owner_agent}` | gap: {step.gap} | action: {step.action} | "
                        f"done-when: {step.completion_criterion}\n"
                    )
            else:
                f.write("- No unresolved gaps; bridge agents not required.\n")
    return out
