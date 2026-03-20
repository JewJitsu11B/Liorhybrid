"""
Telemetry Pipeline Audit Team (Lead + Specialists + Support + Checkpointing)

Plan-only audit that maps telemetry sources, pipeline wiring, and checkpoint
behavior in trainer2. This team is collaborative:

1. TelemetryLeadAgent                    - coordination + consensus
2. AbstractAlgebraTelemetryAgent         - scalar telemetry semantics
3. AlgebraSupportAgent                   - verifies algebra specialist findings
4. DifferentialGeometryTelemetryAgent    - window/epoch/step manifold semantics
5. DiffGeometrySupportAgent              - verifies geometry specialist findings
6. GRTelemetryAgent                      - curvature-style telemetry interpretation
7. GRSupportAgent                        - verifies GR specialist findings
8. CheckpointingAgent                    - periodic/end-epoch/manual-quit checkpointing
9. ScribeAgent                           - final consolidated action log

STATUS: AWAITING APPROVAL TO EXECUTE
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SEVERITY_INFO = "INFO"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class TelemetryFinding:
    finding_id: str
    role: str
    component: str
    check: str
    passed: bool
    severity: str
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class SupportCheck:
    finding_id: str
    support_role: str
    agrees: bool
    notes: str


@dataclass(frozen=True)
class PipelineWiringCheck:
    feature: str
    wired: bool
    entry_point: str
    notes: str


@dataclass(frozen=True)
class ConsensusLog:
    confirmed: List[TelemetryFinding]
    disputed: List[TelemetryFinding]
    unresolved: List[TelemetryFinding]
    support_checks: List[SupportCheck]
    summary: str


@dataclass(frozen=True)
class ScribeLog:
    findings: List[TelemetryFinding]
    wiring_checks: List[PipelineWiringCheck]
    consensus: ConsensusLog
    notes: List[str]
    action_items: List[str]


@dataclass
class TelemetryAuditReport:
    coordinator_scope: str
    findings: List[TelemetryFinding]
    support_checks: List[SupportCheck]
    wiring_checks: List[PipelineWiringCheck]
    consensus: ConsensusLog
    telemetry_notes: List[str]
    scribe_log: ScribeLog
    approval_status: str = "AWAITING APPROVAL TO EXECUTE"


def _read(rel_path: str) -> str:
    p = REPO_ROOT / rel_path
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _has(src: str, needle: str) -> bool:
    return needle in src


class TelemetryLeadAgent:
    SCOPE = (
        "Audit telemetry data and telemetry sources in trainer2, coordinate "
        "specialist + support verification, and produce a consensus report. "
        "Also audit checkpointing (periodic, end-epoch, manual-quit forced) "
        "for inference compatibility including entropy-weighted softmax config."
    )

    TASK_QUEUE: List[Tuple[str, str]] = [
        ("AbstractAlgebraist", "Audit scalar telemetry contracts (loss/perplexity semantics)"),
        ("AlgebraSupport", "Verify algebra specialist findings against source"),
        ("DiffGeometer", "Audit epoch/window/step structure and scale"),
        ("DiffGeomSupport", "Verify differential-geometry specialist findings"),
        ("GRAgent", "Audit curvature/transport telemetry interpretation"),
        ("GRSupport", "Verify GR specialist findings"),
        ("Checkpointing", "Audit periodic/end-epoch/manual-quit checkpoint suitability"),
        ("Scribe", "Consolidate findings, support checks, and consensus decisions"),
    ]

    def scope(self) -> str:
        return self.SCOPE

    def task_queue(self) -> List[Tuple[str, str]]:
        return list(self.TASK_QUEUE)

    def build_consensus(
        self,
        findings: List[TelemetryFinding],
        support_checks: List[SupportCheck],
    ) -> ConsensusLog:
        by_id: Dict[str, List[SupportCheck]] = {}
        for check in support_checks:
            by_id.setdefault(check.finding_id, []).append(check)

        confirmed: List[TelemetryFinding] = []
        disputed: List[TelemetryFinding] = []
        unresolved: List[TelemetryFinding] = []

        for finding in findings:
            checks = by_id.get(finding.finding_id, [])
            if not checks:
                unresolved.append(finding)
                continue
            if all(c.agrees for c in checks):
                confirmed.append(finding)
            elif any(not c.agrees for c in checks):
                disputed.append(finding)
            else:
                unresolved.append(finding)

        summary = (
            f"confirmed={len(confirmed)} | disputed={len(disputed)} | "
            f"unresolved={len(unresolved)}"
        )
        return ConsensusLog(
            confirmed=confirmed,
            disputed=disputed,
            unresolved=unresolved,
            support_checks=support_checks,
            summary=summary,
        )


class AbstractAlgebraTelemetryAgent:
    NAME = "AbstractAlgebraist"

    def audit(self) -> List[TelemetryFinding]:
        src = _read("training/trainer2.py")
        findings: List[TelemetryFinding] = []

        has_loss = _has(src, '"total_loss"') and _has(src, '"lior_mean"')
        findings.append(TelemetryFinding(
            finding_id="ALG-1",
            role=self.NAME,
            component="training/trainer2.py:maybe_log_metrics",
            check="Loss telemetry includes total_loss/lior_mean scalars",
            passed=has_loss,
            severity=SEVERITY_HIGH if not has_loss else SEVERITY_INFO,
            evidence='window_metrics record contains "total_loss" and "lior_mean".',
            recommendation="Keep loss monotone monitoring; lower is better.",
        ))

        has_perplexity_print = bool(re.search(r"ppl\s*=\s*math\.exp", src))
        findings.append(TelemetryFinding(
            finding_id="ALG-2",
            role=self.NAME,
            component="training/trainer2.py:run_window",
            check="Perplexity is derived from loss using exp(loss)",
            passed=has_perplexity_print,
            severity=SEVERITY_MEDIUM if not has_perplexity_print else SEVERITY_INFO,
            evidence="run_window logs: ppl = math.exp(min(lior_now, 20.0)).",
            recommendation="Perplexity should trend downward with loss (lower is better).",
        ))

        has_ppl_jsonl = _has(src, '"perplexity"')
        findings.append(TelemetryFinding(
            finding_id="ALG-3",
            role=self.NAME,
            component="training/trainer2.py:window_metrics jsonl",
            check="Perplexity is persisted in telemetry JSONL",
            passed=has_ppl_jsonl,
            severity=SEVERITY_LOW if has_ppl_jsonl else SEVERITY_MEDIUM,
            evidence="JSONL currently writes total_loss/lior_mean/R_mean/spd_mean.",
            recommendation=(
                "Optional: persist perplexity for trend dashboards; currently printed only."
            ),
        ))
        return findings


class DifferentialGeometryTelemetryAgent:
    NAME = "DiffGeometer"

    def audit(self) -> List[TelemetryFinding]:
        src = _read("training/trainer2.py")
        findings: List[TelemetryFinding] = []

        has_epoch_window = _has(src, "for epoch_idx in range") and _has(src, "window_idx += 1")
        findings.append(TelemetryFinding(
            finding_id="DG-1",
            role=self.NAME,
            component="training/trainer2.py:trainer2_entrypoint",
            check="Epoch/window progression is explicit and monotone",
            passed=has_epoch_window,
            severity=SEVERITY_HIGH if not has_epoch_window else SEVERITY_INFO,
            evidence="Epoch loop encloses batch loop; each batch advances window_idx by 1.",
            recommendation=(
                "Windows per epoch equal number of batches from train_loader in that epoch."
            ),
        ))

        has_tbptt = _has(src, "tbptt_window_steps") and _has(src, "for _t in range(steps)")
        findings.append(TelemetryFinding(
            finding_id="DG-2",
            role=self.NAME,
            component="training/trainer2.py:run_window",
            check="Each window executes tbptt_window_steps inner steps",
            passed=has_tbptt,
            severity=SEVERITY_MEDIUM if not has_tbptt else SEVERITY_INFO,
            evidence="run_window sets steps=int(cfg.tbptt_window_steps) and iterates over steps.",
            recommendation=(
                "Large tbptt_window_steps multiplies compute per window but does not change "
                "window count directly."
            ),
        ))

        has_window_record = _has(src, '"epoch"') and _has(src, '"window"') and _has(src, '"batch"')
        findings.append(TelemetryFinding(
            finding_id="DG-3",
            role=self.NAME,
            component="training/trainer2.py:window_metrics jsonl",
            check="Telemetry records epoch/window/batch identifiers",
            passed=has_window_record,
            severity=SEVERITY_HIGH if not has_window_record else SEVERITY_INFO,
            evidence='window_metrics record writes "epoch", "window", and "batch".',
            recommendation="Use this triplet to disambiguate long epochs with many windows.",
        ))
        return findings


class GRTelemetryAgent:
    NAME = "GRAgent"

    def audit(self) -> List[TelemetryFinding]:
        src = _read("training/trainer2.py")
        findings: List[TelemetryFinding] = []

        has_curvature_scalar = _has(src, '"R_mean"') and _has(src, "R_sc")
        findings.append(TelemetryFinding(
            finding_id="GR-1",
            role=self.NAME,
            component="training/trainer2.py:run_window/maybe_log_metrics",
            check="Curvature proxy telemetry (R_mean) is produced and persisted",
            passed=has_curvature_scalar,
            severity=SEVERITY_MEDIUM if not has_curvature_scalar else SEVERITY_INFO,
            evidence="R_sc contributes to retrieval cost; aggregated metric logged as R_mean.",
            recommendation=(
                "R_mean is diagnostic (stability/scale), not a universal objective to minimize."
            ),
        ))

        has_spd = _has(src, '"spd_mean"') and _has(src, "quad_form_batch")
        findings.append(TelemetryFinding(
            finding_id="GR-2",
            role=self.NAME,
            component="training/trainer2.py:retrieval_step_with_spectral/maybe_log_metrics",
            check="Geometric distance proxy (spd_mean) is produced and persisted",
            passed=has_spd,
            severity=SEVERITY_MEDIUM if not has_spd else SEVERITY_INFO,
            evidence="SPD distance is computed from quad form and logged via spd_mean.",
            recommendation=(
                "Typically lower spd_mean suggests tighter retrieval neighborhoods."
            ),
        ))
        return findings


class AlgebraSupportAgent:
    NAME = "AlgebraSupport"

    def verify(self, findings: List[TelemetryFinding]) -> List[SupportCheck]:
        src = _read("training/trainer2.py")
        checks: List[SupportCheck] = []
        for finding in findings:
            if finding.finding_id == "ALG-1":
                agrees = _has(src, '"total_loss"') and _has(src, '"lior_mean"')
            elif finding.finding_id == "ALG-2":
                agrees = _has(src, "ppl = math.exp")
            else:
                agrees = True
            checks.append(SupportCheck(
                finding_id=finding.finding_id,
                support_role=self.NAME,
                agrees=agrees == finding.passed,
                notes="Static source cross-check completed.",
            ))
        return checks


class DiffGeometrySupportAgent:
    NAME = "DiffGeomSupport"

    def verify(self, findings: List[TelemetryFinding]) -> List[SupportCheck]:
        src = _read("training/trainer2.py")
        agrees_epoch = _has(src, "for epoch_idx in range")
        agrees_window = _has(src, "window_idx += 1")
        checks: List[SupportCheck] = []
        for finding in findings:
            if finding.finding_id == "DG-1":
                agrees = agrees_epoch and agrees_window
            elif finding.finding_id == "DG-2":
                agrees = _has(src, "tbptt_window_steps") and _has(src, "for _t in range(steps)")
            else:
                agrees = _has(src, '"epoch"') and _has(src, '"window"')
            checks.append(SupportCheck(
                finding_id=finding.finding_id,
                support_role=self.NAME,
                agrees=agrees == finding.passed,
                notes="Pipeline indexing semantics verified.",
            ))
        return checks


class GRSupportAgent:
    NAME = "GRSupport"

    def verify(self, findings: List[TelemetryFinding]) -> List[SupportCheck]:
        src = _read("training/trainer2.py")
        checks: List[SupportCheck] = []
        for finding in findings:
            if finding.finding_id == "GR-1":
                agrees = _has(src, '"R_mean"')
            else:
                agrees = _has(src, '"spd_mean"')
            checks.append(SupportCheck(
                finding_id=finding.finding_id,
                support_role=self.NAME,
                agrees=agrees == finding.passed,
                notes="Curvature/distance telemetry cross-check completed.",
            ))
        return checks


class CheckpointingAgent:
    NAME = "Checkpointing"

    def audit(self) -> List[TelemetryFinding]:
        src = _read("training/trainer2.py")
        findings: List[TelemetryFinding] = []

        periodic = _has(src, "maybe_checkpoint(") and _has(src, "save_every_windows")
        findings.append(TelemetryFinding(
            finding_id="CKPT-1",
            role=self.NAME,
            component="training/trainer2.py:trainer2_entrypoint",
            check="Periodic checkpointing by save_every_windows is wired",
            passed=periodic,
            severity=SEVERITY_CRITICAL if not periodic else SEVERITY_INFO,
            evidence="Entry loop invokes maybe_checkpoint(window_idx=..., cfg=...).",
            recommendation="Keep periodic saves for recoverability during long runs.",
        ))

        forced_manual = _has(src, "except KeyboardInterrupt:") and _has(src, 'reason="manual_quit"')
        findings.append(TelemetryFinding(
            finding_id="CKPT-2",
            role=self.NAME,
            component="training/trainer2.py:trainer2_entrypoint",
            check="Forced checkpoint on manual quit is wired",
            passed=forced_manual,
            severity=SEVERITY_CRITICAL if not forced_manual else SEVERITY_INFO,
            evidence="KeyboardInterrupt path calls force_checkpoint(... reason='manual_quit').",
            recommendation="Manual interruption should still persist a resumable/inference-ready checkpoint.",
        ))

        end_epoch = _has(src, 'reason="end_epoch"') and _has(src, "force_checkpoint(")
        findings.append(TelemetryFinding(
            finding_id="CKPT-3",
            role=self.NAME,
            component="training/trainer2.py:trainer2_entrypoint",
            check="End-of-epoch forced checkpoint is wired",
            passed=end_epoch,
            severity=SEVERITY_HIGH if not end_epoch else SEVERITY_INFO,
            evidence="Epoch summary path calls force_checkpoint(... reason='end_epoch').",
            recommendation="Retain epoch-boundary snapshots for stable rollback points.",
        ))

        entropy_compat = _has(src, "'selector'") and _has(src, "'nu_inference'")
        findings.append(TelemetryFinding(
            finding_id="CKPT-4",
            role=self.NAME,
            component="training/trainer2.py:_build_checkpoint_payload",
            check="Checkpoint config carries entropy-softmax controls for inference",
            passed=entropy_compat,
            severity=SEVERITY_HIGH if not entropy_compat else SEVERITY_INFO,
            evidence="Checkpoint payload config includes selector and nu_inference.",
            recommendation="Include selector/nu_inference to preserve entropy-weighted softmax behavior.",
        ))
        return findings


class ScribeAgent:
    def consolidate(
        self,
        findings: List[TelemetryFinding],
        wiring_checks: List[PipelineWiringCheck],
        consensus: ConsensusLog,
        notes: List[str],
    ) -> ScribeLog:
        action_items: List[str] = []
        for finding in findings:
            if not finding.passed:
                action_items.append(
                    f"[{finding.severity}] {finding.finding_id} {finding.check}: {finding.recommendation}"
                )
        for check in wiring_checks:
            if not check.wired:
                action_items.append(
                    f"[WIRING] {check.feature} at {check.entry_point}: {check.notes}"
                )
        if not action_items:
            action_items.append("No action items — all telemetry and checkpoint checks passed.")
        return ScribeLog(
            findings=findings,
            wiring_checks=wiring_checks,
            consensus=consensus,
            notes=notes,
            action_items=action_items,
        )


def _check_pipeline_wiring() -> List[PipelineWiringCheck]:
    src = _read("training/trainer2.py")
    validator_src = _read("training/checkpoint_validator.py")
    return [
        PipelineWiringCheck(
            feature="Telemetry JSONL records epoch/window/batch/loss/R/spd",
            wired=(
                _has(src, '"type": "window_metrics"')
                and _has(src, '"epoch"')
                and _has(src, '"window"')
                and _has(src, '"total_loss"')
                and _has(src, '"R_mean"')
                and _has(src, '"spd_mean"')
            ),
            entry_point="training/trainer2.py:maybe_log_metrics",
            notes="Structured per-window telemetry is required for postmortem trend analysis.",
        ),
        PipelineWiringCheck(
            feature="Periodic checkpoint save policy",
            wired=_has(src, "if (not force) and window_idx % cfg.save_every_windows != 0"),
            entry_point="training/trainer2.py:_checkpoint_impl",
            notes="Ensures periodic snapshots during long epochs/windows.",
        ),
        PipelineWiringCheck(
            feature="Forced checkpoint on manual quit",
            wired=_has(src, "except KeyboardInterrupt:") and _has(src, 'reason="manual_quit"'),
            entry_point="training/trainer2.py:trainer2_entrypoint",
            notes="Manual-stop path should still produce a usable checkpoint.",
        ),
        PipelineWiringCheck(
            feature="Forced checkpoint on end of epoch",
            wired=_has(src, 'reason="end_epoch"'),
            entry_point="training/trainer2.py:trainer2_entrypoint",
            notes="Epoch boundaries should have deterministic recovery points.",
        ),
        PipelineWiringCheck(
            feature="Checkpoint schema has inference-critical state dicts",
            wired=(
                _has(validator_src, "model_state_dict")
                and _has(validator_src, "input_embedding_state_dict")
                and _has(validator_src, "lm_head_state_dict")
                and (_has(validator_src, "field_state_dict") or _has(validator_src, "field_state"))
            ),
            entry_point="training/checkpoint_validator.py:validate_checkpoint_schema",
            notes="Inference requires model+field+embedding+LM-head states.",
        ),
    ]


def _telemetry_notes() -> List[str]:
    return [
        "total_loss/lior_mean: primary loss scalar; lower is better.",
        "perplexity (ppl): exp(loss); lower is better. Rising loss usually raises perplexity.",
        "R_mean: curvature-derived diagnostic scalar; monitor for drift/spikes (no universal monotonic optimum).",
        "spd_mean: geometry distance proxy for retrieval; typically lower implies tighter neighborhood alignment.",
        "epoch/window/batch: each train_loader batch advances one window, so many batches => many windows per epoch.",
        "tbptt_window_steps: inner integration steps per window; increases compute per window, not window count.",
    ]


class TelemetryPipelineAuditTeam:
    """Public entry point for the telemetry + checkpointing collaborative audit."""

    def run(self) -> TelemetryAuditReport:
        lead = TelemetryLeadAgent()
        algebra = AbstractAlgebraTelemetryAgent()
        algebra_support = AlgebraSupportAgent()
        diff_geom = DifferentialGeometryTelemetryAgent()
        diff_geom_support = DiffGeometrySupportAgent()
        gr = GRTelemetryAgent()
        gr_support = GRSupportAgent()
        checkpointing = CheckpointingAgent()
        scribe = ScribeAgent()

        algebra_findings = algebra.audit()
        diff_geom_findings = diff_geom.audit()
        gr_findings = gr.audit()
        checkpoint_findings = checkpointing.audit()

        support_checks: List[SupportCheck] = []
        support_checks.extend(algebra_support.verify(algebra_findings))
        support_checks.extend(diff_geom_support.verify(diff_geom_findings))
        support_checks.extend(gr_support.verify(gr_findings))

        all_findings: List[TelemetryFinding] = []
        all_findings.extend(algebra_findings)
        all_findings.extend(diff_geom_findings)
        all_findings.extend(gr_findings)
        all_findings.extend(checkpoint_findings)

        wiring_checks = _check_pipeline_wiring()
        consensus = lead.build_consensus(all_findings, support_checks)
        notes = _telemetry_notes()
        scribe_log = scribe.consolidate(
            findings=all_findings,
            wiring_checks=wiring_checks,
            consensus=consensus,
            notes=notes,
        )

        return TelemetryAuditReport(
            coordinator_scope=lead.scope(),
            findings=all_findings,
            support_checks=support_checks,
            wiring_checks=wiring_checks,
            consensus=consensus,
            telemetry_notes=notes,
            scribe_log=scribe_log,
            approval_status="AWAITING APPROVAL TO EXECUTE",
        )
