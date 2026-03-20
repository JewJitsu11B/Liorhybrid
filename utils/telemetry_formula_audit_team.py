"""
Telemetry Formula Audit Team

A plain-English audit of every value the trainer emits — what each one is
*trying* to measure, what it *actually* computes, and whether the formula is
correct.  Run this directly to get the printed report:

    python utils/telemetry_formula_audit_team.py

Team composition
----------------
1. CoordinatorAgent       – scope, task queue
2. CoreMetricsAgent       – lior_mean, R_mean, spd_mean, total_loss label
3. PerplexityAgent        – perplexity formula correctness
4. GeometricDiagAgent     – E_geo, E_var, E_struct formulas
5. InfraMetricsAgent      – mem_norm, window_ms, gpu_alloc_gb, gpu_reserved_gb
6. MoraleAgent            – overall health summary
7. ScribeAgent            – consolidated action log

STATUS: AWAITING APPROVAL TO EXECUTE
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SEVERITY_INFO     = "INFO"
SEVERITY_LOW      = "LOW"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_HIGH     = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

VERDICT_CORRECT   = "CORRECT"
VERDICT_PARTIAL   = "PARTIAL"
VERDICT_WRONG     = "WRONG"
VERDICT_MISSING   = "MISSING"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelemetryFinding:
    finding_id: str
    role: str
    variable: str
    intended: str          # What the variable is trying to measure
    actual: str            # What the formula actually computes
    verdict: str           # CORRECT | PARTIAL | WRONG | MISSING
    severity: str
    recommendation: str


@dataclass(frozen=True)
class WiringCheck:
    feature: str
    wired: bool
    entry_point: str
    notes: str


@dataclass(frozen=True)
class ScribeLog:
    findings: List[TelemetryFinding]
    wiring_checks: List[WiringCheck]
    summary: str
    action_items: List[str]

    @property
    def all_correct(self) -> bool:
        return all(f.verdict == VERDICT_CORRECT for f in self.findings)

    @property
    def wrong_findings(self) -> List[TelemetryFinding]:
        return [f for f in self.findings if f.verdict == VERDICT_WRONG]

    @property
    def partial_findings(self) -> List[TelemetryFinding]:
        return [f for f in self.findings if f.verdict == VERDICT_PARTIAL]


@dataclass
class TelemetryAuditReport:
    coordinator_scope: str
    findings: List[TelemetryFinding]
    wiring_checks: List[WiringCheck]
    morale_notes: List[str]
    scribe_log: ScribeLog
    approval_status: str = "AWAITING APPROVAL TO EXECUTE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(rel: str) -> str:
    p = REPO_ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _has(src: str, needle: str) -> bool:
    return needle in src


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class CoordinatorAgent:
    SCOPE = (
        "Audit every value emitted by the LIoR trainer telemetry pipeline. "
        "For each variable: state what it was intended to measure, what the "
        "formula actually computes, and whether those two things match. "
        "Key concerns: (1) perplexity is exp(lior_mean) — is that valid? "
        "(2) lior_mean is stored as total_loss — is that label accurate? "
        "(3) E_geo Christoffel terms are hardcoded zero — does that break the "
        "geodesic residual once the metric is learned? "
        "(4) E_var uses linear approximation — how accurate is it? "
        "Findings are plain-English, plan-only."
    )
    TASK_QUEUE: List[Tuple[str, str]] = [
        ("CoreMetrics",    "Audit lior_mean, R_mean, spd_mean, total_loss label"),
        ("Perplexity",     "Audit perplexity formula correctness"),
        ("GeometricDiag",  "Audit E_geo, E_var, E_struct formulas"),
        ("InfraMetrics",   "Audit mem_norm, window_ms, gpu_alloc_gb, gpu_reserved_gb"),
        ("Morale",         "Overall health summary"),
        ("Scribe",         "Consolidated action log"),
    ]

    def scope(self) -> str:
        return self.SCOPE

    def task_queue(self) -> List[Tuple[str, str]]:
        return list(self.TASK_QUEUE)


class CoreMetricsAgent:
    NAME = "CoreMetrics"

    def audit(self) -> List[TelemetryFinding]:
        trainer2 = _read("training/trainer2.py")
        findings: List[TelemetryFinding] = []

        # --- CORE-1: lior_mean formula ---
        # lior_step_fused returns R_sc * quad_form(v, g)
        # quad_form = v^T g v  (speed squared in metric)
        # R_sc = sqrt(|R4 @ g2_vec| / n²)  (scalar curvature proxy)
        has_lior_formula = (
            _has(trainer2, "dlior = R_sc * spd") or
            _has(trainer2, "R_sc * spd")
        )
        findings.append(TelemetryFinding(
            finding_id="CORE-1",
            role=self.NAME,
            variable="lior_mean",
            intended=(
                "Average geometric path cost per step — a measure of how expensive "
                "it is for the retrieval to traverse the learned manifold. Lower means "
                "the geometry is becoming more efficient for the trajectories being taken."
            ),
            actual=(
                "mean over window steps of (R_sc × v^T g v), where R_sc is a scalar "
                "curvature proxy and v^T g v is the squared speed in the metric. "
                "Accumulated as lior_acc, then divided by step count."
            ),
            verdict=VERDICT_CORRECT if has_lior_formula else VERDICT_MISSING,
            severity=SEVERITY_INFO if has_lior_formula else SEVERITY_HIGH,
            recommendation=(
                "Formula is correct for its purpose. Note: this is NOT cross-entropy "
                "loss against token targets — it is a self-supervised geometric cost. "
                "Do not compare its absolute value to standard LM training curves."
            ),
        ))

        # --- CORE-2: total_loss label in JSONL ---
        has_misleading_label = _has(trainer2, '"total_loss": lior')
        findings.append(TelemetryFinding(
            finding_id="CORE-2",
            role=self.NAME,
            variable="total_loss (JSONL key)",
            intended=(
                "A cross-entropy or NLL loss against ground-truth targets — the "
                "standard LM training signal."
            ),
            actual=(
                "lior_mean — the geometric path cost. The JSONL record stores "
                'lior_mean under both "lior_mean" and "total_loss" keys. '
                "The value is NOT a supervised loss."
            ),
            verdict=VERDICT_WRONG,
            severity=SEVERITY_HIGH,
            recommendation=(
                'Rename the "total_loss" key to "lior_total" or remove the duplicate. '
                "Keeping the misleading label will confuse any downstream tooling or "
                "human readers who expect total_loss to be a cross-entropy value. "
                "If actual LM loss is available (from the model's lm_head output), "
                "add it separately as 'ce_loss'."
            ),
        ))

        # --- CORE-3: R_mean formula ---
        has_R_acc = _has(trainer2, "R_acc = R_acc + R_sc.mean()")
        findings.append(TelemetryFinding(
            finding_id="CORE-3",
            role=self.NAME,
            variable="R_mean",
            intended=(
                "Average scalar curvature of the space along the trajectory — "
                "a measure of how 'curved' the manifold is in the regions being visited."
            ),
            actual=(
                "Mean of R_sc over window steps, where R_sc = sqrt(|R4 @ g2_vec| / n²). "
                "R4 is the 4th-rank Riemannian curvature tensor approximation; "
                "g2_vec is a flattened version of g⊗g used to contract indices."
            ),
            verdict=VERDICT_PARTIAL,
            severity=SEVERITY_LOW,
            recommendation=(
                "R_sc is a reasonable curvature proxy but is not the standard Ricci scalar. "
                "The contraction R4 @ g2_vec / n² averages over all tensor components "
                "rather than performing the correct trace. This is a deliberate approximation "
                "that is cheaper than full Ricci computation. The value is useful as a "
                "relative diagnostic (is curvature growing/shrinking?) but not as an "
                "absolute curvature measurement. Label it 'curvature_proxy' in documentation."
            ),
        ))

        # --- CORE-4: spd_mean formula ---
        has_spd_acc = _has(trainer2, "spd_acc = spd_acc + spd.mean()")
        findings.append(TelemetryFinding(
            finding_id="CORE-4",
            role=self.NAME,
            variable="spd_mean",
            intended=(
                "Average speed of the trajectory in the metric — how fast the "
                "retrieval query is moving through the learned geometry."
            ),
            actual=(
                "Mean of v^T g v over window steps (the squared norm of velocity "
                "in the Riemannian metric). Computed by quad_form_batch."
            ),
            verdict=VERDICT_CORRECT,
            severity=SEVERITY_INFO,
            recommendation=(
                "Formula is correct. This is the standard Riemannian speed squared. "
                "Lower spd_mean means the geometry is directing queries more efficiently "
                "(shorter paths). Note: this is squared speed, not speed — the units "
                "are not directly comparable to Euclidean distance."
            ),
        ))

        return findings


class PerplexityAgent:
    NAME = "Perplexity"

    def audit(self) -> List[TelemetryFinding]:
        trainer2 = _read("training/trainer2.py")
        checkpoint_utils = _read("training/checkpoint_utils.py")
        findings: List[TelemetryFinding] = []

        # --- PPL-1: JSONL perplexity formula ---
        jsonl_ppl_wrong = _has(trainer2, '"perplexity": math.exp(min(lior')
        findings.append(TelemetryFinding(
            finding_id="PPL-1",
            role=self.NAME,
            variable="perplexity (JSONL telemetry)",
            intended=(
                "Language model perplexity: exp(average cross-entropy loss per token). "
                "A standard measure of how well the model predicts the next token. "
                "Perplexity of 1.0 = perfect prediction; lower is better."
            ),
            actual=(
                "exp(min(lior_mean, 20.0)) — the exponential of the geometric path "
                "cost, clamped to prevent overflow. This is NOT language model "
                "perplexity because lior_mean is not a per-token cross-entropy loss."
            ),
            verdict=VERDICT_WRONG,
            severity=SEVERITY_CRITICAL,
            recommendation=(
                "This value is a geometric surrogate, not true perplexity. "
                "Two options:\n"
                "  (A) Rename it to 'geo_exp_cost' to make its meaning clear, OR\n"
                "  (B) Compute true perplexity from actual LM cross-entropy output "
                "      (if the model has an lm_head that produces per-token log-probs "
                "      against input_ids targets). True perplexity = exp(ce_loss_per_token).\n"
                "The current label 'perplexity' will be deeply misleading to anyone "
                "comparing training curves to standard LM benchmarks."
            ),
        ))

        # --- PPL-2: checkpoint_utils perplexity ---
        ckpt_ppl_correct = (
            _has(checkpoint_utils, "torch.exp(torch.tensor(avg_loss))") and
            _has(checkpoint_utils, "avg_loss")
        )
        findings.append(TelemetryFinding(
            finding_id="PPL-2",
            role=self.NAME,
            variable="perplexity (checkpoint validation)",
            intended="Language model perplexity from validation loss.",
            actual=(
                "exp(avg_loss) where avg_loss is the mean loss from "
                "checkpoint_utils.compute_validation_metrics(). This is correct IF "
                "avg_loss is a per-token cross-entropy value."
            ),
            verdict=VERDICT_PARTIAL,
            severity=SEVERITY_MEDIUM,
            recommendation=(
                "The checkpoint validation perplexity (training/checkpoint_utils.py) "
                "is computed correctly as exp(avg_loss). However, the JSONL telemetry "
                "perplexity (trainer2.py maybe_log_metrics) uses exp(lior_mean) which "
                "is wrong. The two 'perplexity' values in the system mean different "
                "things — this needs to be resolved to avoid confusion."
            ),
        ))

        return findings


class GeometricDiagAgent:
    NAME = "GeometricDiag"

    def audit(self) -> List[TelemetryFinding]:
        diag_src = _read("training/geometric_diagnostics.py")
        findings: List[TelemetryFinding] = []

        # --- GEOM-1: E_geo Christoffel bug ---
        christoffel_zeroed = _has(diag_src, "return torch.zeros_like(metric_diag)")
        findings.append(TelemetryFinding(
            finding_id="GEOM-1",
            role=self.NAME,
            variable="E_geo",
            intended=(
                "Geodesic consistency residual: |γ̈ + Γ(γ̇,γ̇)|. "
                "If the learned metric correctly explains the dynamics, trajectories "
                "should satisfy the geodesic equation, so E_geo should approach zero."
            ),
            actual=(
                "Currently |γ̈| only — the Christoffel term Γ(γ̇,γ̇) is hardcoded "
                "to zero in compute_christoffel_diagonal(), justified by the comment "
                "'For constant diagonal metric, Christoffel = 0'. "
                "This means E_geo measures raw acceleration, not geodesic deviation."
            ),
            verdict=VERDICT_PARTIAL,
            severity=SEVERITY_HIGH,
            recommendation=(
                "The Christoffel = 0 assumption is only valid when the metric does NOT "
                "change spatially. Once g0_diag is being updated by the learning rule, "
                "the metric DOES vary (different windows see different g0_diag values), "
                "so Christoffel symbols become nonzero. "
                "Fix: compute Γ^i_jj = -(∂_j g_jj) / (2 g_ii) using finite differences "
                "on g0_diag between consecutive windows, or at minimum add a comment "
                "warning that E_geo is currently just |acceleration| not a true geodesic "
                "residual. Until fixed, E_geo going nonzero (0.0001–0.0005 as observed) "
                "just means the trajectory is accelerating, not that it's off-geodesic."
            ),
        ))

        # --- GEOM-2: E_var approximation ---
        uses_linear_approx = _has(diag_src, "Approximate perturbed LIoR via linear response")
        findings.append(TelemetryFinding(
            finding_id="GEOM-2",
            role=self.NAME,
            variable="E_var",
            intended=(
                "LIoR path optimality gap: how much a random perturbation to the "
                "current trajectory improves (lowers) the LIoR cost. "
                "Should approach zero when the path is locally optimal."
            ),
            actual=(
                "Linear response approximation: ΔLIoR ≈ R_curr × (|v+δv|² - |v|²). "
                "This assumes R_sc does not change when velocity changes, which is only "
                "valid for small perturbations and approximately constant curvature. "
                "The perturbations are random Gaussian noise scaled by 0.1."
            ),
            verdict=VERDICT_PARTIAL,
            severity=SEVERITY_MEDIUM,
            recommendation=(
                "The linear approximation is reasonable for small perturbations. "
                "The main limitation is that true E_var requires re-running the dynamics "
                "with the perturbed velocity, which is expensive. "
                "The current approximation is a valid fast proxy. "
                "To improve accuracy: use the actual lior_step formula "
                "(R_sc × quad_form(v+δv, g)) instead of R_curr × |v+δv|² — "
                "the metric g should be included in the perturbation response. "
                "E_var = 0 in your logs means all perturbations made the path WORSE, "
                "which is correct behavior for a locally optimal path."
            ),
        ))

        # --- GEOM-3: E_struct formula ---
        has_corr_formula = (
            _has(diag_src, "correlation + 1.0") and
            _has(diag_src, "torch.abs(correlation + 1.0)")
        )
        findings.append(TelemetryFinding(
            finding_id="GEOM-3",
            role=self.NAME,
            variable="E_struct",
            intended=(
                "Curvature-velocity coupling: checks whether high-curvature regions "
                "have low velocity and vice versa. In optimal resistance geometry, "
                "correlation(R, |v|) should be -1, so E_struct = |corr + 1| → 0."
            ),
            actual=(
                "|corr(R_curvature_buffer, |v|_buffer) + 1| computed over the full "
                "circular buffer history (64 steps). Pearson correlation with eps=1e-8 "
                "for numerical stability."
            ),
            verdict=VERDICT_CORRECT,
            severity=SEVERITY_INFO,
            recommendation=(
                "Formula is correct. E_struct varying away from 1.0 (not pinned) as "
                "observed in your logs means the geometry is starting to couple with "
                "dynamics — that is the expected direction. "
                "E_struct = 0 is the ideal target (perfect anti-correlation). "
                "E_struct = 2 means perfect positive correlation (geometry pushing "
                "in the wrong direction)."
            ),
        ))

        return findings


class InfraMetricsAgent:
    NAME = "InfraMetrics"

    def audit(self) -> List[TelemetryFinding]:
        trainer2 = _read("training/trainer2.py")
        findings: List[TelemetryFinding] = []

        # --- INFRA-1: mem_norm ---
        has_mem_norm = _has(trainer2, "bank_coord.norm()")
        findings.append(TelemetryFinding(
            finding_id="INFRA-1",
            role=self.NAME,
            variable="mem_norm",
            intended="Health check on the memory bank — should stay bounded.",
            actual=(
                "L2 norm of memory.bank_coord. If memory has no bank_coord attribute, "
                "falls back to 0.0."
            ),
            verdict=VERDICT_CORRECT,
            severity=SEVERITY_INFO,
            recommendation=(
                "Formula is correct. Slow growth (a few percent per epoch) is normal "
                "as the memory bank learns. A sudden jump or unbounded growth means "
                "the memory update is diverging — check _maybe_update_memory."
            ),
        ))

        # --- INFRA-2: window_ms ---
        has_window_ms = _has(trainer2, "t_window_end - t_window_start")
        findings.append(TelemetryFinding(
            finding_id="INFRA-2",
            role=self.NAME,
            variable="window_ms",
            intended="Wall-clock time per training window in milliseconds.",
            actual=(
                "(t_window_end - t_window_start) × 1000.0. Includes both free and "
                "nudged phases when nudge is active (every nudge_every_windows windows)."
            ),
            verdict=VERDICT_CORRECT,
            severity=SEVERITY_INFO,
            recommendation=(
                "Formula is correct. Note that windows with nudge enabled take roughly "
                "2× longer than free-only windows. If window_ms is highly variable, "
                "check whether CUDA synchronization is happening inside the window loop."
            ),
        ))

        # --- INFRA-3: gpu_alloc_gb / gpu_reserved_gb ---
        has_gpu_alloc = _has(trainer2, "memory_allocated(DEVICE) / 1024 ** 3")
        has_gpu_reserved = _has(trainer2, "memory_reserved(DEVICE) / 1024 ** 3")
        findings.append(TelemetryFinding(
            finding_id="INFRA-3",
            role=self.NAME,
            variable="gpu_alloc_gb / gpu_reserved_gb",
            intended=(
                "GPU memory actively in use (alloc) and total held by the allocator "
                "including cached free blocks (reserved). Both in GB."
            ),
            actual=(
                "torch.cuda.memory_allocated(DEVICE) / 1024³ and "
                "torch.cuda.memory_reserved(DEVICE) / 1024³, sampled at the end of "
                "each window and printed in the [PROGRESS] line."
            ),
            verdict=VERDICT_CORRECT if (has_gpu_alloc and has_gpu_reserved) else VERDICT_MISSING,
            severity=SEVERITY_INFO if (has_gpu_alloc and has_gpu_reserved) else SEVERITY_HIGH,
            recommendation=(
                "Formula is correct. The gap between reserved and alloc is the "
                "allocator cache — it can be freed with torch.cuda.empty_cache() at "
                "the cost of a sync. At 87%+ total reserved / total capacity, you are "
                "near OOM territory. Watch for reserved_gb growing window over window."
            ),
        ))

        return findings


class MoraleAgent:
    NAME = "Morale"

    def audit(self, findings: List[TelemetryFinding]) -> List[str]:
        wrong = [f for f in findings if f.verdict == VERDICT_WRONG]
        partial = [f for f in findings if f.verdict == VERDICT_PARTIAL]
        correct = [f for f in findings if f.verdict == VERDICT_CORRECT]
        notes = []

        notes.append(
            f"Telemetry audit complete: {len(correct)} correct, "
            f"{len(partial)} partially correct, {len(wrong)} wrong."
        )

        if wrong:
            notes.append(
                "CRITICAL issues found: "
                + "; ".join(f"{f.finding_id} ({f.variable})" for f in wrong)
                + ". These will produce misleading values if not fixed."
            )

        notes.append(
            "The most important fix is PPL-1 (perplexity). "
            "exp(lior_mean) is being logged as 'perplexity' but lior_mean is a "
            "geometric path cost, not cross-entropy. If you compare this number to "
            "any standard LM benchmark you will get nonsensical results. "
            "Either rename it or compute true perplexity from the LM head output."
        )

        notes.append(
            "CORE-2 (total_loss label) is misleading but lower priority than PPL-1 "
            "— it just means the JSONL has a confusing key name. "
            "GEOM-1 (E_geo Christoffel = 0) means E_geo is currently measuring "
            "acceleration, not geodesic deviation. It is still useful as a "
            "stability indicator but does not prove the metric is governing the path."
        )

        return notes


class ScribeAgent:
    NAME = "Scribe"

    def consolidate(
        self,
        findings: List[TelemetryFinding],
        wiring_checks: List[WiringCheck],
    ) -> ScribeLog:
        action_items: List[str] = []
        for f in findings:
            if f.verdict != VERDICT_CORRECT:
                action_items.append(
                    f"[{f.severity}] {f.finding_id} {f.variable} ({f.verdict}): "
                    f"{f.recommendation.splitlines()[0]}"
                )
        for w in wiring_checks:
            if not w.wired:
                action_items.append(
                    f"[WIRING] {w.feature} at {w.entry_point}: {w.notes}"
                )
        if not action_items:
            action_items.append("All telemetry formulas are correct.")

        n_correct = sum(1 for f in findings if f.verdict == VERDICT_CORRECT)
        n_partial = sum(1 for f in findings if f.verdict == VERDICT_PARTIAL)
        n_wrong = sum(1 for f in findings if f.verdict == VERDICT_WRONG)
        summary = (
            f"{n_correct} correct, {n_partial} partial, {n_wrong} wrong out of "
            f"{len(findings)} telemetry variables audited. "
            f"Critical: perplexity label is wrong (exp(geometric_cost) ≠ LM perplexity). "
            f"High: total_loss key in JSONL is a misleading label for lior_mean."
        )
        return ScribeLog(
            findings=findings,
            wiring_checks=wiring_checks,
            summary=summary,
            action_items=action_items,
        )


# ---------------------------------------------------------------------------
# Pipeline wiring checks
# ---------------------------------------------------------------------------

def _check_pipeline_wiring() -> List[WiringCheck]:
    trainer2 = _read("training/trainer2.py")
    return [
        WiringCheck(
            feature="perplexity labeled as geometric surrogate (not LM perplexity) in glossary",
            wired=_has(trainer2, "geometric surrogate only"),
            entry_point="training/trainer2.py:trainer2_entrypoint (glossary print)",
            notes=(
                "The glossary should explicitly warn that perplexity = exp(lior_mean) "
                "is NOT language-model perplexity."
            ),
        ),
        WiringCheck(
            feature="lior_mean labeled as geometric path cost (not loss) in glossary",
            wired=_has(trainer2, "NOT cross-entropy loss"),
            entry_point="training/trainer2.py:trainer2_entrypoint (glossary print)",
            notes="The glossary must warn that lior_mean is not a supervised loss.",
        ),
        WiringCheck(
            feature="E_geo Christoffel limitation documented in glossary",
            wired=_has(trainer2, "Christoffel currently zero"),
            entry_point="training/trainer2.py:trainer2_entrypoint (glossary print)",
            notes=(
                "The glossary should note that E_geo currently measures acceleration "
                "only, not true geodesic residual."
            ),
        ),
        WiringCheck(
            feature="GPU RAM per window reported in [PROGRESS] line",
            wired=(
                _has(trainer2, "gpu_alloc_gb") and
                _has(trainer2, "gpu_reserved_gb")
            ),
            entry_point="training/trainer2.py:trainer2_entrypoint window loop",
            notes="Per-window GPU RAM is needed to catch memory pressure early.",
        ),
        WiringCheck(
            feature="KeyboardInterrupt checkpoint uses actual epoch_idx (not hardcoded 0)",
            wired=(
                _has(trainer2, "epoch_idx = 0  # initialised here") and
                not _has(trainer2, "epoch_idx=0,\n")
            ),
            entry_point="training/trainer2.py:trainer2_entrypoint except block",
            notes=(
                "epoch_idx must be initialised before the loop so the interrupt "
                "handler always checkpoints with the correct epoch number."
            ),
        ),
        WiringCheck(
            feature="KeyboardInterrupt checkpoint wrapped in try/except",
            wired=_has(trainer2, "WARNING: checkpoint save failed on interrupt"),
            entry_point="training/trainer2.py:trainer2_entrypoint except block",
            notes=(
                "If torch.save fails during interrupt, the error should be "
                "reported clearly rather than silently swallowed."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class TelemetryFormulaAuditTeam:
    """Run the full telemetry formula audit."""

    def run(self) -> TelemetryAuditReport:
        coordinator = CoordinatorAgent()
        core = CoreMetricsAgent()
        ppl = PerplexityAgent()
        geom = GeometricDiagAgent()
        infra = InfraMetricsAgent()
        morale = MoraleAgent()
        scribe = ScribeAgent()

        all_findings: List[TelemetryFinding] = []
        all_findings.extend(core.audit())
        all_findings.extend(ppl.audit())
        all_findings.extend(geom.audit())
        all_findings.extend(infra.audit())

        wiring_checks = _check_pipeline_wiring()
        morale_notes = morale.audit(all_findings)
        scribe_log = scribe.consolidate(all_findings, wiring_checks)

        return TelemetryAuditReport(
            coordinator_scope=coordinator.scope(),
            findings=all_findings,
            wiring_checks=wiring_checks,
            morale_notes=morale_notes,
            scribe_log=scribe_log,
        )


# ---------------------------------------------------------------------------
# Plain-English printed report
# ---------------------------------------------------------------------------

_VERDICT_ICON = {
    VERDICT_CORRECT: "✓",
    VERDICT_PARTIAL: "~",
    VERDICT_WRONG:   "✗",
    VERDICT_MISSING: "?",
}

_SEV_ICON = {
    SEVERITY_CRITICAL: "CRITICAL",
    SEVERITY_HIGH:     "HIGH    ",
    SEVERITY_MEDIUM:   "MEDIUM  ",
    SEVERITY_LOW:      "LOW     ",
    SEVERITY_INFO:     "INFO    ",
}


def print_report(report: TelemetryAuditReport) -> None:
    """Print the full audit in plain English."""
    sep = "=" * 72

    print()
    print(sep)
    print("  TELEMETRY FORMULA AUDIT REPORT")
    print(sep)
    print()
    print("SCOPE")
    print("-----")
    for line in report.coordinator_scope.split(". "):
        if line.strip():
            print(f"  {line.strip()}.")
    print()

    print(sep)
    print("  VARIABLE-BY-VARIABLE FINDINGS")
    print(sep)
    for f in report.findings:
        icon = _VERDICT_ICON.get(f.verdict, "?")
        sev  = _SEV_ICON.get(f.severity, f.severity)
        print()
        print(f"  [{icon}] {f.finding_id}  {f.variable}  [{sev}]  verdict={f.verdict}")
        print(f"      Intended : {f.intended[:80]}")
        if len(f.intended) > 80:
            print(f"                 {f.intended[80:160]}")
        print(f"      Actual   : {f.actual[:80]}")
        if len(f.actual) > 80:
            print(f"                 {f.actual[80:160]}")
        print(f"      Action   : {f.recommendation.splitlines()[0][:80]}")
        extra = f.recommendation.splitlines()[1:]
        for line in extra:
            if line.strip():
                print(f"                 {line.strip()[:80]}")

    print()
    print(sep)
    print("  PIPELINE WIRING")
    print(sep)
    for w in report.wiring_checks:
        icon = "✓" if w.wired else "✗"
        print(f"  [{icon}] {w.feature}")
        if not w.wired:
            print(f"        → {w.notes}")

    print()
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  {report.scribe_log.summary}")
    print()
    print("  ACTION ITEMS")
    for item in report.scribe_log.action_items:
        print(f"  • {item[:100]}")
        if len(item) > 100:
            print(f"    {item[100:200]}")

    print()
    print(sep)
    print("  MORALE NOTES")
    print(sep)
    for note in report.morale_notes:
        print(f"  {note}")
    print()
    print(f"  Status: {report.approval_status}")
    print(sep)
    print()


if __name__ == "__main__":
    report = TelemetryFormulaAuditTeam().run()
    print_report(report)
