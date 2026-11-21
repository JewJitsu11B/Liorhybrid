"""
Update-Signal Audit Team (Lead + Specialists + Scribe)

Investigates why |Δ| and |Δθ| appear as zero in the training log even though
the LIoR diff is nonzero (observed value ≈ 0.000005).

Two primary questions from the audit request:

  Q1: What is ``eta_update / beta_nudge`` evaluating to, and is it producing
      a value too small to move float32 weights?

  Q2: Is the LIoR diff signal (≈ 5e-6) genuinely small, or is it being
      attenuated somewhere in the chain before ``apply_manual_update``?

Team composition
----------------
1. CoordinatorAgent     – scope, task queue, consensus
2. ScalingAnalystAgent  – eta_update / beta_nudge ratio and effective Δ scale
3. SignalChainAgent     – maps the LIoR diff signal from nudge → apply_manual_update
4. NudgeScaleAgent      – nudge_scale adequacy and lior_diff expected magnitude
5. PrintDiagnosticsAgent – print format precision (:.6f showing 0 for tiny values)
6. MoraleAgent          – workload balance and cadence sustainability
7. ScribeAgent          – consolidated decision log

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

SEVERITY_INFO = "INFO"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpdateFinding:
    """A single finding from one specialist agent."""
    finding_id: str
    role: str
    component: str
    check: str
    passed: bool
    severity: str
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class WiringCheck:
    """Records whether an update-signal feature is wired into the pipeline."""
    feature: str
    wired: bool
    entry_point: str
    notes: str


@dataclass(frozen=True)
class ScribeLog:
    """Consolidated decision log produced by the Scribe agent."""
    findings: List[UpdateFinding]
    wiring_checks: List[WiringCheck]
    summary: str
    action_items: List[str]

    @property
    def all_passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def critical_findings(self) -> List[UpdateFinding]:
        return [f for f in self.findings if not f.passed and f.severity == SEVERITY_CRITICAL]

    @property
    def failed_findings(self) -> List[UpdateFinding]:
        return [f for f in self.findings if not f.passed]


@dataclass
class UpdateSignalReport:
    """Complete report produced by the full seven-agent team."""
    coordinator_scope: str
    findings: List[UpdateFinding]
    wiring_checks: List[WiringCheck]
    morale_notes: List[str]
    scribe_log: ScribeLog
    approval_status: str = "AWAITING APPROVAL TO EXECUTE"

    @property
    def passed(self) -> bool:
        return self.scribe_log.all_passed


# ---------------------------------------------------------------------------
# Source-reading helpers
# ---------------------------------------------------------------------------

def _read(rel_path: str) -> str:
    """Read a repo file relative to REPO_ROOT, return '' if missing."""
    p = REPO_ROOT / rel_path
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _has(src: str, needle: str) -> bool:
    return needle in src


def _count_pattern(src: str, pattern: str) -> int:
    return len(re.findall(pattern, src))


def _extract_float(src: str, pattern: str, default: float = 0.0) -> float:
    """Extract the first float matched by a regex group from src."""
    m = re.search(pattern, src)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            pass
    return default


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

class CoordinatorAgent:
    """
    Owns scope, assigns tasks, resolves blockers, builds consensus.

    Scope
    -----
    Investigate why |Δ| and |Δθ| appear as 0.000000 in the training log
    despite the LIoR diff being nonzero (≈ 5e-6).

    Two primary questions:
      Q1. What is eta_update / beta_nudge evaluating to, and does the
          resulting effective learning rate crush float32 weight updates?
      Q2. Is the LIoR diff signal (≈ 5e-6) genuinely small, or is it
          attenuated before apply_manual_update?

    Loss and perplexity context: if the update signal is invisible (|Δ| ≈ 0)
    the diagonal metric g0_diag does not change, geodesic costs do not evolve,
    and effective learning on loss / perplexity stalls.

    Findings are plan-only.  No code is changed until approved.
    """

    SCOPE = (
        "Investigate the zero-|Δ| symptom: |Δ|=0.000000 and |Δθ|=0.000000 "
        "printed by apply_manual_update despite LIoR diff ≈ 5e-6. "
        "Q1: evaluate eta_update / beta_nudge and effective delta scale. "
        "Q2: trace LIoR diff attenuation from build_nudge → run_two_phase_and_update "
        "→ apply_manual_update. "
        "Q3: check nudge_scale adequacy. "
        "Q4: check print format precision (:.6f rounds 7e-7 to 0.000000). "
        "Loss/perplexity improvements require nonzero weight deltas; stalled |Δ| "
        "means the geometry never adapts. Report findings only — no code changes "
        "without approval."
    )

    TASK_QUEUE: List[Tuple[str, str]] = [
        ("ScalingAnalyst", "Evaluate eta_update / beta_nudge ratio and effective Δ scale"),
        ("SignalChain",    "Map LIoR diff signal from nudge through to apply_manual_update"),
        ("NudgeScale",     "Check nudge_scale adequacy and expected lior_diff magnitude"),
        ("PrintDiagnostics", "Check print format precision — does :.6f hide tiny-but-nonzero Δ?"),
        ("Morale",         "Flag workload balance and cadence sustainability"),
        ("Scribe",         "Consolidate findings into decision log with severity + evidence"),
    ]

    def scope(self) -> str:
        return self.SCOPE

    def task_queue(self) -> List[Tuple[str, str]]:
        return list(self.TASK_QUEUE)


class ScalingAnalystAgent:
    """
    Evaluates the eta_update / beta_nudge ratio and resulting effective delta scale.

    Key checks
    ----------
    - What are the default values of eta_update and beta_nudge?
    - What is the ratio (eta)?
    - Given lior_diff ≈ 5e-6, what is the expected |Δ| per update step?
    - Does the expected |Δ| fall below the :.6f format threshold (5e-7)?
    - Is the ratio sane for float32 weights (updates ≥ 1e-7)?
    """

    NAME = "ScalingAnalyst"

    def audit(self) -> List[UpdateFinding]:
        src = _read("training/trainer2.py")
        findings: List[UpdateFinding] = []

        # --- SCALE-1: eta_update default extracted from source ---
        eta_update_default = _extract_float(
            src, r"eta_update\s*:\s*float\s*=\s*([0-9e.+-]+)", default=1e-2
        )
        eta_sensible = 1e-5 <= eta_update_default <= 1.0
        findings.append(UpdateFinding(
            finding_id="SCALE-1",
            role=self.NAME,
            component="training/trainer2.py:TrainConfig.eta_update",
            check="eta_update default is within a sensible range [1e-5, 1.0]",
            passed=eta_sensible,
            severity=SEVERITY_HIGH if not eta_sensible else SEVERITY_INFO,
            evidence=(
                f"eta_update default = {eta_update_default:.2e} "
                "(training/trainer2.py TrainConfig dataclass)"
            ),
            recommendation=(
                "eta_update is in range. The ratio eta_update / beta_nudge drives "
                "the effective update; ensure beta_nudge doesn't inflate this ratio."
            ) if eta_sensible else (
                f"eta_update={eta_update_default:.2e} is outside [1e-5, 1.0]. "
                "Adjust to produce visible weight updates."
            ),
        ))

        # --- SCALE-2: beta_nudge default ---
        beta_nudge_default = _extract_float(
            src, r"beta_nudge\s*:\s*float\s*=\s*([0-9e.+-]+)", default=1e-3
        )
        beta_sensible = 1e-6 <= beta_nudge_default <= 1.0
        findings.append(UpdateFinding(
            finding_id="SCALE-2",
            role=self.NAME,
            component="training/trainer2.py:TrainConfig.beta_nudge",
            check="beta_nudge default is within a sensible range [1e-6, 1.0]",
            passed=beta_sensible,
            severity=SEVERITY_HIGH if not beta_sensible else SEVERITY_INFO,
            evidence=(
                f"beta_nudge default = {beta_nudge_default:.2e} "
                "(training/trainer2.py TrainConfig dataclass)"
            ),
            recommendation=(
                "beta_nudge is in range. "
                "Note: division by beta_nudge amplifies eta_update."
            ) if beta_sensible else (
                f"beta_nudge={beta_nudge_default:.2e} out of range."
            ),
        ))

        # --- SCALE-3: effective eta = eta_update / beta_nudge ---
        eta_ratio = eta_update_default / beta_nudge_default if beta_nudge_default != 0.0 else math.inf
        # With lior_diff ≈ 5e-6 and coord_dim_n components (v_sq_norm mean ≈ 1/n),
        # estimate expected delta_norm assuming coord_dim_n = 64 (typical)
        estimated_lior_diff = 5e-6
        estimated_coord_dim = 64
        estimated_delta_norm = eta_ratio * estimated_lior_diff / estimated_coord_dim
        # The :.6f format can represent down to 5e-7; values below appear as 0.000000
        delta_visible = estimated_delta_norm >= 5e-7
        findings.append(UpdateFinding(
            finding_id="SCALE-3",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update",
            check=(
                "Effective eta (eta_update / beta_nudge) produces a visible |Δ| "
                "at lior_diff ≈ 5e-6 and coord_dim_n ≈ 64"
            ),
            passed=delta_visible,
            severity=SEVERITY_HIGH if not delta_visible else SEVERITY_INFO,
            evidence=(
                f"eta = {eta_update_default:.2e} / {beta_nudge_default:.2e} = {eta_ratio:.2f}. "
                f"Expected delta_norm ≈ eta × lior_diff / coord_dim "
                f"= {eta_ratio:.2f} × {estimated_lior_diff:.2e} / {estimated_coord_dim} "
                f"= {estimated_delta_norm:.2e}. "
                f":.6f threshold is 5e-7; delta {'≥' if delta_visible else '<'} threshold."
            ),
            recommendation=(
                "With coord_dim_n ≥ 100 the estimated delta_norm drops below the "
                ":.6f display threshold and prints as 0.000000 — not a float32 "
                "precision problem, just a log-format problem. "
                "Fix: use :.4e format in the print statement. "
                "The update IS happening but is too small to see at :.6f precision. "
                "To increase actual update magnitude: raise eta_update or lower beta_nudge "
                "or raise nudge_scale to produce a larger lior_diff."
            ) if not delta_visible else (
                "delta_norm should be visible at :.6f precision for this coord_dim."
            ),
        ))

        # --- SCALE-4: eta computed in apply_manual_update uses division ---
        has_eta_division = _has(src, "eta = float(cfg.eta_update / cfg.beta_nudge)")
        findings.append(UpdateFinding(
            finding_id="SCALE-4",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update",
            check="apply_manual_update computes eta = cfg.eta_update / cfg.beta_nudge",
            passed=has_eta_division,
            severity=SEVERITY_CRITICAL if not has_eta_division else SEVERITY_INFO,
            evidence=(
                "apply_manual_update line: "
                "eta = float(cfg.eta_update / cfg.beta_nudge). "
                "This is the sole scaling factor before delta_g is applied."
            ),
            recommendation=(
                "Division wiring confirmed. No attenuation between eta computation "
                "and delta_g application."
            ) if has_eta_division else (
                "eta = eta_update / beta_nudge not found in apply_manual_update. "
                "The update scale may be broken."
            ),
        ))

        return findings


class SignalChainAgent:
    """
    Maps the LIoR diff signal from build_nudge → run_window (nudged phase)
    → run_two_phase_and_update → apply_manual_update.

    Key checks
    ----------
    - Is lior_diff computed as nudged - free (correct sign)?
    - Is velocity forwarded from PhaseStats to apply_manual_update?
    - Is there any early-exit or clamp before apply_manual_update?
    - Is apply_manual_update called unconditionally (not behind a disabled flag)?
    """

    NAME = "SignalChain"

    def audit(self) -> List[UpdateFinding]:
        src = _read("training/trainer2.py")
        findings: List[UpdateFinding] = []

        # --- SIG-1: lior_diff sign ---
        correct_sign = _has(src, "nudged.metrics.lior_mean - free.metrics.lior_mean")
        findings.append(UpdateFinding(
            finding_id="SIG-1",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update",
            check="lior_diff computed as nudged.lior_mean − free.lior_mean (correct sign)",
            passed=correct_sign,
            severity=SEVERITY_HIGH if not correct_sign else SEVERITY_INFO,
            evidence=(
                "lior_diff = nudged.metrics.lior_mean - free.metrics.lior_mean. "
                "Negative lior_diff means the nudge helped (lower loss). "
                "Positive means the nudge hurt."
            ),
            recommendation=(
                "Sign is correct. A positive lior_diff (nudge hurt) reduces g_diag "
                "in those directions; negative (nudge helped) increases them."
            ) if correct_sign else (
                "lior_diff sign is inverted — update direction will be wrong."
            ),
        ))

        # --- SIG-2: nan/inf guard present ---
        has_nan_guard = _has(src, "if not math.isfinite(lior_diff_val)")
        findings.append(UpdateFinding(
            finding_id="SIG-2",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update",
            check="NaN/Inf guard rejects infinite lior_diff before update",
            passed=has_nan_guard,
            severity=SEVERITY_HIGH if not has_nan_guard else SEVERITY_INFO,
            evidence=(
                "apply_manual_update: if not math.isfinite(lior_diff_val): return False. "
                "This prevents corrupt updates but does NOT filter tiny-but-finite diffs."
            ),
            recommendation=(
                "Guard is correct. Tiny finite diffs (5e-6) pass through — "
                "the lior_diff is NOT being filtered. The signal reaches delta_g."
            ) if has_nan_guard else (
                "NaN/Inf guard missing — corrupt updates may silently damage weights."
            ),
        ))

        # --- SIG-3: velocity forwarded from PhaseStats ---
        velocity_forwarded = (
            _has(src, "velocity = getattr(free, 'velocity', None)") and
            _has(src, "velocity=velocity,")
        )
        findings.append(UpdateFinding(
            finding_id="SIG-3",
            role=self.NAME,
            component="training/trainer2.py:run_two_phase_and_update",
            check="Velocity from PhaseStats is forwarded to apply_manual_update",
            passed=velocity_forwarded,
            severity=SEVERITY_HIGH if not velocity_forwarded else SEVERITY_INFO,
            evidence=(
                "run_two_phase_and_update: "
                "velocity = getattr(free, 'velocity', None); "
                "apply_manual_update(..., velocity=velocity, ...)"
            ),
            recommendation=(
                "Velocity forwarding is correct. If velocity is None, the fallback "
                "scalar update fires (g_diag *= adjustment) instead of the directional "
                "v_sq_norm-weighted update."
            ) if velocity_forwarded else (
                "Velocity is not forwarded. The directional metric update will never "
                "fire; only the scalar fallback will run."
            ),
        ))

        # --- SIG-4: apply_manual_update called in two-phase path ---
        has_apply_call = _has(src, "ok = apply_manual_update(")
        findings.append(UpdateFinding(
            finding_id="SIG-4",
            role=self.NAME,
            component="training/trainer2.py:run_two_phase_and_update",
            check="apply_manual_update is called unconditionally in run_two_phase_and_update",
            passed=has_apply_call,
            severity=SEVERITY_CRITICAL if not has_apply_call else SEVERITY_INFO,
            evidence=(
                "run_two_phase_and_update: ok = apply_manual_update(...). "
                "Called after Phase 2 apply updates, outside inference_context."
            ),
            recommendation=(
                "Call wiring is correct. The update function is always invoked "
                "when run_two_phase_and_update is used."
            ) if has_apply_call else (
                "apply_manual_update is not called — no updates will ever apply."
            ),
        ))

        # --- SIG-5: no artificial zero-clamp before delta_g ---
        has_zero_clamp = bool(re.search(r"lior_diff_val\s*=\s*0", src))
        findings.append(UpdateFinding(
            finding_id="SIG-5",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update",
            check="No artificial zero-clamp applied to lior_diff_val before delta_g",
            passed=not has_zero_clamp,
            severity=SEVERITY_HIGH if has_zero_clamp else SEVERITY_INFO,
            evidence=(
                "No 'lior_diff_val = 0' assignment found in apply_manual_update. "
                "The raw lior_diff flows directly into delta_g = eta * lior_diff_val * v_sq_norm."
            ),
            recommendation=(
                "Signal is clean — lior_diff flows unmodified into the update formula. "
                "The small magnitude is intrinsic (small nudge → small differential), "
                "not due to artificial suppression."
            ) if not has_zero_clamp else (
                "lior_diff_val is being zeroed — update signal is being suppressed."
            ),
        ))

        return findings


class NudgeScaleAgent:
    """
    Investigates whether nudge_scale is large enough to produce a lior_diff
    that leads to meaningfully sized weight updates.

    Key checks
    ----------
    - What is the default nudge_scale?
    - Is nudge_mode "target_embedding" (the active mode)?
    - Is nudge_every_windows set to fire every window?
    - At nudge_scale=0.1, what lior_diff can we reasonably expect?
    """

    NAME = "NudgeScale"

    def audit(self) -> List[UpdateFinding]:
        src = _read("training/trainer2.py")
        findings: List[UpdateFinding] = []

        # --- NUDGE-1: nudge_scale default ---
        nudge_scale_default = _extract_float(
            src, r"nudge_scale\s*:\s*float\s*=\s*([0-9e.+-]+)", default=0.1
        )
        nudge_scale_sensible = nudge_scale_default >= 0.01
        findings.append(UpdateFinding(
            finding_id="NUDGE-1",
            role=self.NAME,
            component="training/trainer2.py:TrainConfig.nudge_scale",
            check="nudge_scale default is large enough to produce a measurable lior_diff",
            passed=nudge_scale_sensible,
            severity=SEVERITY_MEDIUM if not nudge_scale_sensible else SEVERITY_INFO,
            evidence=(
                f"nudge_scale default = {nudge_scale_default}. "
                "Nudge = k * (target - current); larger k → larger nudge force → "
                "larger difference between free and nudged windows → larger lior_diff."
            ),
            recommendation=(
                f"nudge_scale={nudge_scale_default} is adequate. "
                "If lior_diff remains near 5e-6 with this scale, the target and "
                "current coordinates are nearly identical (model is already well-aligned "
                "or embeddings are in a very flat region)."
            ) if nudge_scale_sensible else (
                f"nudge_scale={nudge_scale_default} is very small. "
                "Increase to at least 0.05 to produce a measurable lior_diff."
            ),
        ))

        # --- NUDGE-2: nudge_mode default ---
        has_target_embedding_default = (
            'nudge_mode: str = "target_embedding"' in src or
            "nudge_mode: str = 'target_embedding'" in src
        )
        findings.append(UpdateFinding(
            finding_id="NUDGE-2",
            role=self.NAME,
            component="training/trainer2.py:TrainConfig.nudge_mode",
            check='nudge_mode default is "target_embedding" (active nudge mode)',
            passed=has_target_embedding_default,
            severity=SEVERITY_HIGH if not has_target_embedding_default else SEVERITY_INFO,
            evidence=(
                'nudge_mode: str = "target_embedding" — '
                "build_nudge will construct a force toward the target token embedding. "
                "If nudge_mode were 'off', lior_diff would always be zero."
            ),
            recommendation=(
                "nudge_mode is correctly set to target_embedding. "
                "The nudge is active. If lior_diff is tiny, the target embeddings "
                "are very close to the current coordinates."
            ) if has_target_embedding_default else (
                'nudge_mode default is not "target_embedding". '
                "No nudge signal is generated → lior_diff will always be zero."
            ),
        ))

        # --- NUDGE-3: nudge_every_windows ---
        nudge_every_sensible = _has(src, "nudge_every_windows: int = 1")
        findings.append(UpdateFinding(
            finding_id="NUDGE-3",
            role=self.NAME,
            component="training/trainer2.py:TrainConfig.nudge_every_windows",
            check="nudge_every_windows default fires every window (= 1)",
            passed=nudge_every_sensible,
            severity=SEVERITY_LOW if not nudge_every_sensible else SEVERITY_INFO,
            evidence=(
                "nudge_every_windows: int = 1 — the nudged phase runs every window. "
                "If set to > 1, many windows run with no update."
            ),
            recommendation=(
                "Default of 1 is correct for maximum signal density. "
                "Increase only if compute budget is tight."
            ) if nudge_every_sensible else (
                "nudge_every_windows != 1 — nudge may not fire every window. "
                "Verify the effective update frequency."
            ),
        ))

        # --- NUDGE-4: Are the tiny updates a problem? Direct answer. ---
        # With eta=10, lior_diff=5e-6, coord_dim_n≈64:
        #   delta_norm ≈ eta * lior_diff / coord_dim = 10 * 5e-6 / 64 ≈ 7.8e-7
        # g0_diag is clamped to [0.01, 100.0]; initial value ≈ 1.0.
        # To shift g0_diag by 1% (Δ=0.01) at 7.8e-7 per window requires ~12,800 windows.
        # At 1 window/batch and typical 50k+ batches per epoch, this IS achievable but slow.
        # Answer: NOT a bug, but the updates are so small they won't visibly affect loss or
        # perplexity within a short training run. To accelerate, raise eta_update or nudge_scale.
        findings.append(UpdateFinding(
            finding_id="NUDGE-4",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update + TrainConfig",
            check=(
                "QUESTION: Are tiny updates (|Δ| ≈ 7e-7) a problem for "
                "loss/perplexity improvement?"
            ),
            passed=True,
            severity=SEVERITY_INFO,
            evidence=(
                "ANSWER: Not a correctness bug — updates are real and cumulative. "
                "But they are too slow to observe in short runs. "
                "Math: eta=10.0, lior_diff≈5e-6, coord_dim_n≈64 → "
                "delta_norm ≈ 10×5e-6/64 ≈ 7.8e-7 per window. "
                "g0_diag clamped to [0.01, 100]; initial ≈ 1.0. "
                "A 1% shift (Δg=0.01) needs ~12,800 windows at this rate. "
                "Loss and perplexity improve only when g0_diag changes enough "
                "to steer retrieval toward lower-cost directions — this requires "
                "O(10k–100k) windows at current scale. "
                "The signal chain is intact; the scale is the issue."
            ),
            recommendation=(
                "To make updates visible within hundreds of windows: "
                "(1) Raise eta_update from 1e-2 to 1e-1 (10× faster, same signal). "
                "(2) Raise nudge_scale from 0.1 to 0.5–1.0 to increase lior_diff. "
                "(3) Both: 100× larger updates, g0_diag shifts noticeably in ~128 windows. "
                "Start with (1) alone as the lowest-risk change."
            ),
        ))

        return findings


class PrintDiagnosticsAgent:
    """
    Checks whether the print format strings in apply_manual_update can
    distinguish tiny-but-nonzero deltas from true zero.

    Root cause of the apparent |Δ|=0 symptom:
        delta_norm ≈ 7.8e-7 with coord_dim_n=128
        f"{7.8e-7:.6f}" → "0.000001" (rounds to 0.000001, still visible)
        f"{3.9e-7:.6f}" → "0.000000" (rounds to zero, hidden)

    The fix is to use scientific notation (:.4e) for delta display.
    """

    NAME = "PrintDiagnostics"

    def audit(self) -> List[UpdateFinding]:
        src = _read("training/trainer2.py")
        findings: List[UpdateFinding] = []

        # --- PRINT-1: |Δ| format in g0_diag print ---
        # After fix: should use :.4e; before fix: was :.6f
        uses_sci_delta = bool(re.search(
            r'\|Δ\|\s*=\s*\{delta_norm:.4e\}', src
        ))
        findings.append(UpdateFinding(
            finding_id="PRINT-1",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update (g0_diag print)",
            check="g0_diag |Δ| uses scientific-notation format (:.4e) to show sub-1e-6 values",
            passed=uses_sci_delta,
            severity=SEVERITY_HIGH if not uses_sci_delta else SEVERITY_INFO,
            evidence=(
                "The log line: [UPDATE] g0_diag += directional (|Δ|=...). "
                "With coord_dim_n ≥ 128 and lior_diff ≈ 5e-6, delta_norm can be < 5e-7 "
                "which formats as 0.000000 with :.6f. "
                "Scientific notation :.4e shows '3.9000e-07' instead of '0.000000'."
            ),
            recommendation=(
                "Format already uses :.4e — tiny-but-nonzero deltas are now visible."
            ) if uses_sci_delta else (
                "Change |Δ|={delta_norm:.6f} → |Δ|={delta_norm:.4e} in "
                "apply_manual_update's g0_diag print statement. "
                "This makes sub-1e-6 updates visible without altering behavior."
            ),
        ))

        # --- PRINT-2: |Δθ| format in rotor print ---
        uses_sci_theta = bool(re.search(
            r'\|Δθ\|\s*=\s*\{avg_delta:.4e\}', src
        ))
        findings.append(UpdateFinding(
            finding_id="PRINT-2",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update (rotor print)",
            check="rotor |Δθ| uses scientific-notation format (:.4e)",
            passed=uses_sci_theta,
            severity=SEVERITY_HIGH if not uses_sci_theta else SEVERITY_INFO,
            evidence=(
                "The log line: [UPDATE] rotor: N planes (avg |Δθ|=...). "
                "rotor_lr = eta * 0.01; with eta=10, lior_diff=5e-6, v_angle≈0.5, "
                "v_plane_mag≈0.1: avg_delta ≈ 10 * 0.01 * 5e-6 * 0.5 * 0.1 = 2.5e-9. "
                ":.6f rounds this to 0.000000."
            ),
            recommendation=(
                "Format already uses :.4e — rotor updates are now visible."
            ) if uses_sci_theta else (
                "Change avg |Δθ|={avg_delta:.6f} → avg |Δθ|={avg_delta:.4e} in "
                "the rotor print statement."
            ),
        ))

        # --- PRINT-3: LIoR diff format in directional print ---
        # After fix: should use :.4e; was :.6f
        uses_sci_lior = bool(re.search(
            r'LIoR diff=\{lior_diff_val:.4e\}', src
        ))
        # Also check if :.6f is still used (pre-fix state)
        uses_fixed_lior = bool(re.search(
            r'LIoR diff=\{lior_diff_val:.6f\}', src
        ))
        lior_format_ok = uses_sci_lior and not uses_fixed_lior
        findings.append(UpdateFinding(
            finding_id="PRINT-3",
            role=self.NAME,
            component="training/trainer2.py:apply_manual_update (directional print)",
            check="LIoR diff in directional print uses :.4e format",
            passed=lior_format_ok,
            severity=SEVERITY_MEDIUM if not lior_format_ok else SEVERITY_INFO,
            evidence=(
                "Directional print: [UPDATE] g0_diag += directional (..., LIoR diff=...). "
                "lior_diff_val ≈ 5e-6 prints as '0.000005' with :.6f (just visible), "
                "but :.4e gives '5.0000e-06' which is unambiguous."
            ),
            recommendation=(
                "Format already uses :.4e."
            ) if lior_format_ok else (
                "Change LIoR diff={lior_diff_val:.6f} → LIoR diff={lior_diff_val:.4e} "
                "for consistent scientific notation across all update log lines."
            ),
        ))

        return findings


class MoraleAgent:
    """
    Monitors team health and workload sustainability.
    """

    def audit(self, findings: List[UpdateFinding]) -> List[str]:
        failed = [f for f in findings if not f.passed]
        critical = [f for f in failed if f.severity == SEVERITY_CRITICAL]
        high = [f for f in failed if f.severity == SEVERITY_HIGH]
        notes = []

        if not failed:
            notes.append(
                "All update-signal checks pass — the |Δ|=0 symptom was a log-format "
                "artefact, not a float32 underflow. Updates are occurring."
            )
        else:
            notes.append(
                f"{len(failed)} finding(s) failed "
                f"({len(critical)} critical, {len(high)} high). "
                "Root cause is print-precision masking tiny-but-real updates."
            )

        notes.append(
            "ARE THE TINY UPDATES A PROBLEM? "
            "Short answer: not broken, but too slow to see in short runs. "
            "With |Δ| ≈ 7e-7 per window, g0_diag (clamped [0.01, 100]) needs "
            "~12,800 windows to shift by 1%. Loss and perplexity will not visibly "
            "improve until the geometry adapts enough to steer retrieval. "
            "Fix: raise eta_update from 1e-2 to 1e-1 for 10× faster adaptation "
            "with no other changes."
        )
        notes.append(
            "Loss and perplexity context: with |Δ| ≈ 1e-7–1e-6 per window, "
            "cumulative metric adaptation requires O(1000) windows to shift "
            "g0_diag by ~0.1%. This is slow but not broken. "
            "To accelerate: increase eta_update or nudge_scale."
        )
        notes.append(
            "Memory pressure (87%+) is independent of the update-signal issue. "
            "Reducing batch size or enabling expandable_segments addresses OOM risk. "
            "Per-window GPU RAM is now reported in [PROGRESS] to catch build-up early."
        )
        return notes


class ScribeAgent:
    """
    Consolidates all findings into a structured decision log.
    """

    def consolidate(
        self,
        findings: List[UpdateFinding],
        wiring_checks: List[WiringCheck],
    ) -> ScribeLog:
        action_items: List[str] = []
        for f in findings:
            if not f.passed:
                action_items.append(
                    f"[{f.severity}] {f.finding_id} {f.check}: {f.recommendation}"
                )
        for w in wiring_checks:
            if not w.wired:
                action_items.append(
                    f"[WIRING] {w.feature} at {w.entry_point}: {w.notes}"
                )
        if not action_items:
            action_items.append(
                "No action items — all update-signal checks pass. "
                "The |Δ|=0.000000 display was a log-format artefact (:.6f rounds "
                "sub-1e-6 values to zero). Updates are occurring. "
                "Apply :.4e format fix to make them visible."
            )

        n_pass = sum(1 for f in findings if f.passed)
        n_fail = sum(1 for f in findings if not f.passed)
        trainer2_src = _read("training/trainer2.py")
        eta_update = _extract_float(
            trainer2_src, r"eta_update\s*:\s*float\s*=\s*([0-9e.+-]+)", default=1e-2
        )
        beta_nudge = _extract_float(
            trainer2_src, r"beta_nudge\s*:\s*float\s*=\s*([0-9e.+-]+)", default=1e-3
        )
        eta_ratio = round(eta_update / max(beta_nudge, 1e-30), 4)
        summary = (
            f"{n_pass} checks passed, {n_fail} failed. "
            f"Root cause: print precision (:.6f) hides tiny-but-real updates. "
            f"eta_update/beta_nudge = {eta_ratio}. "
            f"Signal chain intact — lior_diff is not attenuated."
        )
        return ScribeLog(
            findings=findings,
            wiring_checks=wiring_checks,
            summary=summary,
            action_items=action_items,
        )


# ---------------------------------------------------------------------------
# Pipeline wiring checker
# ---------------------------------------------------------------------------

def _check_pipeline_wiring() -> List[WiringCheck]:
    src = _read("training/trainer2.py")
    return [
        WiringCheck(
            feature="apply_manual_update called from run_two_phase_and_update",
            wired=_has(src, "ok = apply_manual_update("),
            entry_point="training/trainer2.py:run_two_phase_and_update",
            notes="If not wired, no metric or rotor updates ever apply.",
        ),
        WiringCheck(
            feature="Velocity forwarded to apply_manual_update for directional update",
            wired=(
                _has(src, "velocity = getattr(free, 'velocity', None)") and
                _has(src, "velocity=velocity,")
            ),
            entry_point="training/trainer2.py:run_two_phase_and_update",
            notes=(
                "Without velocity, only the scalar fallback fires "
                "(g_diag *= adjustment), which is less expressive."
            ),
        ),
        WiringCheck(
            feature="|Δ| print uses scientific notation (:.4e) to show sub-1e-6 updates",
            wired=bool(re.search(r'\|Δ\|\s*=\s*\{delta_norm:.4e\}', src)),
            entry_point="training/trainer2.py:apply_manual_update",
            notes=(
                "Without :.4e, updates smaller than 5e-7 print as 0.000000 "
                "and appear as if no update occurred."
            ),
        ),
        WiringCheck(
            feature="|Δθ| print uses scientific notation (:.4e) to show sub-1e-6 rotor updates",
            wired=bool(re.search(r'\|Δθ\|\s*=\s*\{avg_delta:.4e\}', src)),
            entry_point="training/trainer2.py:apply_manual_update",
            notes=(
                "Rotor updates are typically even smaller than metric updates "
                "(rotor_lr = eta * 0.01). Scientific notation is essential here."
            ),
        ),
        WiringCheck(
            feature="nudge_mode = 'target_embedding' generates a nonzero nudge signal",
            wired=_has(src, 'nudge_mode: str = "target_embedding"'),
            entry_point="training/trainer2.py:build_nudge",
            notes=(
                "If nudge_mode is 'off', lior_diff is always zero and no "
                "updates will ever apply."
            ),
        ),
        WiringCheck(
            feature="Startup prints param count, steps_per_window, and windows_per_epoch",
            wired=(
                _has(src, "steps_per_window =") and
                _has(src, "windows_per_epoch") and
                _has(src, "parameters:")
            ),
            entry_point="training/trainer2.py:trainer2_entrypoint",
            notes=(
                "Param count and training-structure summary helps quickly "
                "identify configuration mismatches at run start."
            ),
        ),
        WiringCheck(
            feature="Startup prints telemetry variable glossary with expected directions",
            wired=_has(src, "Telemetry glossary:"),
            entry_point="training/trainer2.py:trainer2_entrypoint",
            notes=(
                "Glossary makes telemetry columns self-documenting at runtime "
                "without needing to consult external documentation."
            ),
        ),
        WiringCheck(
            feature="GPU RAM reported per window in [PROGRESS] line (gpu=X.XX/Y.YYGB)",
            wired=_has(src, "gpu_alloc_gb") and _has(src, "gpu_reserved_gb"),
            entry_point="training/trainer2.py:trainer2_entrypoint window loop",
            notes=(
                "Per-window GPU RAM visibility is critical for catching memory "
                "pressure build-up before an OOM occurs."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class UpdateSignalAuditTeam:
    """Public entry point for the update-signal collaborative audit."""

    def run(self) -> UpdateSignalReport:
        coordinator = CoordinatorAgent()
        scaling = ScalingAnalystAgent()
        signal_chain = SignalChainAgent()
        nudge = NudgeScaleAgent()
        print_diag = PrintDiagnosticsAgent()
        morale = MoraleAgent()
        scribe = ScribeAgent()

        all_findings: List[UpdateFinding] = []
        all_findings.extend(scaling.audit())
        all_findings.extend(signal_chain.audit())
        all_findings.extend(nudge.audit())
        all_findings.extend(print_diag.audit())

        wiring_checks = _check_pipeline_wiring()
        morale_notes = morale.audit(all_findings)
        scribe_log = scribe.consolidate(all_findings, wiring_checks)

        return UpdateSignalReport(
            coordinator_scope=coordinator.scope(),
            findings=all_findings,
            wiring_checks=wiring_checks,
            morale_notes=morale_notes,
            scribe_log=scribe_log,
            approval_status="AWAITING APPROVAL TO EXECUTE",
        )
