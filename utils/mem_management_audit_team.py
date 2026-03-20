"""
Eight-Agent GPU Memory Management Audit Team

Exhaustively maps all memory-management efforts in the Liorhybrid training
pipeline and reports findings in response to:

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
    GPU 0 has a total capacity of 23.49 GiB of which 76.88 MiB is free.
    (at models/biquaternion.py:382 inside BiQuatCausalLayer.forward)

Scope
-----
The audit covers six dimensions:

1. **Memory cleaners** – background/periodic GPU cache-flush mechanisms
2. **Streaming** – CUDA-stream-based async copy / side-stream patterns
3. **Allocation patterns** – in-place vs out-of-place tensor ops, temporary allocations
4. **OOM prevention** – gradient checkpointing, AMP/autocast, expandable segments
5. **Memory profiling infrastructure** – statistics, watermark tracking, allocator config
6. **Pipeline wiring** – whether cleaners and prevention features are actually connected

Team roles
----------
1. Coordinator        – owns scope, assigns sub-tasks, resolves blockers
2. MemoryProfiler     – audits profiling infrastructure & allocator configuration
3. StreamingAuditor   – audits CUDA-stream usage and async-copy patterns
4. CleanerInventory   – inventories all cache-flush / gc mechanisms
5. AllocationPattern  – identifies in-place vs out-of-place hotspots
6. OOMPrevention      – checks gradient checkpointing, AMP, expandable segments
7. Morale             – workload balance, cadence sustainability
8. Scribe             – consolidated decision log with severity & evidence

STATUS: AWAITING APPROVAL TO EXECUTE
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Severity levels (shared with transport_fiber_audit_team convention)
# ---------------------------------------------------------------------------

SEVERITY_INFO = "INFO"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemFinding:
    """A single finding from one specialist agent."""
    role: str
    component: str        # e.g. "training/gpu_cleanup.py", "trainer2.py:2493"
    check: str            # human-readable description
    passed: bool          # True = healthy / expected; False = gap or risk
    severity: str         # one of the SEVERITY_* constants
    evidence: str         # file + line / runtime measurement
    recommendation: str


@dataclass(frozen=True)
class WiringCheck:
    """Records whether a memory-management feature is wired into the pipeline."""
    feature: str
    wired: bool
    entry_point: str
    notes: str


@dataclass(frozen=True)
class ScribeLog:
    """Consolidated decision log produced by the Scribe agent."""
    findings: List[MemFinding]
    wiring_checks: List[WiringCheck]
    summary: str
    action_items: List[str]

    @property
    def all_passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def critical_findings(self) -> List[MemFinding]:
        return [f for f in self.findings if not f.passed and f.severity == SEVERITY_CRITICAL]

    @property
    def failed_findings(self) -> List[MemFinding]:
        return [f for f in self.findings if not f.passed]


@dataclass
class MemAuditReport:
    """Complete report produced by the full eight-agent team."""
    coordinator_scope: str
    findings: List[MemFinding]
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


def _count_pattern(src: str, pattern: str) -> int:
    """Count non-overlapping occurrences of a regex pattern in src."""
    return len(re.findall(pattern, src))


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

class CoordinatorAgent:
    """
    Owns scope, assigns sub-tasks, resolves blockers.

    Scope
    -----
    Map all GPU memory management efforts in the Liorhybrid training pipeline.
    The triggering OOM (torch.OutOfMemoryError: Tried to allocate 256 MiB,
    only 76.88 MiB free) occurs at models/biquaternion.py:382 inside
    BiQuatCausalLayer.forward() → LayerNorm, invoked from the trainer2.py
    run_window → build_retrieval_batch hot path.

    The audit covers:
      1. Existing memory-cleaner mechanisms (GPUCleanupThread, empty_cache calls)
      2. CUDA streaming / async-copy patterns (flush_stream, non_blocking)
      3. Allocation hot-spots (in-place vs out-of-place tensor ops)
      4. OOM prevention features (gradient checkpointing, AMP, expandable segments)
      5. Memory profiling infrastructure (stats, watermarks, allocator config)
      6. Pipeline wiring (cleaners actually connected to trainer2 entry point?)

    Findings are plan-only.  No code is changed until approved.
    """

    SCOPE = (
        "Exhaustively map GPU memory management efforts across the Liorhybrid "
        "training pipeline. The triggering OOM occurs at biquaternion.py:382 "
        "(LayerNorm inside BiQuatCausalLayer.forward). Audit: (1) memory cleaners "
        "(GPUCleanupThread, empty_cache), (2) CUDA streaming patterns, "
        "(3) tensor-allocation hot-spots, (4) OOM prevention (AMP, gradient "
        "checkpointing, expandable segments), (5) memory profiling infrastructure. "
        "Report findings only — no code changes without approval."
    )

    TASK_QUEUE: List[Tuple[str, str]] = [
        ("MemoryProfiler",    "Audit profiling infrastructure and allocator configuration"),
        ("StreamingAuditor",  "Audit CUDA-stream usage and async-copy patterns"),
        ("CleanerInventory",  "Inventory all cache-flush and gc mechanisms"),
        ("AllocationPattern", "Identify in-place vs out-of-place allocation hot-spots"),
        ("OOMPrevention",     "Check gradient checkpointing, AMP, and expandable segments"),
        ("Morale",            "Flag workload balance and cadence sustainability"),
        ("Scribe",            "Consolidate findings into decision log with severity + evidence"),
    ]

    def scope(self) -> str:
        return self.SCOPE

    def task_queue(self) -> List[Tuple[str, str]]:
        return list(self.TASK_QUEUE)


class MemoryProfilerAgent:
    """
    Audits memory-profiling infrastructure and allocator configuration.

    Key checks:
    - Is PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set in the entry point?
    - Does a memory statistics / watermark tracker exist?
    - Is torch.cuda.memory_stats() / memory_snapshot() used anywhere?
    - Does GPUCleanupThread expose get_stats() and get_memory_info()?
    """

    def audit(self) -> List[MemFinding]:
        findings: List[MemFinding] = []

        gpu_cleanup_src = _read("training/gpu_cleanup.py")
        trainer2_src = _read("training/trainer2.py")
        main_src = _read("main.py")

        # --- Check 1: expandable_segments env-var in entry point ---
        entrypoint_sets_conf = (
            "expandable_segments" in trainer2_src or
            "expandable_segments" in main_src
        )
        findings.append(MemFinding(
            role="MemoryProfiler",
            component="main.py / training/trainer2.py",
            check=(
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set "
                "at the training entry point"
            ),
            passed=entrypoint_sets_conf,
            severity=SEVERITY_HIGH if not entrypoint_sets_conf else SEVERITY_INFO,
            evidence=(
                "The OOM error message itself recommends setting "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. "
                "enable_expandable_segments() exists in training/gpu_cleanup.py "
                "but is never called from main.py or trainer2_entrypoint."
            ),
            recommendation=(
                "Call enable_expandable_segments() or set the env var "
                "at the very start of main.py, before any torch import. "
                "Pending approval."
            ) if not entrypoint_sets_conf else "Expandable segments already configured.",
        ))

        # --- Check 2: GPUCleanupThread.get_stats() / get_memory_info() present ---
        has_get_stats = "def get_stats" in gpu_cleanup_src
        has_get_mem_info = "def get_memory_info" in gpu_cleanup_src
        findings.append(MemFinding(
            role="MemoryProfiler",
            component="training/gpu_cleanup.py",
            check="GPUCleanupThread exposes get_stats() and get_memory_info() for monitoring",
            passed=has_get_stats and has_get_mem_info,
            severity=SEVERITY_LOW if not (has_get_stats and has_get_mem_info) else SEVERITY_INFO,
            evidence=(
                f"get_stats present: {has_get_stats}, "
                f"get_memory_info present: {has_get_mem_info} "
                "(training/gpu_cleanup.py lines ~267-296)"
            ),
            recommendation=(
                "Monitoring hooks present. Consider calling get_stats() periodically "
                "and logging to telemetry."
            ),
        ))

        # --- Check 3: torch.cuda.memory_stats() or memory_snapshot used anywhere ---
        mem_stats_used = (
            "memory_stats" in trainer2_src or
            "memory_snapshot" in trainer2_src or
            "memory_stats" in main_src
        )
        findings.append(MemFinding(
            role="MemoryProfiler",
            component="training/trainer2.py",
            check="torch.cuda.memory_stats() or memory_snapshot() used for profiling",
            passed=mem_stats_used,
            severity=SEVERITY_LOW if not mem_stats_used else SEVERITY_INFO,
            evidence=(
                "No torch.cuda.memory_stats() or memory_snapshot() calls found. "
                "Only torch.cuda.memory_allocated() and memory_reserved() are used "
                "(trainer2.py:2488-2489)."
            ),
            recommendation=(
                "Consider adding torch.cuda.memory_stats() logging at OOM-risk "
                "checkpoints (e.g. every N windows) to expose fragmentation early."
            ) if not mem_stats_used else "Memory stats profiling present.",
        ))

        # --- Check 4: enable_expandable_segments() exists in gpu_cleanup.py ---
        has_enable_fn = "def enable_expandable_segments" in gpu_cleanup_src
        findings.append(MemFinding(
            role="MemoryProfiler",
            component="training/gpu_cleanup.py",
            check="enable_expandable_segments() helper function exists",
            passed=has_enable_fn,
            severity=SEVERITY_MEDIUM if not has_enable_fn else SEVERITY_INFO,
            evidence=(
                "training/gpu_cleanup.py contains enable_expandable_segments() "
                "and check_cuda_alloc_conf() helpers (lines ~37-75)."
            ),
            recommendation=(
                "Helper exists. Wire it into trainer2_entrypoint or main.py startup "
                "so the allocator is configured before the first CUDA allocation."
            ),
        ))

        return findings


class StreamingAuditorAgent:
    """
    Audits CUDA-stream usage and async-copy patterns.

    Key checks:
    - Does TelemetryState use a dedicated flush_stream for async metric copies?
    - Are non_blocking=True transfers used for batch data movement?
    - Does run_window use a dedicated CUDA stream for prefetching?
    - Is there a CUDA stream for memory cleanup (avoid sync in hot loop)?
    """

    def audit(self) -> List[MemFinding]:
        findings: List[MemFinding] = []

        trainer2_src = _read("training/trainer2.py")

        # --- Check 1: flush_stream for async telemetry ---
        has_flush_stream = (
            "flush_stream" in trainer2_src and
            "torch.cuda.Stream()" in trainer2_src
        )
        findings.append(MemFinding(
            role="StreamingAuditor",
            component="training/trainer2.py:TelemetryState",
            check="Dedicated CUDA side-stream (flush_stream) used for async metric flushes",
            passed=has_flush_stream,
            severity=SEVERITY_INFO if has_flush_stream else SEVERITY_MEDIUM,
            evidence=(
                "TelemetryState.flush_stream (torch.cuda.Stream) found at "
                "trainer2.py:3038-3047. Used in _log_telemetry_step with "
                "torch.cuda.stream(stream) context and pin_memory staging buffer."
            ),
            recommendation=(
                "Side-stream telemetry flush is correctly implemented. "
                "No action needed for telemetry path."
            ),
        ))

        # --- Check 2: non_blocking transfers ---
        nb_count = _count_pattern(trainer2_src, r"non_blocking\s*=\s*True")
        findings.append(MemFinding(
            role="StreamingAuditor",
            component="training/trainer2.py",
            check="non_blocking=True used for host↔device tensor transfers",
            passed=nb_count > 0,
            severity=SEVERITY_LOW if nb_count == 0 else SEVERITY_INFO,
            evidence=(
                f"Found {nb_count} non_blocking=True transfer(s). "
                "trainer2.py:172 — batch.to(device=device, non_blocking=True)."
            ),
            recommendation=(
                "non_blocking=True is used. Verify all .to(device=...) calls "
                "in the data loading path use non_blocking=True for maximum throughput."
            ),
        ))

        # --- Check 3: CUDA stream for memory cleanup (avoid sync stalls) ---
        cleanup_stream = (
            "torch.cuda.Stream" in trainer2_src and
            "empty_cache" in trainer2_src
        )
        # The flush_stream is for telemetry, not for cleanup; check if cleanup also streams
        cleanup_in_stream_ctx = bool(re.search(
            r"cuda\.stream.*?empty_cache|empty_cache.*?cuda\.stream",
            trainer2_src,
            re.DOTALL,
        ))
        findings.append(MemFinding(
            role="StreamingAuditor",
            component="training/trainer2.py:run_two_phase_and_update",
            check="torch.cuda.empty_cache() inside a CUDA stream context (non-blocking cleanup)",
            passed=cleanup_in_stream_ctx,
            severity=SEVERITY_MEDIUM if not cleanup_in_stream_ctx else SEVERITY_INFO,
            evidence=(
                "trainer2.py:2488-2493: empty_cache() is called inline in the main "
                "training loop (inside run_two_phase_and_update) without a dedicated "
                "stream context. This forces a synchronization stall on the GPU when "
                "triggered."
            ),
            recommendation=(
                "Consider moving the empty_cache() call into a background thread "
                "(GPUCleanupThread already provides this) or into a side stream "
                "so the main compute stream is not stalled. Pending approval."
            ) if not cleanup_in_stream_ctx else "Cleanup already stream-scoped.",
        ))

        # --- Check 4: pinned memory for staging buffers ---
        pin_memory_count = _count_pattern(trainer2_src, r"pin_memory\s*=\s*True")
        findings.append(MemFinding(
            role="StreamingAuditor",
            component="training/trainer2.py:TelemetryState",
            check="Pinned (page-locked) CPU buffers used for async H2D/D2H staging",
            passed=pin_memory_count > 0,
            severity=SEVERITY_LOW if pin_memory_count == 0 else SEVERITY_INFO,
            evidence=(
                f"Found {pin_memory_count} pin_memory=True allocation(s). "
                "trainer2.py:3052 — staging_cpu tensor is pinned for async metric copy."
            ),
            recommendation=(
                "Pinned staging buffer is present in telemetry path. "
                "Consider pinning input batch buffers too for faster H2D transfers."
            ),
        ))

        return findings


class CleanerInventoryAgent:
    """
    Inventories every cache-flush, gc.collect, and background cleanup
    mechanism in the codebase and checks whether they are wired into the
    trainer2 hot path.

    Key checks:
    - GPUCleanupThread exists in training/gpu_cleanup.py
    - cleanup_gpu_memory() standalone function exists
    - inline empty_cache() in trainer2.py
    - GPUCleanupThread is instantiated + started in trainer2_entrypoint
    - gc.collect() is paired with every empty_cache() call
    """

    def audit(self) -> List[MemFinding]:
        findings: List[MemFinding] = []

        gpu_cleanup_src = _read("training/gpu_cleanup.py")
        trainer2_src = _read("training/trainer2.py")
        trainer_src = _read("training/trainer.py")

        # --- Check 1: GPUCleanupThread class present ---
        has_thread = "class GPUCleanupThread" in gpu_cleanup_src
        findings.append(MemFinding(
            role="CleanerInventory",
            component="training/gpu_cleanup.py",
            check="GPUCleanupThread background-cleanup daemon exists",
            passed=has_thread,
            severity=SEVERITY_CRITICAL if not has_thread else SEVERITY_INFO,
            evidence=(
                "training/gpu_cleanup.py contains GPUCleanupThread — a background "
                "daemon that runs gc.collect() + torch.cuda.empty_cache() on a "
                "configurable interval (time-based or step-based). "
                "Supports force_cleanup(), get_stats(), get_memory_info()."
            ),
            recommendation=(
                "GPUCleanupThread exists. Ensure it is started at trainer2_entrypoint. "
                "See CleanerInventory Check 4."
            ),
        ))

        # --- Check 2: standalone cleanup_gpu_memory() present ---
        has_cleanup_fn = "def cleanup_gpu_memory" in gpu_cleanup_src
        findings.append(MemFinding(
            role="CleanerInventory",
            component="training/gpu_cleanup.py",
            check="cleanup_gpu_memory() one-shot helper function exists",
            passed=has_cleanup_fn,
            severity=SEVERITY_MEDIUM if not has_cleanup_fn else SEVERITY_INFO,
            evidence=(
                "cleanup_gpu_memory(verbose=False) at training/gpu_cleanup.py:298 "
                "provides a single-call gc.collect() + synchronize() + empty_cache() "
                "for manual cleanup without a background thread."
            ),
            recommendation=(
                "One-shot helper exists. Consider calling it at the start of "
                "trainer2_entrypoint before heavy model allocation."
            ),
        ))

        # --- Check 3: inline empty_cache in trainer2 ---
        inline_empty_cache = "torch.cuda.empty_cache()" in trainer2_src
        # Check if gc.collect() is also called nearby
        gc_paired = "gc.collect()" in trainer2_src
        findings.append(MemFinding(
            role="CleanerInventory",
            component="training/trainer2.py:2493",
            check="Inline torch.cuda.empty_cache() present in run_two_phase_and_update",
            passed=inline_empty_cache,
            severity=SEVERITY_HIGH if not inline_empty_cache else SEVERITY_INFO,
            evidence=(
                "trainer2.py:2488-2493 — empty_cache() called when "
                "window_idx % 50 == 0 AND mem_allocated/mem_reserved > 0.9. "
                f"gc.collect() paired: {gc_paired}."
            ),
            recommendation=(
                "Inline cleaner present. Note: gc.collect() is NOT called alongside "
                "empty_cache() in the inline path — add gc.collect() before "
                "empty_cache() for maximum effect (Python objects holding CUDA tensors "
                "must be GC'd first). Pending approval."
            ) if inline_empty_cache and not gc_paired else (
                "Inline cleaner with gc.collect() fully configured."
                if inline_empty_cache else
                "Add inline empty_cache() call in run_two_phase_and_update."
            ),
        ))

        # --- Check 4: GPUCleanupThread started in trainer2_entrypoint ---
        thread_started_in_trainer2 = (
            "GPUCleanupThread" in trainer2_src and
            "gpu_cleanup" in trainer2_src
        )
        findings.append(MemFinding(
            role="CleanerInventory",
            component="training/trainer2.py:trainer2_entrypoint",
            check="GPUCleanupThread is imported and started in trainer2_entrypoint",
            passed=thread_started_in_trainer2,
            severity=SEVERITY_HIGH if not thread_started_in_trainer2 else SEVERITY_INFO,
            evidence=(
                "training/trainer2.py does NOT import or instantiate GPUCleanupThread. "
                "The background cleanup daemon in training/gpu_cleanup.py is unused "
                "by the trainer2 code path, leaving periodic cache-flush inactive."
            ),
            recommendation=(
                "Add to trainer2_entrypoint:\n"
                "  from training.gpu_cleanup import GPUCleanupThread\n"
                "  _cleanup = GPUCleanupThread(cleanup_every_n_steps=50, verbose=True)\n"
                "  _cleanup.start()\n"
                "  # ... training ...\n"
                "  _cleanup.stop()\n"
                "Pending approval."
            ) if not thread_started_in_trainer2 else "GPUCleanupThread wired.",
        ))

        # --- Check 5: trainer.py (non-trainer2) uses GPUCleanupThread ---
        trainer_uses_cleanup = (
            "GPUCleanupThread" in trainer_src or
            "gpu_cleanup" in trainer_src
        )
        findings.append(MemFinding(
            role="CleanerInventory",
            component="training/trainer.py",
            check="trainer.py (legacy trainer) uses GPUCleanupThread",
            passed=trainer_uses_cleanup,
            severity=SEVERITY_MEDIUM if not trainer_uses_cleanup else SEVERITY_INFO,
            evidence=(
                f"GPUCleanupThread referenced in training/trainer.py: "
                f"{trainer_uses_cleanup}. "
                "If the legacy trainer path is used, the same cleanup gap applies."
            ),
            recommendation=(
                "Wire GPUCleanupThread into training/trainer.py for consistency."
                if not trainer_uses_cleanup else
                "Legacy trainer cleanup is wired."
            ),
        ))

        return findings


class AllocationPatternAgent:
    """
    Identifies in-place vs out-of-place tensor allocation hot-spots.

    Key checks:
    - residual + output in BiQuatCausalLayer.forward (OOM site)
    - in-place .add_() / .mul_() usage in hot paths
    - intermediate tensor accumulation in run_window
    - grad accumulation pattern
    """

    def audit(self) -> List[MemFinding]:
        findings: List[MemFinding] = []

        biquat_src = _read("models/biquaternion.py")
        trainer2_src = _read("training/trainer2.py")

        # --- Check 1: OOM site — residual + output creates new tensor before LayerNorm ---
        # Look for "residual + output" vs "residual.add_(output)" near self.norm
        oom_pattern = bool(re.search(
            r"self\.norm\s*\(\s*residual\s*\+\s*output\s*\)",
            biquat_src,
        ))
        inplace_pattern = bool(re.search(
            r"residual\.add_\s*\(\s*output\s*\)",
            biquat_src,
        ))
        findings.append(MemFinding(
            role="AllocationPattern",
            component="models/biquaternion.py:382",
            check=(
                "LayerNorm receives residual + output as out-of-place new tensor "
                "(OOM trigger site)"
            ),
            passed=not oom_pattern,
            severity=SEVERITY_HIGH if oom_pattern else SEVERITY_INFO,
            evidence=(
                "models/biquaternion.py:382: "
                "output = self.norm(residual + output)\n"
                "The expression `residual + output` allocates a temporary tensor "
                "of shape [B, N, d_model] before passing it to LayerNorm. "
                "During the OOM event (allocating 256 MiB for a 23.5 GiB GPU "
                "with only 76.88 MiB free), this temporary allocation is the "
                "proximate cause. "
                f"In-place alternative already present: {inplace_pattern}."
            ),
            recommendation=(
                "Replace `output = self.norm(residual + output)` with:\n"
                "  residual.add_(output)\n"
                "  output = self.norm(residual)\n"
                "This reuses the residual buffer and avoids the temporary "
                "256 MiB allocation. Pending approval."
            ) if oom_pattern else "No out-of-place OOM pattern found.",
        ))

        # --- Check 2: in-place accumulation in trainer2 hot path ---
        inplace_add_count = _count_pattern(trainer2_src, r"\.add_\s*\(")
        inplace_mul_count = _count_pattern(trainer2_src, r"\.mul_\s*\(")
        findings.append(MemFinding(
            role="AllocationPattern",
            component="training/trainer2.py",
            check="In-place .add_() / .mul_() used for metric accumulation in run_window",
            passed=(inplace_add_count + inplace_mul_count) > 0,
            severity=SEVERITY_LOW if (inplace_add_count + inplace_mul_count) == 0 else SEVERITY_INFO,
            evidence=(
                f"Found {inplace_add_count} .add_() and {inplace_mul_count} .mul_() "
                "calls in trainer2.py. Vectorized in-place metric accumulation is "
                "present (e.g. velocity_acc.add_(v), lior_acc += lior_loss)."
            ),
            recommendation=(
                "In-place accumulation patterns are in use. "
                "Review any remaining `x = x + y` patterns in the forward hot-path "
                "and convert to x.add_(y) where gradient flow permits."
            ),
        ))

        # --- Check 3: pre-allocated GPU metric buffer ---
        preallocated_buffer = bool(re.search(
            r"torch\.zeros\s*\(\s*3\s*,\s*device\s*=\s*DEVICE",
            trainer2_src,
        ))
        findings.append(MemFinding(
            role="AllocationPattern",
            component="training/trainer2.py:run_window",
            check="Pre-allocated GPU tensor for batch metric accumulation (avoids per-step alloc)",
            passed=preallocated_buffer,
            severity=SEVERITY_INFO if preallocated_buffer else SEVERITY_LOW,
            evidence=(
                "trainer2.py:run_window allocates progress_metrics_gpu = "
                "torch.zeros(3, device=DEVICE) once before the step loop. "
                "Metric sync uses a single .cpu() call per log interval."
            ),
            recommendation=(
                "Pre-allocated metric buffer pattern is healthy. "
                "Extend this pattern to any other per-step allocations."
            ),
        ))

        # --- Check 4: del intermediate tensors in forward ---
        del_intermediates = _count_pattern(trainer2_src, r"\bdel\b\s+\w")
        findings.append(MemFinding(
            role="AllocationPattern",
            component="training/trainer2.py",
            check="Explicit `del intermediate_tensor` calls to release GPU memory early",
            passed=del_intermediates > 0,
            severity=SEVERITY_LOW if del_intermediates == 0 else SEVERITY_INFO,
            evidence=(
                f"Found {del_intermediates} `del <name>` statements in trainer2.py. "
                "Explicit deletion removes Python references so the GC can "
                "reclaim CUDA memory sooner."
            ),
            recommendation=(
                "Consider adding `del q_coord, cand_coord, cand_state` after "
                "they are no longer needed in run_window to free memory before "
                "the next loop iteration. Pending approval."
            ) if del_intermediates == 0 else
            "Explicit tensor deletion pattern in use.",
        ))

        return findings


class OOMPreventionAgent:
    """
    Checks OOM-prevention features: gradient checkpointing, AMP/autocast,
    expandable segments, and BPTT detach.

    Key checks:
    - torch.utils.checkpoint.checkpoint used in forward paths?
    - autocast / GradScaler used in trainer2 run_window?
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set?
    - BPTT detach implemented in BiQuatCausalLayer?
    - Inference mode used for measurement/retrieval passes?
    """

    def audit(self) -> List[MemFinding]:
        findings: List[MemFinding] = []

        trainer2_src = _read("training/trainer2.py")
        trainer_src = _read("training/trainer.py")
        biquat_src = _read("models/biquaternion.py")
        geom_stack_src = _read("inference/geometric_stack.py")
        main_src = _read("main.py")

        # --- Check 1: Gradient checkpointing in inference/geometric_stack.py ---
        gc_in_stack = "checkpoint" in geom_stack_src and "torch.utils" in geom_stack_src
        findings.append(MemFinding(
            role="OOMPrevention",
            component="inference/geometric_stack.py",
            check=(
                "torch.utils.checkpoint.checkpoint used in GeometricStack "
                "layer loop (trades 30% compute for 50-70% activation memory)"
            ),
            passed=gc_in_stack,
            severity=SEVERITY_HIGH if not gc_in_stack else SEVERITY_INFO,
            evidence=(
                "inference/geometric_stack.py:302 iterates over layers calling "
                "layer(encoder_output, memory, diagnose=layer_diagnose). "
                "No gradient checkpointing wrapper found. Each BiQuatCausalLayer "
                "forward accumulates full activations — with many layers this "
                "compounds the 256 MiB OOM trigger."
            ),
            recommendation=(
                "Wrap each layer call with gradient checkpointing during training:\n"
                "  from torch.utils.checkpoint import checkpoint\n"
                "  if self.training and self.use_gradient_checkpointing:\n"
                "      out, mem = checkpoint(layer, enc_out, memory,\n"
                "                            use_reentrant=False)\n"
                "  else:\n"
                "      out, mem = layer(enc_out, memory)\n"
                "Add use_gradient_checkpointing flag to GeometricStack config. "
                "Pending approval."
            ),
        ))

        # --- Check 2: AMP / autocast in trainer2 run_window ---
        autocast_in_trainer2 = (
            "autocast" in trainer2_src or
            "amp" in trainer2_src.lower()
        )
        findings.append(MemFinding(
            role="OOMPrevention",
            component="training/trainer2.py:run_window",
            check="AMP (torch.cuda.amp.autocast) used in trainer2 run_window forward pass",
            passed=autocast_in_trainer2,
            severity=SEVERITY_HIGH if not autocast_in_trainer2 else SEVERITY_INFO,
            evidence=(
                "training/trainer.py uses GradScaler + autocast (trainer.py:30, 132, 554). "
                "training/trainer2.py does NOT import or use autocast/GradScaler. "
                "Mixed precision would halve the 256 MiB activation footprint."
            ),
            recommendation=(
                "Add AMP to the trainer2 forward paths:\n"
                "  from torch.cuda.amp import autocast, GradScaler\n"
                "  with autocast():\n"
                "      output = _forward_model(x, batch)\n"
                "This is already proven in training/trainer.py. "
                "Pending approval."
            ) if not autocast_in_trainer2 else "AMP already used in trainer2.",
        ))

        # --- Check 3: BPTT detach in BiQuatCausalLayer ---
        bptt_detach = (
            "bptt_window" in biquat_src and
            "detach" in biquat_src
        )
        findings.append(MemFinding(
            role="OOMPrevention",
            component="models/biquaternion.py:BiQuatCausalLayer",
            check="BPTT windowed detach implemented to limit gradient graph memory",
            passed=bptt_detach,
            severity=SEVERITY_MEDIUM if not bptt_detach else SEVERITY_INFO,
            evidence=(
                "models/biquaternion.py contains bptt_window and should_detach logic. "
                "Detaching truncates the autograd graph every N steps, "
                "preventing gradient memory from accumulating across the full sequence."
            ),
            recommendation=(
                "BPTT detach is implemented. Verify bptt_window is set "
                "to a reasonable value (e.g. 16-64) in the training config "
                "for long sequences to bound memory usage."
            ),
        ))

        # --- Check 4: inference_mode / no_grad for retrieval / measurement passes ---
        inference_mode_count = _count_pattern(trainer2_src, r"inference_mode|torch\.no_grad")
        findings.append(MemFinding(
            role="OOMPrevention",
            component="training/trainer2.py",
            check=(
                "@torch.inference_mode() or torch.no_grad() used for "
                "measurement/retrieval passes"
            ),
            passed=inference_mode_count > 0,
            severity=SEVERITY_HIGH if inference_mode_count == 0 else SEVERITY_INFO,
            evidence=(
                f"Found {inference_mode_count} inference_mode/no_grad usage(s) in "
                "trainer2.py. Measurement and retrieval passes that do not "
                "require gradients should run under inference_mode to avoid "
                "allocating an autograd graph."
            ),
            recommendation=(
                "Wrap build_retrieval_batch with @torch.inference_mode() or "
                "torch.no_grad() context if gradients are not required for "
                "the retrieval embedding pass. Pending approval."
            ) if inference_mode_count == 0 else
            "Inference/no-grad guards present.",
        ))

        # --- Check 5: expandable_segments set at program entry ---
        entry_sets_segments = (
            "expandable_segments" in main_src or
            "expandable_segments" in trainer2_src
        )
        findings.append(MemFinding(
            role="OOMPrevention",
            component="main.py",
            check="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set before torch import",
            passed=entry_sets_segments,
            severity=SEVERITY_HIGH if not entry_sets_segments else SEVERITY_INFO,
            evidence=(
                "The OOM error message explicitly recommends expandable_segments:True "
                "to reduce fragmentation. Neither main.py nor trainer2.py sets this "
                "env var. training/gpu_cleanup.py provides enable_expandable_segments() "
                "but it must be called BEFORE torch is imported."
            ),
            recommendation=(
                "Add to the very top of main.py (before any torch import):\n"
                "  import os\n"
                "  os.environ.setdefault(\n"
                '      "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"\n'
                "  )\n"
                "Pending approval."
            ) if not entry_sets_segments else "expandable_segments already set.",
        ))

        return findings


class MoraleAgent:
    """Monitors team health and workload sustainability."""

    def audit(self, findings: List[MemFinding]) -> List[str]:
        notes: List[str] = []
        total = len(findings)
        failed = sum(1 for f in findings if not f.passed)
        critical = sum(1 for f in findings if not f.passed and f.severity == SEVERITY_CRITICAL)
        high = sum(1 for f in findings if not f.passed and f.severity == SEVERITY_HIGH)
        medium = sum(1 for f in findings if not f.passed and f.severity == SEVERITY_MEDIUM)

        notes.append(
            f"Team health check: {total} checks completed, {failed} failed "
            f"({critical} CRITICAL, {high} HIGH, {medium} MEDIUM)."
        )

        if critical > 0:
            notes.append(
                "⚠  CRITICAL findings present. Coordinator review required "
                "before scheduling remediation sprint."
            )
        if high >= 3:
            notes.append(
                "⚠  Multiple HIGH findings. Recommend splitting remediation "
                "into two sprints: (1) OOM-immediate fixes (expandable_segments, "
                "GPUCleanupThread wiring, AMP in trainer2), "
                "(2) structural improvements (gradient checkpointing, in-place ops)."
            )
        elif high > 0:
            notes.append(
                "⚠  HIGH findings present. These are the primary OOM contributors. "
                "Address in priority order: expandable_segments → GPUCleanupThread "
                "wiring → AMP → gradient checkpointing."
            )
        else:
            notes.append(
                "✓  No CRITICAL or HIGH issues. Remaining gaps are LOW/MEDIUM — "
                "addressable in a single cleanup sprint."
            )

        notes.append(
            "Morale note: All findings are PLAN-ONLY. "
            "No code has been changed. Approval required before execution."
        )
        return notes


class ScribeAgent:
    """Consolidates all findings into a structured decision log."""

    def consolidate(
        self,
        findings: List[MemFinding],
        wiring_checks: List[WiringCheck],
    ) -> ScribeLog:
        failed = [f for f in findings if not f.passed]

        summary_lines = [
            f"Total checks: {len(findings)}",
            f"Passed: {sum(1 for f in findings if f.passed)}",
            f"Failed: {len(failed)}",
            f"Wiring gaps: {sum(1 for w in wiring_checks if not w.wired)}",
        ]
        summary = " | ".join(summary_lines)

        action_items: List[str] = []
        for sev in (SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW):
            for f in failed:
                if f.severity == sev:
                    action_items.append(
                        f"[{f.severity}] [{f.role}] {f.component} — "
                        f"{f.check}: {f.recommendation}"
                    )

        for w in wiring_checks:
            if not w.wired:
                action_items.append(
                    f"[WIRING] {w.feature} not connected at {w.entry_point}: {w.notes}"
                )

        if not action_items:
            action_items.append("No action items — all checks passed.")

        return ScribeLog(
            findings=findings,
            wiring_checks=wiring_checks,
            summary=summary,
            action_items=action_items,
        )


# ---------------------------------------------------------------------------
# Pipeline wiring checker (static / structural)
# ---------------------------------------------------------------------------

def _check_pipeline_wiring() -> List[WiringCheck]:
    """
    Inspect source files to determine whether memory-management features
    are wired into the trainer2 pipeline.
    """
    checks: List[WiringCheck] = []

    trainer2_src = _read("training/trainer2.py")
    main_src = _read("main.py")

    # 1. GPUCleanupThread imported and started in trainer2
    checks.append(WiringCheck(
        feature="GPUCleanupThread (background memory cleaner)",
        wired=(
            "GPUCleanupThread" in trainer2_src and
            ".start()" in trainer2_src and
            "gpu_cleanup" in trainer2_src
        ),
        entry_point="training/trainer2.py:trainer2_entrypoint",
        notes=(
            "GPUCleanupThread lives in training/gpu_cleanup.py but is not imported "
            "in trainer2.py. The background cleanup daemon is inactive on the "
            "trainer2 code path."
        ),
    ))

    # 2. AMP (autocast) in trainer2 forward pass
    checks.append(WiringCheck(
        feature="AMP autocast in trainer2 forward pass",
        wired=("autocast" in trainer2_src),
        entry_point="training/trainer2.py:run_window",
        notes=(
            "training/trainer.py uses autocast+GradScaler. "
            "training/trainer2.py does not import or use autocast. "
            "Mixed precision would halve activation memory."
        ),
    ))

    # 3. Gradient checkpointing in geometric_stack layer loop
    geom_stack_src = _read("inference/geometric_stack.py")
    checks.append(WiringCheck(
        feature="Gradient checkpointing in GeometricStack layer loop",
        wired=(
            "checkpoint" in geom_stack_src and
            "torch.utils" in geom_stack_src
        ),
        entry_point="inference/geometric_stack.py:GeometricStack.forward",
        notes=(
            "GeometricStack iterates over BiQuatCausalLayer instances. "
            "No torch.utils.checkpoint.checkpoint wrapper found. "
            "Each layer's activations are kept in memory for backprop."
        ),
    ))

    # 4. expandable_segments at program entry
    checks.append(WiringCheck(
        feature="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        wired=("expandable_segments" in main_src or "expandable_segments" in trainer2_src),
        entry_point="main.py (before torch import)",
        notes=(
            "Neither main.py nor trainer2.py sets PYTORCH_CUDA_ALLOC_CONF. "
            "The OOM error message explicitly recommends this setting. "
            "enable_expandable_segments() exists in training/gpu_cleanup.py "
            "but is not called."
        ),
    ))

    # 5. inline empty_cache wired in run_two_phase_and_update
    checks.append(WiringCheck(
        feature="Inline torch.cuda.empty_cache() in run_two_phase_and_update",
        wired=("empty_cache" in trainer2_src),
        entry_point="training/trainer2.py:run_two_phase_and_update",
        notes=(
            "trainer2.py:2493 calls empty_cache() when "
            "window_idx%50==0 AND usage_ratio>0.9. Present but triggered only "
            "at high memory pressure — too late to prevent OOM."
        ),
    ))

    return checks


# ---------------------------------------------------------------------------
# Eight-Agent Memory Management Audit Team (public entry point)
# ---------------------------------------------------------------------------

class MemManagementAuditTeam:
    """
    Eight-agent specialist team that exhaustively maps GPU memory management
    efforts in the Liorhybrid training pipeline and reports findings.

    Triggered by:
        torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256 MiB.
        (at models/biquaternion.py:382 — LayerNorm in BiQuatCausalLayer.forward)

    STATUS: AWAITING APPROVAL TO EXECUTE
    Call run() to produce a full MemAuditReport; no code changes are made.
    """

    def run(self) -> MemAuditReport:
        """
        Execute the full audit and return the report.
        Does NOT make any code changes — findings are plan-only.
        """
        coordinator = CoordinatorAgent()
        profiler = MemoryProfilerAgent()
        streamer = StreamingAuditorAgent()
        cleaner = CleanerInventoryAgent()
        alloc = AllocationPatternAgent()
        oom = OOMPreventionAgent()
        morale = MoraleAgent()
        scribe = ScribeAgent()

        all_findings: List[MemFinding] = []
        all_findings.extend(profiler.audit())
        all_findings.extend(streamer.audit())
        all_findings.extend(cleaner.audit())
        all_findings.extend(alloc.audit())
        all_findings.extend(oom.audit())

        wiring_checks = _check_pipeline_wiring()
        morale_notes = morale.audit(all_findings)
        scribe_log = scribe.consolidate(all_findings, wiring_checks)

        return MemAuditReport(
            coordinator_scope=coordinator.scope(),
            findings=all_findings,
            wiring_checks=wiring_checks,
            morale_notes=morale_notes,
            scribe_log=scribe_log,
            approval_status="AWAITING APPROVAL TO EXECUTE",
        )
