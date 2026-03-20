"""
Pipeline Consistency Agent Teams.

Two mirror teams audit data flowing through the inference pipeline:

  VariableConsistencyTeam  — tracks *variable names*.
  DimensionConsistencyTeam — tracks *tensor shapes, dtypes, and units*.

Each team has the same internal structure:

  BookkeeperAgent (× 1)
      CEO of the team.  Records every variable/tensor it first encounters,
      then logs the final outcome of each tracked path.

  ScanningAgent (× 5)
      Each agent latches on to one variable/tensor at the pipeline's entry
      point and follows it step-by-step until the path either completes or
      breaks (name mismatch, shape change, dtype flip, …).  When an agent
      finishes a path it returns to the marker queue for its next assignment.

  MarkerAgent (× 2)
      The two markers leapfrog ahead of the scanning agents, always
      pointing at the next two unclaimed variables/tensors.  This guarantees
      that a scanning agent returning from a completed path always has a
      pre-identified next target to jump to immediately.

      Leapfrog rule:
        • After all 5 scanning agents have claimed the first 5 variables,
          Marker-1 sits on variable 6, Marker-2 sits on variable 7.
        • When a scanning agent finishes and calls ``next_target()``:
          – The agent takes Marker-1's variable.
          – Marker-1 jumps to the next unclaimed variable.
          – Marker-2 stays put (now the new Marker-1 target).
          (Conceptually the two markers alternate roles each handoff.)

Usage
-----
    from utils.pipeline_consistency_agents import (
        VariableConsistencyTeam,
        DimensionConsistencyTeam,
        PipelineStep,
    )

    # Register steps manually …
    team = VariableConsistencyTeam()
    team.register_step(PipelineStep(
        name="address_builder.forward",
        inputs={"embedding": emb, "neighbor_embeddings": nbr},
        outputs={"addr.core": core, "addr.metric": metric},
    ))
    report = team.report()

    # … or wrap a callable with the context manager:
    with team.instrument("neighbor_selector.select_neighbors") as ctx:
        result = selector.select_neighbors(...)
        ctx.record_outputs({"selected_neighbors": result[0], "scores": result[1]})
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except Exception:
    pass

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

class PathOutcome(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"    # variable reached the pipeline's final step
    NAME_BREAK  = "name_break"   # variable disappeared / renamed without alias
    SHAPE_BREAK = "shape_break"  # tensor shape changed unexpectedly
    DTYPE_BREAK = "dtype_break"  # tensor dtype changed unexpectedly
    UNIT_BREAK  = "unit_break"   # dimensional unit inconsistency detected
    DROPPED     = "dropped"      # variable stopped appearing mid-pipeline


def _snapshot(value: Any) -> Dict[str, Any]:
    """Capture a lightweight, serialisable snapshot of a value."""
    snap: Dict[str, Any] = {}
    if hasattr(value, "shape"):
        snap["shape"] = tuple(value.shape)
    if hasattr(value, "dtype"):
        snap["dtype"] = str(value.dtype)
    if hasattr(value, "is_complex"):
        snap["complex"] = bool(value.is_complex())
    if not snap:
        snap["repr"] = repr(value)[:120]
    return snap


@dataclass
class PathRecord:
    """Full lifecycle record for one tracked variable / tensor."""
    name: str
    first_step: str
    first_snapshot: Dict[str, Any]
    steps: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    outcome: PathOutcome = PathOutcome.IN_PROGRESS
    break_detail: str = ""   # e.g. "expected 'scores', got 'raw_scores'"


@dataclass
class PipelineStep:
    """One observable step in the pipeline with named inputs and outputs."""
    name: str
    inputs:  Dict[str, Any]
    outputs: Dict[str, Any]


# ---------------------------------------------------------------------------
# BookkeeperAgent
# ---------------------------------------------------------------------------

class BookkeeperAgent:
    """
    Records every variable first encountered and its final path outcome.

    Thread-safe: multiple scanning agents may call it concurrently.
    """

    def __init__(self, team_label: str = "variable"):
        self.team_label = team_label
        self._lock = threading.Lock()
        self.records: Dict[str, PathRecord] = {}
        self.action_log: List[str] = []

    # -- write ----------------------------------------------------------------

    def note_first_encounter(self, name: str, step: str, value: Any) -> None:
        with self._lock:
            if name not in self.records:
                snap = _snapshot(value)
                self.records[name] = PathRecord(
                    name=name, first_step=step, first_snapshot=snap
                )
                self.action_log.append(
                    f"[{self.team_label}] FIRST  {name!r:40s} at {step!r}"
                )

    def note_step(self, name: str, step: str, value: Any) -> None:
        with self._lock:
            if name in self.records:
                self.records[name].steps.append((step, _snapshot(value)))

    def note_outcome(
        self,
        name: str,
        outcome: PathOutcome,
        break_detail: str = "",
    ) -> None:
        with self._lock:
            if name not in self.records:
                return
            rec = self.records[name]
            if rec.outcome != PathOutcome.IN_PROGRESS:
                return  # never overwrite an already-resolved outcome
            rec.outcome = outcome
            rec.break_detail = break_detail
            flag = "✓" if outcome == PathOutcome.COMPLETED else "✗"
            detail = f" — {break_detail}" if break_detail else ""
            self.action_log.append(
                f"[{self.team_label}] {flag} {name!r:40s} {outcome.value}{detail}"
            )

    # -- read -----------------------------------------------------------------

    def report(self) -> str:
        lines = [f"=== Pipeline Consistency Report ({self.team_label}) ==="]
        for rec in self.records.values():
            lines.append(
                f"  {rec.name!r:42s} {rec.outcome.value}"
                + (f"  [{rec.break_detail}]" if rec.break_detail else "")
            )
            lines.append(
                f"    first at {rec.first_step!r}  snapshot={rec.first_snapshot}"
            )
        return "\n".join(lines)

    @property
    def has_breaks(self) -> bool:
        bad = {PathOutcome.NAME_BREAK, PathOutcome.SHAPE_BREAK,
               PathOutcome.DTYPE_BREAK, PathOutcome.UNIT_BREAK,
               PathOutcome.DROPPED}
        return any(r.outcome in bad for r in self.records.values())


# ---------------------------------------------------------------------------
# ScanningAgent
# ---------------------------------------------------------------------------

class ScanningAgent:
    """
    Follows one variable / tensor through consecutive pipeline steps.

    When the variable is no longer present in a step's outputs the agent
    records a break and releases itself back to the coordinator.

    Note: designed for single-threaded use through ``_BaseConsistencyTeam``.
    The ``BookkeeperAgent`` it writes to is thread-safe, but ``ScanningAgent``
    itself is not and should not be shared across threads.
    """

    def __init__(self, agent_id: int, bookkeeper: BookkeeperAgent):
        self.agent_id = agent_id
        self.bookkeeper = bookkeeper
        self._name: Optional[str] = None
        self._busy = False

    # -- state ----------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def current_name(self) -> Optional[str]:
        return self._name

    # -- lifecycle ------------------------------------------------------------

    def pick_up(self, name: str, step: PipelineStep) -> None:
        """Latch on to *name* as seen in *step*."""
        value = step.inputs.get(name) or step.outputs.get(name)
        self.bookkeeper.note_first_encounter(name, step.name, value)
        self.bookkeeper.note_step(name, step.name, value)
        self._name = name
        self._busy = True

    def follow(self, step: PipelineStep) -> Optional[PathOutcome]:
        """
        Advance through *step*.

        Returns ``None`` if the variable is still live, otherwise the
        ``PathOutcome`` that terminated the path (and clears agent state).
        """
        if not self._busy or self._name is None:
            return None

        if self._name in step.outputs:
            self.bookkeeper.note_step(self._name, step.name, step.outputs[self._name])
            return None

        # Variable missing from this step's outputs → break
        outcome = PathOutcome.NAME_BREAK
        detail  = f"'{self._name}' absent from outputs of step '{step.name}'"
        self.bookkeeper.note_outcome(self._name, outcome, detail)
        self._release()
        return outcome

    def complete(self) -> None:
        """Call when the pipeline ends and the variable survived all steps."""
        if self._busy and self._name:
            self.bookkeeper.note_outcome(self._name, PathOutcome.COMPLETED)
            self._release()

    def _release(self) -> None:
        self._name  = None
        self._busy  = False


# ---------------------------------------------------------------------------
# MarkerAgent
# ---------------------------------------------------------------------------

class MarkerAgent:
    """
    Points at the next unclaimed variable in the queue.

    The two markers leapfrog so a returning scanning agent always finds a
    pre-identified target immediately.
    """

    def __init__(self, marker_id: int):
        self.marker_id = marker_id
        self._marked_name:  Optional[str]          = None
        self._marked_step:  Optional[PipelineStep] = None

    @property
    def has_target(self) -> bool:
        return self._marked_name is not None

    def mark(self, name: str, step: PipelineStep) -> None:
        self._marked_name = name
        self._marked_step = step

    def handoff(self) -> Tuple[Optional[str], Optional[PipelineStep]]:
        """Return and clear the current target so a scanning agent can take it."""
        name, step = self._marked_name, self._marked_step
        self._marked_name = None
        self._marked_step = None
        return name, step


# ---------------------------------------------------------------------------
# Base coordinator  (shared by both teams)
# ---------------------------------------------------------------------------

class _BaseConsistencyTeam:
    """
    Internal base used by both VariableConsistencyTeam and DimensionConsistencyTeam.

    Subclasses override ``_candidate_names`` to control which keys from a step
    are offered to the queue and ``_advance_scanner`` to add domain-specific
    consistency checks before following a path.
    """

    N_SCANNERS = 5

    def __init__(self, team_label: str):
        self.bookkeeper = BookkeeperAgent(team_label=team_label)
        self.scanners   = [ScanningAgent(i, self.bookkeeper) for i in range(self.N_SCANNERS)]
        self.markers    = [MarkerAgent(0), MarkerAgent(1)]
        self._steps:    List[PipelineStep] = []
        # Names seen but not yet assigned to any scanner
        self._pending:  deque = deque()
        # Names currently held by a scanner
        self._active:   Dict[str, ScanningAgent] = {}

    # -- public API -----------------------------------------------------------

    def register_step(self, step: PipelineStep) -> None:
        """Feed one pipeline step to the team and advance all tracking."""
        self._steps.append(step)

        # 1. Bookkeeper records every output variable it has not seen yet.
        for name, value in step.outputs.items():
            self.bookkeeper.note_first_encounter(name, step.name, value)

        # 2. Enqueue new candidate names (not already active or pending).
        for name in self._candidate_names(step):
            if name not in self._active and name not in self._pending:
                self._pending.append(name)

        # 3. Top up both markers from the queue.
        self._refill_markers(step)

        # 4. Advance every busy scanner; collect those that just finished.
        for scanner in self.scanners:
            if scanner.is_busy:
                tracked_name = scanner.current_name
                outcome = self._advance_scanner(scanner, step)
                if outcome is not None:          # scanner is now idle
                    self._active.pop(tracked_name, None)

        # 5. Assign every idle scanner its next target via the marker queue.
        for scanner in self.scanners:
            if not scanner.is_busy:
                self._dispatch_scanner(scanner, step)

    def finalise(self) -> None:
        """Mark still-busy scanners as completed; resolve any untracked records."""
        for scanner in self.scanners:
            if scanner.is_busy:
                scanner.complete()
                self._active.pop(scanner.current_name or "", None)

        # Resolve any variables the bookkeeper noted but no scanner reached.
        last_outputs = set(self._steps[-1].outputs.keys()) if self._steps else set()
        for name, rec in self.bookkeeper.records.items():
            if rec.outcome == PathOutcome.IN_PROGRESS:
                if name in last_outputs:
                    self.bookkeeper.note_outcome(name, PathOutcome.COMPLETED)
                else:
                    self.bookkeeper.note_outcome(
                        name, PathOutcome.DROPPED, "untracked — never assigned to a scanner"
                    )

    def report(self) -> str:
        return self.bookkeeper.report()

    @property
    def has_breaks(self) -> bool:
        return self.bookkeeper.has_breaks

    # -- subclass hooks -------------------------------------------------------

    def _candidate_names(self, step: PipelineStep) -> List[str]:
        """Return the names from this step that should enter the tracking queue."""
        return list(step.outputs.keys())

    def _advance_scanner(
        self, scanner: ScanningAgent, step: PipelineStep
    ) -> Optional[PathOutcome]:
        """Advance *scanner* through *step*; return outcome if path ended."""
        return scanner.follow(step)

    # -- internal helpers -----------------------------------------------------

    def _refill_markers(self, step: PipelineStep) -> None:
        """Give each empty marker the next unclaimed name from the queue."""
        for marker in self.markers:
            if not marker.has_target and self._pending:
                name = self._pending.popleft()
                marker.mark(name, step)

    def _dispatch_scanner(
        self, scanner: ScanningAgent, step: PipelineStep
    ) -> None:
        """
        Assign the next target to *scanner* via the marker leapfrog:

          Marker-0 hands its target to the scanner.
          Marker-1 slides into Marker-0's slot (now the new front marker).
          Marker-1 (vacated) leaps to the next unclaimed name.
        """
        if not self.markers[0].has_target:
            return                          # nothing left to assign

        name, src_step = self.markers[0].handoff()

        # Marker-1 → Marker-0 slot
        if self.markers[1].has_target:
            n1, s1 = self.markers[1].handoff()
            self.markers[0].mark(n1, s1)

        # Advance Marker-1 (now empty) to next unclaimed name
        if self._pending:
            nxt = self._pending.popleft()
            self.markers[1].mark(nxt, step)

        # Give the scanner its assignment
        scanner.pick_up(name, src_step or step)
        self._active[name] = scanner


# ---------------------------------------------------------------------------
# VariableConsistencyTeam
# ---------------------------------------------------------------------------

class VariableConsistencyTeam(_BaseConsistencyTeam):
    """
    Tracks *variable names* through pipeline steps.

    Detects: missing outputs (NAME_BREAK), dropped variables (DROPPED).
    """

    def __init__(self):
        super().__init__(team_label="variable")


# ---------------------------------------------------------------------------
# DimensionConsistencyTeam
# ---------------------------------------------------------------------------

class DimensionConsistencyTeam(_BaseConsistencyTeam):
    """
    Tracks *tensor shapes, dtypes, and dimensional units* through steps.

    On top of name-tracking, every step checks that:
      • Shape rank is preserved (unless the step explicitly declares a reshape).
      • dtype is preserved unless explicitly cast.
      • Complex tensors are not silently narrowed to real.

    ``register_step`` accepts an optional ``allowed_shape_changes`` mapping so
    that intentional reshapes are not flagged as breaks.

        team.register_step(step, allowed_shape_changes={"x": "*"})
        # "*"  = any shape change is allowed for this variable in this step
    """

    def __init__(self):
        super().__init__(team_label="dimension")
        self._last_snapshots: Dict[str, Dict[str, Any]] = {}

    def register_step(
        self,
        step: PipelineStep,
        allowed_shape_changes: Optional[Dict[str, str]] = None,
    ) -> None:
        allowed = allowed_shape_changes or {}

        # Run dimension checks before the base tracking logic
        for name, value in step.outputs.items():
            if name in self._last_snapshots:
                prev = self._last_snapshots[name]
                curr = _snapshot(value)
                break_outcome, detail = self._check_break(
                    name, prev, curr, allowed.get(name, "")
                )
                if break_outcome:
                    self.bookkeeper.note_first_encounter(name, step.name, value)
                    self.bookkeeper.note_outcome(name, break_outcome, detail)
            self._last_snapshots[name] = _snapshot(value)

        super().register_step(step)

    # -- subclass hook --------------------------------------------------------

    def _advance_scanner(
        self, scanner: ScanningAgent, step: PipelineStep
    ) -> Optional[PathOutcome]:
        """Check shape/dtype before delegating to base follow()."""
        name = scanner.current_name
        if name and name in step.outputs:
            prev = self._last_snapshots.get(name, {})
            curr = _snapshot(step.outputs[name])
            outcome, detail = self._check_break(name, prev, curr, "")
            if outcome:
                self.bookkeeper.note_outcome(name, outcome, detail)
                scanner._release()
                return outcome
        return scanner.follow(step)

    @staticmethod
    def _check_break(
        name: str,
        prev: Dict[str, Any],
        curr: Dict[str, Any],
        allowed: str,
    ) -> Tuple[Optional[PathOutcome], str]:
        """Return (outcome, detail) if a consistency break is detected."""
        if allowed == "*":
            return None, ""

        # Shape rank change
        if "shape" in prev and "shape" in curr:
            if len(prev["shape"]) != len(curr["shape"]) and allowed != "reshape":
                return (
                    PathOutcome.SHAPE_BREAK,
                    f"'{name}' rank {len(prev['shape'])} → {len(curr['shape'])}: "
                    f"{prev['shape']} → {curr['shape']}",
                )
            # Dimension size changes (flag unexpected last-dim changes)
            if prev["shape"] != curr["shape"] and allowed not in ("reshape", "batch"):
                return (
                    PathOutcome.SHAPE_BREAK,
                    f"'{name}' shape {prev['shape']} → {curr['shape']}",
                )

        # dtype change
        if "dtype" in prev and "dtype" in curr and prev["dtype"] != curr["dtype"]:
            return (
                PathOutcome.DTYPE_BREAK,
                f"'{name}' dtype {prev['dtype']} → {curr['dtype']}",
            )

        # Complex → real narrowing (unit/information loss)
        if prev.get("complex") and not curr.get("complex"):
            return (
                PathOutcome.UNIT_BREAK,
                f"'{name}' complex narrowed to real — imaginary part discarded",
            )

        return None, ""


# ---------------------------------------------------------------------------
# Convenience: instrument() context manager
# ---------------------------------------------------------------------------

class _StepContext:
    """Returned by ``instrument()``; collects outputs before handing to team."""

    def __init__(self, team: _BaseConsistencyTeam, step_name: str, inputs: Dict[str, Any]):
        self._team      = team
        self._step_name = step_name
        self._inputs    = inputs
        self._outputs:  Dict[str, Any] = {}

    def record_outputs(self, outputs: Dict[str, Any]) -> None:
        self._outputs = outputs

    def _flush(self) -> None:
        step = PipelineStep(
            name=self._step_name,
            inputs=self._inputs,
            outputs=self._outputs,
        )
        self._team.register_step(step)


class _InstrumentCM:
    def __init__(
        self,
        team: _BaseConsistencyTeam,
        step_name: str,
        inputs: Dict[str, Any],
    ):
        self._ctx = _StepContext(team, step_name, inputs)

    def __enter__(self) -> _StepContext:
        return self._ctx

    def __exit__(self, *_) -> None:
        self._ctx._flush()


def instrument(
    team: _BaseConsistencyTeam,
    step_name: str,
    inputs: Optional[Dict[str, Any]] = None,
) -> _InstrumentCM:
    """
    Context-manager helper::

        with instrument(team, "my_step", inputs={"x": x}) as ctx:
            y = transform(x)
            ctx.record_outputs({"y": y})
    """
    return _InstrumentCM(team, step_name, inputs or {})
