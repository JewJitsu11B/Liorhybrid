"""
Tests for the pipeline consistency agent teams.

Validates:
  - BookkeeperAgent records first encounters and outcomes correctly.
  - ScanningAgent follows a variable through steps and detects breaks.
  - MarkerAgent leapfrogs correctly so scanning agents always have a next target.
  - VariableConsistencyTeam end-to-end: clean path + name-break detection.
  - DimensionConsistencyTeam end-to-end: clean path + shape/dtype/unit breaks.
  - instrument() context manager wires steps into a team transparently.
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except Exception:
    pass

import pytest

from ..utils.pipeline_consistency_agents import (
    BookkeeperAgent,
    CheckpointSurveySpecialistAgent,
    DimensionConsistencyTeam,
    LeadCheckpointInferenceAgent,
    MarkerAgent,
    PathOutcome,
    PathRecord,
    PipelineSidesReadySpecialistAgent,
    PipelineStep,
    ScanningAgent,
    VariableConsistencyTeam,
    _snapshot,
    instrument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(name: str, keys: list, values: list) -> PipelineStep:
    d = dict(zip(keys, values))
    return PipelineStep(name=name, inputs={}, outputs=d)


class _FakeTensor:
    """Minimal stand-in for a torch.Tensor (no torch dependency in tests)."""
    def __init__(self, shape, dtype="float32", complex_=False):
        self.shape = shape
        self.dtype = dtype
        self._complex = complex_

    def is_complex(self):
        return self._complex


# ---------------------------------------------------------------------------
# _snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_captures_shape_and_dtype(self):
        t = _FakeTensor((2, 64, 6))
        s = _snapshot(t)
        assert s["shape"] == (2, 64, 6)
        assert s["dtype"] == "float32"

    def test_fallback_repr_for_non_tensor(self):
        s = _snapshot(42)
        assert "repr" in s

    def test_complex_flag(self):
        t = _FakeTensor((4,), complex_=True)
        s = _snapshot(t)
        assert s["complex"] is True


# ---------------------------------------------------------------------------
# BookkeeperAgent
# ---------------------------------------------------------------------------

class TestBookkeeperAgent:
    def test_first_encounter_recorded(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("x", "step_A", _FakeTensor((2, 4)))
        assert "x" in bk.records
        assert bk.records["x"].first_step == "step_A"

    def test_duplicate_first_encounter_ignored(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("x", "step_A", _FakeTensor((2, 4)))
        bk.note_first_encounter("x", "step_B", _FakeTensor((2, 4)))
        assert bk.records["x"].first_step == "step_A"  # first wins

    def test_note_outcome_completed(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("x", "step_A", _FakeTensor((2, 4)))
        bk.note_outcome("x", PathOutcome.COMPLETED)
        assert bk.records["x"].outcome == PathOutcome.COMPLETED

    def test_note_outcome_break_with_detail(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("x", "step_A", _FakeTensor((2, 4)))
        bk.note_outcome("x", PathOutcome.NAME_BREAK, "wrong name")
        assert bk.records["x"].outcome == PathOutcome.NAME_BREAK
        assert bk.records["x"].break_detail == "wrong name"

    def test_has_breaks_false_when_all_completed(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("x", "s", _FakeTensor((1,)))
        bk.note_outcome("x", PathOutcome.COMPLETED)
        assert not bk.has_breaks

    def test_has_breaks_true_after_name_break(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("x", "s", _FakeTensor((1,)))
        bk.note_outcome("x", PathOutcome.NAME_BREAK, "gone")
        assert bk.has_breaks

    def test_report_includes_variable_names(self):
        bk = BookkeeperAgent("var")
        bk.note_first_encounter("embedding", "step_A", _FakeTensor((2, 512)))
        bk.note_outcome("embedding", PathOutcome.COMPLETED)
        report = bk.report()
        assert "embedding" in report
        assert "completed" in report


# ---------------------------------------------------------------------------
# ScanningAgent
# ---------------------------------------------------------------------------

class TestScanningAgent:
    def _bk(self):
        return BookkeeperAgent("var")

    def test_pick_up_makes_agent_busy(self):
        bk = self._bk()
        agent = ScanningAgent(0, bk)
        step = _step("s0", ["x"], [_FakeTensor((2,))])
        agent.pick_up("x", step)
        assert agent.is_busy
        assert agent.current_name == "x"

    def test_follow_returns_none_when_variable_present(self):
        bk = self._bk()
        agent = ScanningAgent(0, bk)
        s0 = _step("s0", ["x"], [_FakeTensor((2,))])
        s1 = _step("s1", ["x"], [_FakeTensor((2,))])
        agent.pick_up("x", s0)
        result = agent.follow(s1)
        assert result is None
        assert agent.is_busy

    def test_follow_returns_name_break_when_missing(self):
        bk = self._bk()
        agent = ScanningAgent(0, bk)
        s0 = _step("s0", ["x"], [_FakeTensor((2,))])
        s1 = _step("s1", ["y"], [_FakeTensor((2,))])  # x absent
        agent.pick_up("x", s0)
        outcome = agent.follow(s1)
        assert outcome == PathOutcome.NAME_BREAK
        assert not agent.is_busy

    def test_complete_marks_path_completed(self):
        bk = self._bk()
        agent = ScanningAgent(0, bk)
        s0 = _step("s0", ["x"], [_FakeTensor((2,))])
        agent.pick_up("x", s0)
        agent.complete()
        assert not agent.is_busy
        assert bk.records["x"].outcome == PathOutcome.COMPLETED


# ---------------------------------------------------------------------------
# MarkerAgent
# ---------------------------------------------------------------------------

class TestMarkerAgent:
    def test_mark_and_handoff(self):
        m = MarkerAgent(0)
        step = _step("s0", ["v"], [1])
        m.mark("v", step)
        assert m.has_target
        name, s = m.handoff()
        assert name == "v"
        assert s is step
        assert not m.has_target

    def test_handoff_empty_returns_none(self):
        m = MarkerAgent(0)
        name, step = m.handoff()
        assert name is None
        assert step is None

    def test_leapfrog_two_markers(self):
        """Marker-0 hands off, marker-1 slides into marker-0's slot."""
        m0, m1 = MarkerAgent(0), MarkerAgent(1)
        s = _step("s", ["a", "b", "c"], [1, 2, 3])
        m0.mark("a", s)
        m1.mark("b", s)

        name0, _ = m0.handoff()   # scanning agent takes "a"
        assert name0 == "a"
        assert not m0.has_target  # m0 is now free to leap to "c"

        # m1 still holds "b" — scanning agent's next target
        assert m1.has_target
        assert m1._marked_name == "b"


# ---------------------------------------------------------------------------
# VariableConsistencyTeam
# ---------------------------------------------------------------------------

class TestVariableConsistencyTeam:
    def _make_steps(self):
        t = _FakeTensor
        # All output variables must persist through every step for a clean run.
        return [
            PipelineStep("encode",  inputs={}, outputs={"embedding": t((2, 512))}),
            PipelineStep("project", inputs={}, outputs={"embedding": t((2, 512)), "metric": t((2, 512))}),
            PipelineStep("select",  inputs={}, outputs={"embedding": t((2, 512)), "metric": t((2, 512))}),
        ]

    def test_clean_pipeline_no_breaks(self):
        team = VariableConsistencyTeam()
        for step in self._make_steps():
            team.register_step(step)
        team.finalise()
        assert not team.has_breaks

    def test_name_break_detected(self):
        team = VariableConsistencyTeam()
        t = _FakeTensor
        team.register_step(PipelineStep("s0", {}, {"embedding": t((2, 512))}))
        # embedding disappears — renamed to hidden_state without alias
        team.register_step(PipelineStep("s1", {}, {"hidden_state": t((2, 512))}))
        team.finalise()
        assert team.has_breaks

    def test_bookkeeper_records_all_variables(self):
        team = VariableConsistencyTeam()
        for step in self._make_steps():
            team.register_step(step)
        team.finalise()
        # All unique output names across all steps should be recorded
        all_names = {"embedding", "metric"}
        recorded = set(team.bookkeeper.records.keys())
        assert all_names.issubset(recorded)

    def test_report_string_non_empty(self):
        team = VariableConsistencyTeam()
        team.register_step(_step("s0", ["x"], [_FakeTensor((2,))]))
        team.finalise()
        assert len(team.report()) > 0

    def test_more_than_five_variables_processed(self):
        """All 5 scanners plus marker leapfrog must handle > 5 variables."""
        team = VariableConsistencyTeam()
        keys = [f"v{i}" for i in range(8)]
        vals = [_FakeTensor((2,)) for _ in keys]
        team.register_step(PipelineStep("s0", {}, dict(zip(keys, vals))))
        team.register_step(PipelineStep("s1", {}, dict(zip(keys, vals))))
        team.finalise()
        # All 8 should be tracked
        assert len(team.bookkeeper.records) == 8


# ---------------------------------------------------------------------------
# DimensionConsistencyTeam
# ---------------------------------------------------------------------------

class TestDimensionConsistencyTeam:
    def test_clean_pipeline_no_breaks(self):
        team = DimensionConsistencyTeam()
        t = _FakeTensor
        team.register_step(PipelineStep("s0", {}, {"x": t((2, 512))}))
        team.register_step(PipelineStep("s1", {}, {"x": t((2, 512))}))
        team.finalise()
        assert not team.has_breaks

    def test_shape_break_detected(self):
        team = DimensionConsistencyTeam()
        t = _FakeTensor
        team.register_step(PipelineStep("s0", {}, {"x": t((2, 512))}))
        team.register_step(PipelineStep("s1", {}, {"x": t((2, 256))}))  # dim changed
        team.finalise()
        assert team.has_breaks
        rec = team.bookkeeper.records.get("x")
        assert rec is not None
        assert rec.outcome == PathOutcome.SHAPE_BREAK

    def test_dtype_break_detected(self):
        team = DimensionConsistencyTeam()
        t = _FakeTensor
        team.register_step(PipelineStep("s0", {}, {"x": t((4,), dtype="float32")}))
        team.register_step(PipelineStep("s1", {}, {"x": t((4,), dtype="float16")}))
        team.finalise()
        assert team.has_breaks
        rec = team.bookkeeper.records.get("x")
        assert rec.outcome == PathOutcome.DTYPE_BREAK

    def test_complex_narrowing_detected(self):
        team = DimensionConsistencyTeam()
        team.register_step(PipelineStep("s0", {}, {"x": _FakeTensor((4,), complex_=True)}))
        team.register_step(PipelineStep("s1", {}, {"x": _FakeTensor((4,), complex_=False)}))
        team.finalise()
        assert team.has_breaks
        rec = team.bookkeeper.records.get("x")
        assert rec.outcome == PathOutcome.UNIT_BREAK

    def test_allowed_reshape_not_flagged(self):
        team = DimensionConsistencyTeam()
        t = _FakeTensor
        team.register_step(PipelineStep("s0", {}, {"x": t((2, 512))}))
        team.register_step(
            PipelineStep("s1", {}, {"x": t((2, 256))}),
            allowed_shape_changes={"x": "reshape"},
        )
        team.finalise()
        assert not team.has_breaks

    def test_wildcard_allowed_any_change(self):
        team = DimensionConsistencyTeam()
        t = _FakeTensor
        team.register_step(PipelineStep("s0", {}, {"x": t((2, 512), dtype="float32")}))
        team.register_step(
            PipelineStep("s1", {}, {"x": t((8, 64, 6), dtype="float16")}),
            allowed_shape_changes={"x": "*"},
        )
        team.finalise()
        assert not team.has_breaks


# ---------------------------------------------------------------------------
# instrument() context manager
# ---------------------------------------------------------------------------

class TestInstrumentContextManager:
    def test_instrument_registers_step(self):
        team = VariableConsistencyTeam()
        inp = {"emb": _FakeTensor((2, 128))}
        with instrument(team, "my_step", inputs=inp) as ctx:
            out = _FakeTensor((2, 128))
            ctx.record_outputs({"emb": out})

        assert "emb" in team.bookkeeper.records

    def test_instrument_dimension_team(self):
        team = DimensionConsistencyTeam()
        with instrument(team, "proj", inputs={"z": _FakeTensor((2, 64))}) as ctx:
            ctx.record_outputs({"z": _FakeTensor((2, 64))})
        with instrument(team, "bad_proj", inputs={"z": _FakeTensor((2, 64))}) as ctx:
            ctx.record_outputs({"z": _FakeTensor((2, 32))})  # shape break
        team.finalise()
        assert team.has_breaks


class TestLeadCheckpointInferenceAgent:
    def test_lead_agent_assigns_two_specialists(self):
        agent = LeadCheckpointInferenceAgent()
        report = agent.run()
        assert CheckpointSurveySpecialistAgent.NAME in report.assigned_specialists
        assert PipelineSidesReadySpecialistAgent.NAME in report.assigned_specialists
        assert len(report.assigned_specialists) == 2

    def test_lead_agent_finds_periodic_and_manual_quit_checkpoint_signals(self):
        agent = LeadCheckpointInferenceAgent()
        report = agent.run()
        checks = {f.check: f for f in report.findings}
        assert checks["Periodic checkpoint cadence setting exists"].passed
        assert checks["Manual quit checkpoint save exists"].passed
