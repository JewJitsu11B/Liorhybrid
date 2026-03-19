"""
Tests for the detokenize / output agent team.

Validates:
  - DetokenizeScannerAgent discovers tokenizer, LM head, and generation files.
  - TokenizerHealthAgent source-text checks pass on the actual source files.
  - LMHeadAuditAgent source-text checks pass on the actual source files.
  - GenerationPipelineAgent source-text checks pass on the actual source files.
  - OperationalizationAgent produces a plan covering all four phases.
  - OutputTeamCoordinator full run produces a complete report.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except Exception: pass

import pytest
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


# =============================================================================
# DetokenizeScannerAgent
# =============================================================================

class TestDetokenizeScannerAgent:
    """Tests that the scanner discovers the expected pipeline files."""

    def _agent(self):
        from utils.detokenize_output_agents import DetokenizeScannerAgent
        return DetokenizeScannerAgent(repo_root=REPO_ROOT)

    def test_scan_finds_tokenizer_file(self):
        agent = self._agent()
        locations = agent.scan()
        paths = [loc.path for loc in locations]
        assert any("tokenizer" in p for p in paths), (
            f"Expected a tokenizer file in discovered paths; got: {paths[:10]}"
        )

    def test_scan_finds_lm_head_file(self):
        agent = self._agent()
        locations = agent.scan()
        roles = {loc.role for loc in locations}
        assert "lm_head" in roles, (
            f"Expected 'lm_head' role in discovered locations; roles found: {roles}"
        )

    def test_scan_finds_generation_file(self):
        agent = self._agent()
        locations = agent.scan()
        roles = {loc.role for loc in locations}
        assert "generation" in roles, (
            f"Expected 'generation' role in discovered locations; roles found: {roles}"
        )

    def test_scan_finds_tokenizer_role(self):
        agent = self._agent()
        locations = agent.scan()
        roles = {loc.role for loc in locations}
        assert "tokenizer" in roles, (
            f"Expected 'tokenizer' role in discovered locations; roles found: {roles}"
        )

    def test_relevant_lines_are_integers(self):
        agent = self._agent()
        locations = agent.scan()
        for loc in locations:
            assert all(isinstance(l, int) and l > 0 for l in loc.relevant_lines), (
                f"relevant_lines must be positive integers for {loc.path}: "
                f"{loc.relevant_lines[:5]}"
            )


# =============================================================================
# TokenizerHealthAgent – source-text checks
# =============================================================================

class TestTokenizerHealthAgentSource:
    """Source-text checks against the actual training/tokenizer.py."""

    def _src(self) -> str:
        return (Path(REPO_ROOT) / "training" / "tokenizer.py").read_text(
            encoding="utf-8", errors="replace"
        )

    def _agent(self):
        from utils.detokenize_output_agents import TokenizerHealthAgent
        return TokenizerHealthAgent()

    def test_special_tokens_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "training/tokenizer.py")
        st_finding = next(f for f in findings if "SPECIAL_TOKENS" in f.check)
        assert st_finding.passed, f"Special tokens check failed: {st_finding.details}"

    def test_decode_defined_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "training/tokenizer.py")
        dec_finding = next(f for f in findings if "decode()" in f.check)
        assert dec_finding.passed, f"decode() check failed: {dec_finding.details}"

    def test_encode_defined_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "training/tokenizer.py")
        enc_finding = next(f for f in findings if "encode()" in f.check)
        assert enc_finding.passed, f"encode() check failed: {enc_finding.details}"

    def test_eos_property_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "training/tokenizer.py")
        eos_finding = next(f for f in findings if "eos_token_id" in f.check)
        assert eos_finding.passed, f"eos_token_id check failed: {eos_finding.details}"

    def test_wrong_source_fails_special_tokens(self):
        """A source missing special tokens must fail the check."""
        agent = self._agent()
        bad_src = "class CognitiveTokenizer: pass"
        findings = agent.audit_source(bad_src, "fake.py")
        st_finding = next(f for f in findings if "SPECIAL_TOKENS" in f.check)
        assert not st_finding.passed

    def test_wrong_source_fails_decode(self):
        agent = self._agent()
        bad_src = "class CognitiveTokenizer:\n    pass"
        findings = agent.audit_source(bad_src, "fake.py")
        dec_finding = next(f for f in findings if "decode()" in f.check)
        assert not dec_finding.passed


# =============================================================================
# LMHeadAuditAgent – source-text checks
# =============================================================================

class TestLMHeadAuditAgentSource:
    """Source-text checks against the actual models/language_head.py."""

    def _src(self) -> str:
        return (Path(REPO_ROOT) / "models" / "language_head.py").read_text(
            encoding="utf-8", errors="replace"
        )

    def _agent(self):
        from utils.detokenize_output_agents import LMHeadAuditAgent
        return LMHeadAuditAgent()

    def test_layer_norm_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "models/language_head.py")
        ln_finding = next(f for f in findings if "LayerNorm" in f.check)
        assert ln_finding.passed, f"LayerNorm check failed: {ln_finding.details}"

    def test_output_projection_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "models/language_head.py")
        proj_finding = next(f for f in findings if "output_projection" in f.check)
        assert proj_finding.passed, f"output_projection check failed: {proj_finding.details}"

    def test_weight_tying_guarded_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "models/language_head.py")
        tie_finding = next(f for f in findings if "tie_weights" in f.check)
        assert tie_finding.passed, f"weight tying check failed: {tie_finding.details}"

    def test_missing_layer_norm_fails(self):
        agent = self._agent()
        bad_src = (
            "class LanguageModelHead:\n"
            "    def __init__(self, d_model, vocab_size):\n"
            "        self.output_projection = nn.Linear(d_model, vocab_size)\n"
        )
        findings = agent.audit_source(bad_src, "fake.py")
        ln_finding = next(f for f in findings if "LayerNorm" in f.check)
        assert not ln_finding.passed

    def test_missing_output_projection_fails(self):
        agent = self._agent()
        bad_src = "class LanguageModelHead:\n    pass"
        findings = agent.audit_source(bad_src, "fake.py")
        proj_finding = next(f for f in findings if "output_projection" in f.check)
        assert not proj_finding.passed


# =============================================================================
# GenerationPipelineAgent – source-text checks
# =============================================================================

class TestGenerationPipelineAgentSource:
    """Source-text checks against the actual inference/inference.py."""

    def _src(self) -> str:
        return (Path(REPO_ROOT) / "inference" / "inference.py").read_text(
            encoding="utf-8", errors="replace"
        )

    def _agent(self):
        from utils.detokenize_output_agents import GenerationPipelineAgent
        return GenerationPipelineAgent()

    def test_eos_termination_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "inference/inference.py")
        eos_finding = next(f for f in findings if "EOS" in f.check)
        assert eos_finding.passed, f"EOS termination check failed: {eos_finding.details}"

    def test_seq_len_clipping_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "inference/inference.py")
        clip_finding = next(f for f in findings if "max_seq_len" in f.check)
        assert clip_finding.passed, f"max_seq_len clipping check failed: {clip_finding.details}"

    def test_field_evolve_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "inference/inference.py")
        evolve_finding = next(f for f in findings if "evolve_step" in f.check)
        assert evolve_finding.passed, f"evolve_step check failed: {evolve_finding.details}"

    def test_decode_called_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "inference/inference.py")
        decode_finding = next(f for f in findings if "tokenizer.decode" in f.check)
        assert decode_finding.passed, f"tokenizer.decode check failed: {decode_finding.details}"

    def test_entropy_gating_check_passes(self):
        agent = self._agent()
        findings = agent.audit_source(self._src(), "inference/inference.py")
        gate_finding = next(f for f in findings if "Entropy gating" in f.check)
        assert gate_finding.passed, f"Entropy gating check failed: {gate_finding.details}"

    def test_missing_eos_fails(self):
        agent = self._agent()
        bad_src = "def generate(self):\n    for i in range(100):\n        pass"
        findings = agent.audit_source(bad_src, "fake.py")
        eos_finding = next(f for f in findings if "EOS" in f.check)
        assert not eos_finding.passed

    def test_missing_decode_fails(self):
        agent = self._agent()
        bad_src = "def generate(self):\n    return ''"
        findings = agent.audit_source(bad_src, "fake.py")
        decode_finding = next(f for f in findings if "tokenizer.decode" in f.check)
        assert not decode_finding.passed


# =============================================================================
# GenerationPipelineAgent – numerical checks
# =============================================================================

class TestGenerationPipelineNumerical:
    """Numerical runtime checks for the generation pipeline helpers."""

    def _agent(self):
        from utils.detokenize_output_agents import GenerationPipelineAgent
        return GenerationPipelineAgent()

    def test_selector_probs_sums_to_one(self):
        import torch
        agent = self._agent()
        findings = agent._check_selector_probs(torch)
        assert findings, "Expected at least one finding from _check_selector_probs"
        finding = findings[0]
        assert finding.passed, f"selector_probs check failed: {finding.details}"

    def test_entropy_from_probs_non_negative(self):
        import torch
        from inference.inference import InferenceEngine  # type: ignore
        agent = self._agent()
        findings = agent._check_entropy_from_probs(torch, InferenceEngine._entropy_from_probs)
        assert findings
        assert findings[0].passed, f"entropy_from_probs check failed: {findings[0].details}"

    def test_entropy_gate_zero_tau_is_identity(self):
        import torch
        from inference.inference import InferenceEngine  # type: ignore
        agent = self._agent()
        findings = agent._check_entropy_gate_zero_tau(torch, InferenceEngine._entropy_gate)
        assert findings
        assert findings[0].passed, f"entropy_gate tau=0 check failed: {findings[0].details}"


# =============================================================================
# LMHeadAuditAgent – numerical checks
# =============================================================================

class TestLMHeadAuditNumerical:
    """Numerical runtime checks for LanguageModelHead."""

    def _agent(self):
        from utils.detokenize_output_agents import LMHeadAuditAgent
        return LMHeadAuditAgent()

    def test_output_shape_correct(self):
        agent = self._agent()
        findings = agent.audit_numerical(d_model=32, vocab_size=64)
        shape_finding = next((f for f in findings if "shape" in f.check.lower()), None)
        assert shape_finding is not None, "Expected a shape finding"
        assert shape_finding.passed, f"LM head shape check failed: {shape_finding.details}"

    def test_output_logits_finite(self):
        agent = self._agent()
        findings = agent.audit_numerical(d_model=32, vocab_size=64)
        finite_finding = next((f for f in findings if "finite" in f.check.lower()), None)
        assert finite_finding is not None, "Expected a finiteness finding"
        assert finite_finding.passed, f"LM head finite check failed: {finite_finding.details}"


# =============================================================================
# OperationalizationAgent
# =============================================================================

class TestOperationalizationAgent:
    """Tests that the plan covers all four phases and summarises critical failures."""

    def _agent(self):
        from utils.detokenize_output_agents import OperationalizationAgent
        return OperationalizationAgent()

    def test_plan_covers_four_phases(self):
        agent = self._agent()
        plan = agent.build_plan([])
        plan_text = "\n".join(plan)
        for phase in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
            assert phase in plan_text, f"Operationalization plan missing {phase}"

    def test_plan_lists_critical_failures(self):
        from utils.detokenize_output_agents import DetokenizeFinding
        agent = self._agent()
        failing = DetokenizeFinding(
            agent="Test Agent",
            check="Some critical check",
            passed=False,
            severity="critical",
            details="This is broken.",
            file_path="some/file.py",
            fix_hint="Fix it.",
        )
        plan = agent.build_plan([failing])
        plan_text = "\n".join(plan)
        assert "Some critical check" in plan_text
        assert "Fix it." in plan_text

    def test_plan_summary_line_present(self):
        from utils.detokenize_output_agents import DetokenizeFinding
        agent = self._agent()
        passing = DetokenizeFinding(
            agent="Test Agent",
            check="A passing check",
            passed=True,
            severity="info",
            details="All good.",
            file_path="some/file.py",
            fix_hint="",
        )
        plan = agent.build_plan([passing])
        plan_text = "\n".join(plan)
        assert "1 passed" in plan_text
        assert "0 failed" in plan_text


# =============================================================================
# OutputTeamCoordinator – integration tests
# =============================================================================

class TestOutputTeamCoordinator:
    """Integration tests for the full coordinator run."""

    def test_run_produces_report(self):
        from utils.detokenize_output_agents import OutputTeamCoordinator, DetokenizeOutputReport
        coord = OutputTeamCoordinator(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        assert isinstance(report, DetokenizeOutputReport)
        assert report.locations
        assert report.findings
        assert report.action_log
        assert report.operationalization_plan

    def test_action_log_has_summary(self):
        from utils.detokenize_output_agents import OutputTeamCoordinator
        coord = OutputTeamCoordinator(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        log_text = "\n".join(report.action_log)
        assert "Summary" in log_text

    def test_action_log_has_discovered_files(self):
        from utils.detokenize_output_agents import OutputTeamCoordinator
        coord = OutputTeamCoordinator(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        log_text = "\n".join(report.action_log)
        assert "Discovered Pipeline Files" in log_text

    def test_plan_has_all_phases(self):
        from utils.detokenize_output_agents import OutputTeamCoordinator
        coord = OutputTeamCoordinator(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        plan_text = "\n".join(report.operationalization_plan)
        for phase in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
            assert phase in plan_text, f"Plan missing {phase}"

    def test_source_text_checks_all_pass(self):
        """All source-text (non-numerical) checks must pass on the current codebase."""
        from utils.detokenize_output_agents import OutputTeamCoordinator
        coord = OutputTeamCoordinator(repo_root=REPO_ROOT)
        report = coord.run(run_numerical=False)
        critical_failures = report.critical_failures
        assert not critical_failures, (
            "Expected no CRITICAL failures on the current source code.\n"
            + "\n".join(
                f"  [{f.severity.upper()}] {f.agent}: {f.check}\n"
                f"    {f.details}"
                for f in critical_failures
            )
        )
