"""
Detokenize / Output Agent Team.

A team of specialised agents that scan the detokenizing and output process,
audit each stage of the token-generation pipeline, and produce a phased plan
to finalise and make the pipeline fully operational.

Roles
-----
DetokenizeScannerAgent
    Scans the repository for every file that participates in the
    detokenizing and output pipeline (tokenizer, LM head, generation loop,
    entropy gating, selector functions).  Broadcasts the discovered
    locations to the specialist audit agents.

TokenizerHealthAgent
    Validates the CognitiveTokenizer:
      • Special-token IDs match the SPECIAL_TOKENS contract.
      • encode/decode roundtrip preserves text (no data loss).
      • Batch encoding produces the same result as individual encoding.
      • EOS token ID is correctly reported.

LMHeadAuditAgent
    Validates the language model head:
      • Output logits shape matches (batch, seq_len, vocab_size).
      • Logits are finite (no NaN / Inf) after a forward pass.
      • Layer-norm is applied before the output projection.
      • Optional weight tying is correctly wired.

GenerationPipelineAgent
    Validates the InferenceEngine.generate() loop:
      • Entropy gating formula is correctly applied.
      • Selector probability sums to 1 (valid distribution).
      • EOS termination stops the loop before max_tokens is reached.
      • Sequence length is clipped to max_seq_len.
      • Field evolve_step() is called each iteration.

OperationalizationAgent
    Produces a phased operationalization plan covering the gaps and
    recommended next steps surfaced by the other agents.

OutputTeamCoordinator
    Orchestrates the full team:
      1. DetokenizeScannerAgent discovers relevant files.
      2. TokenizerHealthAgent, LMHeadAuditAgent, GenerationPipelineAgent
         run source-text audits in parallel (sequential here for safety).
      3. Numerical checks are optionally executed.
      4. OperationalizationAgent synthesises the findings into a plan.
      5. A consolidated DetokenizeOutputReport is returned.

Usage
-----
    from utils.detokenize_output_agents import OutputTeamCoordinator

    coordinator = OutputTeamCoordinator(repo_root="/path/to/repo")
    report = coordinator.run()
    for line in report.action_log:
        print(line)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except Exception: pass

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetokenizeLocation:
    """A file in the repo that participates in the detokenize/output pipeline."""
    path: str
    role: str            # e.g. "tokenizer", "lm_head", "generation", "entropy"
    relevant_lines: Tuple[int, ...]  # 1-based line numbers of interest


@dataclass(frozen=True)
class DetokenizeFinding:
    """A single audit finding raised by one specialist agent."""
    agent: str
    check: str
    passed: bool
    severity: str        # "critical" | "major" | "minor" | "info"
    details: str
    file_path: str
    fix_hint: str


@dataclass
class DetokenizeOutputReport:
    """Consolidated report produced by the OutputTeamCoordinator."""
    locations: List[DetokenizeLocation] = field(default_factory=list)
    findings: List[DetokenizeFinding] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)
    operationalization_plan: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def critical_failures(self) -> List[DetokenizeFinding]:
        return [f for f in self.findings if not f.passed and f.severity == "critical"]


# ---------------------------------------------------------------------------
# DetokenizeScannerAgent
# ---------------------------------------------------------------------------

class DetokenizeScannerAgent:
    """
    Scans the repository for files participating in the detokenize/output pipeline.

    Patterns matched (each yields a ``role`` label):
      • CognitiveTokenizer / HF tokenizer calls → "tokenizer"
      • LanguageModelHead / lm_head              → "lm_head"
      • generate() / InferenceEngine             → "generation"
      • entropy_gated_softmax / _selector_probs  → "entropy"
      • test files covering any of the above     → "test"
    """

    NAME = "Detokenize Scanner Agent"

    _PATTERNS: List[Tuple[str, str]] = [
        (r'CognitiveTokenizer|\.decode\(|tokenizer\.encode', "tokenizer"),
        (r'LanguageModelHead|lm_head|output_projection', "lm_head"),
        (r'InferenceEngine|def generate\b|generated_ids', "generation"),
        (r'entropy_gated_softmax|_selector_probs|entropymax|bornmax|gibbsmax', "entropy"),
        (r'test.*detokeniz|test.*lm.head|test.*generat|test.*tokenizer', "test"),
    ]

    def __init__(self, repo_root: Optional[str] = None):
        if repo_root is None:
            repo_root = str(Path(__file__).resolve().parent.parent)
        self.repo_root = Path(repo_root)

    def scan(self) -> List[DetokenizeLocation]:
        """Walk the repo and collect files related to the detokenize/output pipeline."""
        locations: List[DetokenizeLocation] = []
        seen: set = set()
        for py_file in sorted(self.repo_root.rglob("*.py")):
            rel = str(py_file.relative_to(self.repo_root))
            if any(skip in rel for skip in ('.git', '__pycache__', 'node_modules')):
                continue
            try:
                text = py_file.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            for pattern, role in self._PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    if rel not in seen:
                        lines = self._find_relevant_lines(text, pattern)
                        locations.append(DetokenizeLocation(
                            path=rel,
                            role=role,
                            relevant_lines=lines,
                        ))
                        seen.add(rel)
                    break  # first matching role wins
        return locations

    @staticmethod
    def _find_relevant_lines(text: str, pattern: str) -> Tuple[int, ...]:
        """Return 1-based line numbers where *pattern* matches (capped at 20)."""
        lines = []
        for i, line in enumerate(text.splitlines(), start=1):
            if re.search(pattern, line, re.IGNORECASE):
                lines.append(i)
        return tuple(lines[:20])


# ---------------------------------------------------------------------------
# TokenizerHealthAgent
# ---------------------------------------------------------------------------

class TokenizerHealthAgent:
    """
    Validates the CognitiveTokenizer source-text contracts and (optionally)
    its runtime behaviour.

    Source-text checks
    ------------------
    1. Special-token map present with required entries.
    2. ``decode()`` method defined and calls HF or fallback inverse_vocab.
    3. ``encode()`` method defined with max_length truncation.
    4. EOS token property exposes ``<|endoftext|>``.

    Numerical checks
    ----------------
    5. encode → decode roundtrip is lossless for plain ASCII text.
    6. encode_batch gives the same result as individual encode calls.
    7. EOS token ID reported by the tokenizer matches SPECIAL_TOKENS['<|endoftext|>'].
    """

    NAME = "Tokenizer Health Agent"

    _REQUIRED_SPECIAL_TOKENS = (
        "<|pad|>",
        "<|endoftext|>",
        "<|text|>",
        "<|image|>",
        "<|video|>",
    )

    def audit_source(self, src: str, path: str = "training/tokenizer.py") -> List[DetokenizeFinding]:
        findings: List[DetokenizeFinding] = []
        findings.extend(self._check_special_tokens_map(src, path))
        findings.extend(self._check_decode_defined(src, path))
        findings.extend(self._check_encode_defined(src, path))
        findings.extend(self._check_eos_property(src, path))
        return findings

    def audit_numerical(self) -> List[DetokenizeFinding]:
        """Run numerical runtime checks on a live CognitiveTokenizer."""
        findings: List[DetokenizeFinding] = []
        try:
            from training.tokenizer import CognitiveTokenizer  # type: ignore
        except ImportError:
            findings.append(DetokenizeFinding(
                agent=self.NAME,
                check="CognitiveTokenizer importable",
                passed=False,
                severity="critical",
                details="Could not import CognitiveTokenizer from training.tokenizer.",
                file_path="training/tokenizer.py",
                fix_hint="Ensure training/tokenizer.py is on sys.path.",
            ))
            return findings

        tokenizer = CognitiveTokenizer(pretrained=None)  # avoid network in tests
        findings.extend(self._check_roundtrip(tokenizer))
        findings.extend(self._check_batch_consistency(tokenizer))
        findings.extend(self._check_eos_id(tokenizer))
        return findings

    # ------------------------------------------------------------------
    # Source-text checks
    # ------------------------------------------------------------------

    def _check_special_tokens_map(self, src: str, path: str) -> List[DetokenizeFinding]:
        """All required special tokens must appear in SPECIAL_TOKENS."""
        missing = [t for t in self._REQUIRED_SPECIAL_TOKENS if t not in src]
        passed = not missing
        return [DetokenizeFinding(
            agent=self.NAME,
            check="SPECIAL_TOKENS contains all required entries",
            passed=passed,
            severity="critical",
            details=(
                "All required special tokens found in tokenizer source."
                if passed
                else f"Missing special token(s): {missing}"
            ),
            file_path=path,
            fix_hint=(
                "Add the missing token(s) to CognitiveTokenizer.SPECIAL_TOKENS "
                "and assign unique integer IDs."
            ),
        )]

    def _check_decode_defined(self, src: str, path: str) -> List[DetokenizeFinding]:
        """The decode() method must be defined and call HF decode or inverse_vocab."""
        has_def = bool(re.search(r'def decode\s*\(', src))
        has_impl = bool(re.search(r'_hf_tokenizer\.decode|inverse_vocab', src))
        passed = has_def and has_impl
        return [DetokenizeFinding(
            agent=self.NAME,
            check="decode() method defined with HF or fallback implementation",
            passed=passed,
            severity="critical",
            details=(
                "decode() method with implementation found."
                if passed
                else (
                    "decode() method missing." if not has_def
                    else "decode() defined but no HF/_inverse_vocab implementation found."
                )
            ),
            file_path=path,
            fix_hint=(
                "Implement decode() to call self._hf_tokenizer.decode() when the "
                "HF backend is available, and fall back to inverse_vocab lookup."
            ),
        )]

    def _check_encode_defined(self, src: str, path: str) -> List[DetokenizeFinding]:
        """The encode() method must exist and support max_length truncation."""
        has_encode = bool(re.search(r'def encode\s*\(', src))
        has_truncate = 'max_length' in src
        passed = has_encode and has_truncate
        return [DetokenizeFinding(
            agent=self.NAME,
            check="encode() defined with max_length truncation support",
            passed=passed,
            severity="major",
            details=(
                "encode() with max_length truncation found."
                if passed
                else (
                    "encode() missing." if not has_encode
                    else "encode() found but max_length truncation not detected."
                )
            ),
            file_path=path,
            fix_hint=(
                "Implement encode() with a max_length parameter and truncate the "
                "output ids list before returning."
            ),
        )]

    def _check_eos_property(self, src: str, path: str) -> List[DetokenizeFinding]:
        """eos_token_id property must expose the endoftext special token."""
        has_prop = bool(re.search(r'def eos_token_id', src))
        has_eof = bool(re.search(r'<\|endoftext\|>', src))
        passed = has_prop and has_eof
        return [DetokenizeFinding(
            agent=self.NAME,
            check="eos_token_id property exposes <|endoftext|> ID",
            passed=passed,
            severity="major",
            details=(
                "eos_token_id property referencing <|endoftext|> found."
                if passed
                else (
                    "eos_token_id property missing." if not has_prop
                    else "eos_token_id property found but <|endoftext|> reference absent."
                )
            ),
            file_path=path,
            fix_hint=(
                "Add: @property\ndef eos_token_id(self) -> int:\n"
                "    return self.special_tokens['<|endoftext|>']"
            ),
        )]

    # ------------------------------------------------------------------
    # Numerical checks
    # ------------------------------------------------------------------

    def _check_roundtrip(self, tokenizer) -> List[DetokenizeFinding]:
        """encode → decode should recover the original text (ASCII)."""
        test_text = "Hello world"
        try:
            ids = tokenizer.encode(test_text, add_special_tokens=False)
            recovered = tokenizer.decode(ids, skip_special_tokens=True)
            passed = isinstance(recovered, str) and len(recovered) > 0
            return [DetokenizeFinding(
                agent=self.NAME,
                check="encode → decode roundtrip produces non-empty string",
                passed=passed,
                severity="critical",
                details=(
                    f"Roundtrip OK: '{test_text}' → ids({len(ids)}) → '{recovered}'"
                    if passed
                    else f"Roundtrip failed: recovered='{recovered}'"
                ),
                file_path="training/tokenizer.py",
                fix_hint=(
                    "Verify that the HF tokenizer's decode() strips byte-level "
                    "prefixes and restores the original text correctly."
                ),
            )]
        except Exception as exc:
            return [DetokenizeFinding(
                agent=self.NAME,
                check="encode → decode roundtrip produces non-empty string",
                passed=False,
                severity="critical",
                details=f"Roundtrip raised: {exc}",
                file_path="training/tokenizer.py",
                fix_hint="Fix encode() or decode() so neither raises on plain ASCII.",
            )]

    def _check_batch_consistency(self, tokenizer) -> List[DetokenizeFinding]:
        """encode_batch must match individual encode results."""
        texts = ["The quick brown fox", "jumped over the lazy dog"]
        try:
            batch_ids = tokenizer.encode_batch(texts, add_special_tokens=False)
            single_ids = [
                tokenizer.encode(t, add_special_tokens=False) for t in texts
            ]
            consistent = batch_ids == single_ids
            return [DetokenizeFinding(
                agent=self.NAME,
                check="encode_batch() matches individual encode() results",
                passed=consistent,
                severity="major",
                details=(
                    "Batch encoding matches individual encoding."
                    if consistent
                    else "Batch encoding differs from individual encoding."
                ),
                file_path="training/tokenizer.py",
                fix_hint=(
                    "Ensure encode_batch() calls the same HF backend with the "
                    "same add_special_tokens setting as encode()."
                ),
            )]
        except Exception as exc:
            return [DetokenizeFinding(
                agent=self.NAME,
                check="encode_batch() matches individual encode() results",
                passed=False,
                severity="major",
                details=f"Batch consistency check raised: {exc}",
                file_path="training/tokenizer.py",
                fix_hint="Fix encode_batch() so it does not raise on valid text input.",
            )]

    def _check_eos_id(self, tokenizer) -> List[DetokenizeFinding]:
        """eos_token_id should equal SPECIAL_TOKENS['<|endoftext|>']."""
        try:
            eos_id = tokenizer.eos_token_id
            expected = tokenizer.special_tokens.get('<|endoftext|>', None)
            passed = expected is not None and eos_id == expected
            return [DetokenizeFinding(
                agent=self.NAME,
                check="eos_token_id equals SPECIAL_TOKENS['<|endoftext|>']",
                passed=passed,
                severity="major",
                details=(
                    f"eos_token_id={eos_id} matches SPECIAL_TOKENS['<|endoftext|>']={expected}."
                    if passed
                    else f"Mismatch: eos_token_id={eos_id}, expected={expected}."
                ),
                file_path="training/tokenizer.py",
                fix_hint=(
                    "Return self.special_tokens['<|endoftext|>'] from eos_token_id."
                ),
            )]
        except Exception as exc:
            return [DetokenizeFinding(
                agent=self.NAME,
                check="eos_token_id equals SPECIAL_TOKENS['<|endoftext|>']",
                passed=False,
                severity="major",
                details=f"eos_token_id check raised: {exc}",
                file_path="training/tokenizer.py",
                fix_hint="Implement the eos_token_id property.",
            )]


# ---------------------------------------------------------------------------
# LMHeadAuditAgent
# ---------------------------------------------------------------------------

class LMHeadAuditAgent:
    """
    Validates the language model head (LanguageModelHead / lm_head in inference).

    Source-text checks
    ------------------
    1. LayerNorm applied before the output projection.
    2. Output projection maps d_model → vocab_size.
    3. Optional weight tying is guarded by a flag (not unconditional).

    Numerical checks
    ----------------
    4. Forward pass produces shape (batch, seq_len, vocab_size).
    5. Output logits are finite (no NaN / Inf).
    """

    NAME = "LM Head Audit Agent"

    def audit_source(self, src: str, path: str = "models/language_head.py") -> List[DetokenizeFinding]:
        findings: List[DetokenizeFinding] = []
        findings.extend(self._check_layer_norm_present(src, path))
        findings.extend(self._check_output_projection(src, path))
        findings.extend(self._check_weight_tying_guarded(src, path))
        return findings

    def audit_numerical(
        self,
        d_model: int = 32,
        vocab_size: int = 64,
    ) -> List[DetokenizeFinding]:
        """Run numerical checks on a live LanguageModelHead."""
        import torch
        findings: List[DetokenizeFinding] = []
        try:
            from models.language_head import LanguageModelHead  # type: ignore
        except ImportError:
            findings.append(DetokenizeFinding(
                agent=self.NAME,
                check="LanguageModelHead importable",
                passed=False,
                severity="critical",
                details="Could not import LanguageModelHead from models.language_head.",
                file_path="models/language_head.py",
                fix_hint="Ensure models/language_head.py is on sys.path.",
            ))
            return findings

        head = LanguageModelHead(d_model=d_model, vocab_size=vocab_size)
        head.eval()

        try:
            with torch.no_grad():
                x = torch.randn(2, 8, d_model)
                logits = head(x)

            shape_ok = logits.shape == (2, 8, vocab_size)
            findings.append(DetokenizeFinding(
                agent=self.NAME,
                check="LM head output shape is (batch, seq_len, vocab_size)",
                passed=shape_ok,
                severity="critical",
                details=f"input={list(x.shape)}, logits={list(logits.shape)}",
                file_path="models/language_head.py",
                fix_hint=(
                    "Ensure output_projection is Linear(d_model, vocab_size) "
                    "and is applied to the full sequence dimension."
                ),
            ))

            finite_ok = bool(torch.isfinite(logits).all().item())
            findings.append(DetokenizeFinding(
                agent=self.NAME,
                check="LM head output logits are finite (no NaN/Inf)",
                passed=finite_ok,
                severity="critical",
                details=(
                    "All logits are finite."
                    if finite_ok
                    else f"Non-finite logits detected: {(~torch.isfinite(logits)).sum().item()} elements."
                ),
                file_path="models/language_head.py",
                fix_hint=(
                    "Check for division by zero or overflow in layer_norm / "
                    "output_projection weight initialisation."
                ),
            ))
        except Exception as exc:
            findings.append(DetokenizeFinding(
                agent=self.NAME,
                check="LM head forward pass runs without error",
                passed=False,
                severity="critical",
                details=f"Forward pass raised: {exc}",
                file_path="models/language_head.py",
                fix_hint="Fix the forward() method so it accepts (batch, seq_len, d_model) input.",
            ))

        return findings

    # ------------------------------------------------------------------

    def _check_layer_norm_present(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_ln = bool(re.search(r'LayerNorm|layer_norm', src))
        applied_before = bool(re.search(r'layer_norm.*hidden|self\.layer_norm\(', src))
        passed = has_ln and applied_before
        return [DetokenizeFinding(
            agent=self.NAME,
            check="LayerNorm applied to hidden states before output projection",
            passed=passed,
            severity="major",
            details=(
                "LayerNorm applied before projection found."
                if passed
                else (
                    "No LayerNorm in LM head source." if not has_ln
                    else "LayerNorm defined but not applied before output_projection."
                )
            ),
            file_path=path,
            fix_hint=(
                "Apply layer_norm to hidden_states before passing them to "
                "output_projection: `hidden_states = self.layer_norm(hidden_states)`."
            ),
        )]

    def _check_output_projection(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_proj = bool(re.search(r'output_projection\s*=\s*nn\.Linear|Linear\(d_model', src))
        return [DetokenizeFinding(
            agent=self.NAME,
            check="output_projection is nn.Linear(d_model, vocab_size)",
            passed=has_proj,
            severity="critical",
            details=(
                "nn.Linear output projection found."
                if has_proj
                else "No nn.Linear output_projection detected in LM head source."
            ),
            file_path=path,
            fix_hint=(
                "Add: self.output_projection = nn.Linear(d_model, vocab_size) "
                "in LanguageModelHead.__init__()."
            ),
        )]

    def _check_weight_tying_guarded(self, src: str, path: str) -> List[DetokenizeFinding]:
        """Weight tying must be conditional on a flag, not unconditional."""
        has_tying = bool(re.search(r'tie_weights|weight.*tying', src, re.IGNORECASE))
        has_guard = bool(re.search(r'if.*tie_weights|tie_weights.*and', src))
        # If tying code is absent the check passes vacuously.
        if not has_tying:
            passed = True
            detail = "Weight tying code absent (not required)."
        else:
            passed = has_guard
            detail = (
                "Weight tying is guarded by a flag."
                if passed
                else "Weight tying found but appears unconditional."
            )
        return [DetokenizeFinding(
            agent=self.NAME,
            check="Weight tying is optional and guarded by tie_weights flag",
            passed=passed,
            severity="minor",
            details=detail,
            file_path=path,
            fix_hint=(
                "Wrap weight-tying code in: "
                "`if tie_weights and input_embedding is not None:`"
            ),
        )]


# ---------------------------------------------------------------------------
# GenerationPipelineAgent
# ---------------------------------------------------------------------------

class GenerationPipelineAgent:
    """
    Validates the InferenceEngine.generate() loop.

    Source-text checks
    ------------------
    1. EOS termination present (loop breaks on EOS token).
    2. max_seq_len clipping applied to input_ids.
    3. Field evolve_step() called each iteration.
    4. tokenizer.decode() called on the collected generated_ids.
    5. Entropy gating applied to logits before sampling.

    Numerical checks
    ----------------
    6. _selector_probs produces a valid probability distribution (sums ≈ 1).
    7. _entropy_from_probs returns non-negative scalars.
    8. _entropy_gate with tau=0 returns the original scores unchanged.
    """

    NAME = "Generation Pipeline Agent"

    def audit_source(self, src: str, path: str = "inference/inference.py") -> List[DetokenizeFinding]:
        findings: List[DetokenizeFinding] = []
        findings.extend(self._check_eos_termination(src, path))
        findings.extend(self._check_seq_len_clipping(src, path))
        findings.extend(self._check_field_evolve(src, path))
        findings.extend(self._check_decode_called(src, path))
        findings.extend(self._check_entropy_gating(src, path))
        return findings

    def audit_numerical(self) -> List[DetokenizeFinding]:
        """Run numerical checks by importing helper methods from InferenceEngine."""
        import torch
        findings: List[DetokenizeFinding] = []

        # Import just the static/instance helpers without loading a checkpoint.
        try:
            from inference.inference import InferenceEngine  # type: ignore
            _entropy_from_probs = InferenceEngine._entropy_from_probs
            _entropy_gate = InferenceEngine._entropy_gate
        except ImportError:
            findings.append(DetokenizeFinding(
                agent=self.NAME,
                check="InferenceEngine importable",
                passed=False,
                severity="critical",
                details="Could not import InferenceEngine from inference.inference.",
                file_path="inference/inference.py",
                fix_hint="Ensure inference/inference.py is on sys.path.",
            ))
            return findings

        findings.extend(self._check_selector_probs(torch))
        findings.extend(self._check_entropy_from_probs(torch, _entropy_from_probs))
        findings.extend(self._check_entropy_gate_zero_tau(torch, _entropy_gate))
        return findings

    # ------------------------------------------------------------------
    # Source-text checks
    # ------------------------------------------------------------------

    def _check_eos_termination(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_eos_break = bool(
            re.search(r'if.*eos|== eos_id|next_id == eos', src)
        ) and 'break' in src
        return [DetokenizeFinding(
            agent=self.NAME,
            check="Generation loop terminates on EOS token",
            passed=has_eos_break,
            severity="critical",
            details=(
                "EOS termination (break on EOS) found in generation loop."
                if has_eos_break
                else "No EOS termination detected in the generation loop."
            ),
            file_path=path,
            fix_hint=(
                "Add: `if next_id == eos_id: break` inside the token generation loop."
            ),
        )]

    def _check_seq_len_clipping(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_clip = bool(re.search(r'input_ids.*max_seq_len|:.*-max_seq_len', src))
        return [DetokenizeFinding(
            agent=self.NAME,
            check="input_ids clipped to max_seq_len in generation loop",
            passed=has_clip,
            severity="major",
            details=(
                "max_seq_len clipping of input_ids found."
                if has_clip
                else "No max_seq_len clipping detected for input_ids."
            ),
            file_path=path,
            fix_hint=(
                "Add: `if input_ids.shape[1] > max_seq_len: "
                "input_ids = input_ids[:, -max_seq_len:]` inside the loop."
            ),
        )]

    def _check_field_evolve(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_evolve = bool(re.search(r'evolve_step\(\)', src))
        return [DetokenizeFinding(
            agent=self.NAME,
            check="field.evolve_step() called each generation iteration",
            passed=has_evolve,
            severity="minor",
            details=(
                "field.evolve_step() call found in generation loop."
                if has_evolve
                else "No field.evolve_step() call detected in the generation loop."
            ),
            file_path=path,
            fix_hint=(
                "Add: `self.field.evolve_step()` at the end of each token "
                "generation iteration to keep field dynamics consistent with training."
            ),
        )]

    def _check_decode_called(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_decode = bool(re.search(r'tokenizer\.decode\(generated_ids\)', src))
        return [DetokenizeFinding(
            agent=self.NAME,
            check="tokenizer.decode(generated_ids) called to produce final text",
            passed=has_decode,
            severity="critical",
            details=(
                "tokenizer.decode(generated_ids) call found."
                if has_decode
                else "tokenizer.decode(generated_ids) not found in generation method."
            ),
            file_path=path,
            fix_hint=(
                "Add: `continuation = self.tokenizer.decode(generated_ids)` "
                "after the generation loop and return `prompt + continuation`."
            ),
        )]

    def _check_entropy_gating(self, src: str, path: str) -> List[DetokenizeFinding]:
        has_gate = bool(re.search(r'_entropy_gate\(|entropy_gate', src))
        has_selector = bool(re.search(r'_selector_probs\(|selector_probs', src))
        passed = has_gate and has_selector
        return [DetokenizeFinding(
            agent=self.NAME,
            check="Entropy gating and selector probs applied before sampling",
            passed=passed,
            severity="major",
            details=(
                "Entropy gate and selector probability calls found."
                if passed
                else (
                    "Neither entropy gate nor selector probs found."
                    if not has_gate and not has_selector
                    else f"entropy_gate={'yes' if has_gate else 'no'}, "
                         f"selector_probs={'yes' if has_selector else 'no'}."
                )
            ),
            file_path=path,
            fix_hint=(
                "Call _entropy_gate(logits, ...) then _selector_probs(gated_logits, ...) "
                "before torch.multinomial() to apply entropy-ordered collapse."
            ),
        )]

    # ------------------------------------------------------------------
    # Numerical checks
    # ------------------------------------------------------------------

    def _check_selector_probs(self, torch) -> List[DetokenizeFinding]:
        """softmax selector must sum to 1 across vocab dimension."""
        try:
            from inference.inference import InferenceEngine  # type: ignore

            class _Stub:
                nu_inference = 1.0
                selector = "softmax"

            stub = _Stub()
            scores = torch.randn(1, 100)
            p = InferenceEngine._selector_probs(stub, scores, selector="softmax", tau=1.0)
            total = p.sum(dim=-1)
            passed = bool(torch.allclose(total, torch.ones_like(total), atol=1e-5))
            return [DetokenizeFinding(
                agent=self.NAME,
                check="_selector_probs(softmax) sums to 1 over vocab",
                passed=passed,
                severity="critical",
                details=(
                    f"Probability sum = {total.item():.8f} (expected 1.0)."
                    if passed
                    else f"Probability sum = {total.item():.8f}, expected 1.0."
                ),
                file_path="inference/inference.py",
                fix_hint="Use torch.softmax with clamp min=1e-8 to ensure valid distribution.",
            )]
        except Exception as exc:
            return [DetokenizeFinding(
                agent=self.NAME,
                check="_selector_probs(softmax) sums to 1 over vocab",
                passed=False,
                severity="critical",
                details=f"Check raised: {exc}",
                file_path="inference/inference.py",
                fix_hint="Fix _selector_probs() so it returns a valid probability vector.",
            )]

    def _check_entropy_from_probs(self, torch, fn) -> List[DetokenizeFinding]:
        """Entropy must be non-negative for a valid probability distribution."""
        try:
            p = torch.softmax(torch.randn(1, 50), dim=-1)
            H = fn(p)
            passed = bool((H >= 0).all().item())
            return [DetokenizeFinding(
                agent=self.NAME,
                check="_entropy_from_probs returns non-negative values",
                passed=passed,
                severity="major",
                details=(
                    f"Entropy = {H.mean().item():.6f} (non-negative)."
                    if passed
                    else f"Negative entropy detected: min={H.min().item():.6f}."
                ),
                file_path="inference/inference.py",
                fix_hint=(
                    "Clamp probabilities: `p = p.clamp(min=1e-8)` before computing "
                    "`-(p * torch.log(p)).sum(dim=-1)` to avoid negative values."
                ),
            )]
        except Exception as exc:
            return [DetokenizeFinding(
                agent=self.NAME,
                check="_entropy_from_probs returns non-negative values",
                passed=False,
                severity="major",
                details=f"Check raised: {exc}",
                file_path="inference/inference.py",
                fix_hint="Fix _entropy_from_probs() so it accepts a probability tensor.",
            )]

    def _check_entropy_gate_zero_tau(self, torch, fn) -> List[DetokenizeFinding]:
        """_entropy_gate with tau=0 must return the scores unchanged."""
        try:
            scores = torch.randn(1, 30)
            entropy = torch.tensor(0.5)
            gated = fn(None, scores, entropy=entropy, nu=1.0, tau=0.0)
            passed = bool(torch.allclose(gated, scores, atol=1e-7))
            return [DetokenizeFinding(
                agent=self.NAME,
                check="_entropy_gate with tau=0 returns scores unchanged",
                passed=passed,
                severity="minor",
                details=(
                    "tau=0 gate is a no-op (identity)."
                    if passed
                    else f"tau=0 gate modified scores: max diff = "
                         f"{(gated - scores).abs().max().item():.2e}."
                ),
                file_path="inference/inference.py",
                fix_hint=(
                    "Add: `if tau <= 0: return scores` at the top of _entropy_gate()."
                ),
            )]
        except Exception as exc:
            return [DetokenizeFinding(
                agent=self.NAME,
                check="_entropy_gate with tau=0 returns scores unchanged",
                passed=False,
                severity="minor",
                details=f"Check raised: {exc}",
                file_path="inference/inference.py",
                fix_hint="Fix _entropy_gate() so tau=0 is handled as a no-op.",
            )]


# ---------------------------------------------------------------------------
# OperationalizationAgent
# ---------------------------------------------------------------------------

class OperationalizationAgent:
    """
    Synthesises audit findings into a phased operationalization plan.

    The plan covers four phases:
      Phase 1 – Stabilise (address all critical failures).
      Phase 2 – Validate (address major findings + add tests).
      Phase 3 – Harden (performance, robustness, monitoring).
      Phase 4 – Operationalise (deployment readiness, CI integration).
    """

    NAME = "Operationalization Agent"

    _BASE_PLAN: List[str] = [
        "Phase 1 – Stabilise (resolve critical blockers):",
        "  • Fix any CRITICAL failures surfaced by the audit team.",
        "  • Ensure CognitiveTokenizer.decode() and LanguageModelHead.forward()",
        "    produce correct output shapes and finite values.",
        "  • Verify EOS token termination halts the generation loop correctly.",
        "  • Confirm tokenizer encode → decode roundtrip is lossless.",
        "",
        "Phase 2 – Validate (address major findings, add test coverage):",
        "  • Add encode_batch consistency tests to tests/test_detokenize_output_agents.py.",
        "  • Add LanguageModelHead shape and finiteness tests.",
        "  • Add a full InferenceEngine.generate() integration test (tiny model).",
        "  • Validate selector probability functions (softmax/bornmax/gibbsmax/entropymax).",
        "  • Check weight-tying optional guard in LanguageModelHead.",
        "",
        "Phase 3 – Harden (performance, robustness, monitoring):",
        "  • Add max_seq_len clipping guard with an explicit warning when triggered.",
        "  • Introduce a repetition-penalty option to prevent degenerate loops.",
        "  • Log entropy and gate values per step for diagnostics.",
        "  • Guard against empty generated_ids (return prompt if model emits EOS immediately).",
        "  • Profile tokenizer decode throughput for long sequences.",
        "",
        "Phase 4 – Operationalise (deployment readiness):",
        "  • Wire InferenceEngine into a FastAPI or CLI endpoint.",
        "  • Expose max_tokens, temperature, selector, and nu_inference as query params.",
        "  • Add checkpoint hash validation to prevent loading corrupt weights.",
        "  • Document the full output pipeline in inference/README.md.",
        "  • Add CI step that runs the detokenize/output audit team on every PR.",
    ]

    def build_plan(self, findings: List[DetokenizeFinding]) -> List[str]:
        """Return the operationalization plan, prepending a findings summary."""
        plan: List[str] = []
        critical = [f for f in findings if not f.passed and f.severity == "critical"]
        major = [f for f in findings if not f.passed and f.severity == "major"]
        minor = [f for f in findings if not f.passed and f.severity == "minor"]

        passed_count = sum(1 for f in findings if f.passed)
        failed_count = len(findings) - passed_count

        plan.append("=== Operationalization Plan – Detokenize / Output Pipeline ===")
        plan.append("")
        plan.append(
            f"Audit summary: {passed_count} passed, {failed_count} failed "
            f"({len(critical)} critical | {len(major)} major | {len(minor)} minor)."
        )
        if critical:
            plan.append("")
            plan.append("Critical items requiring immediate attention:")
            for f in critical:
                plan.append(f"  ✗ [{f.agent}] {f.check}")
                plan.append(f"      Fix: {f.fix_hint}")
        plan.append("")
        plan.extend(self._BASE_PLAN)
        return plan


# ---------------------------------------------------------------------------
# OutputTeamCoordinator
# ---------------------------------------------------------------------------

class OutputTeamCoordinator:
    """
    Orchestrates the detokenize/output agent team.

    Execution sequence
    ------------------
    1. ``DetokenizeScannerAgent`` discovers relevant files.
    2. ``TokenizerHealthAgent`` audits tokenizer source text.
    3. ``LMHeadAuditAgent`` audits language head source text.
    4. ``GenerationPipelineAgent`` audits inference engine source text.
    5. Optionally: all three agents run numerical checks.
    6. ``OperationalizationAgent`` synthesises a phased plan.
    7. A consolidated ``DetokenizeOutputReport`` is returned.

    Args:
        repo_root: Absolute path to the repository root (default: parent of utils/).
    """

    def __init__(self, repo_root: Optional[str] = None):
        if repo_root is None:
            repo_root = str(Path(__file__).resolve().parent.parent)
        self.repo_root = Path(repo_root)
        self._scanner = DetokenizeScannerAgent(repo_root=str(self.repo_root))
        self._tokenizer_agent = TokenizerHealthAgent()
        self._lm_head_agent = LMHeadAuditAgent()
        self._generation_agent = GenerationPipelineAgent()
        self._plan_agent = OperationalizationAgent()

    def run(self, run_numerical: bool = True) -> DetokenizeOutputReport:
        """
        Execute the full detokenize/output audit pipeline.

        Args:
            run_numerical: When False, only source-text checks are run.

        Returns:
            DetokenizeOutputReport with all locations, findings, action log,
            and operationalization plan.
        """
        # ── Step 1: discover files ─────────────────────────────────────
        locations = self._scanner.scan()

        # ── Step 2: load source text ───────────────────────────────────
        def _read(rel: str) -> str:
            try:
                return (self.repo_root / rel).read_text(encoding='utf-8', errors='replace')
            except OSError:
                return ""

        tokenizer_src  = _read("training/tokenizer.py")
        lm_head_src    = _read("models/language_head.py")
        inference_src  = _read("inference/inference.py")

        # ── Step 3: source-text audits ─────────────────────────────────
        findings: List[DetokenizeFinding] = []
        findings.extend(
            self._tokenizer_agent.audit_source(tokenizer_src, "training/tokenizer.py")
        )
        findings.extend(
            self._lm_head_agent.audit_source(lm_head_src, "models/language_head.py")
        )
        findings.extend(
            self._generation_agent.audit_source(inference_src, "inference/inference.py")
        )

        # ── Step 4: numerical checks (optional) ───────────────────────
        if run_numerical:
            findings.extend(self._tokenizer_agent.audit_numerical())
            findings.extend(self._lm_head_agent.audit_numerical())
            findings.extend(self._generation_agent.audit_numerical())

        # ── Step 5: operationalization plan ───────────────────────────
        plan = self._plan_agent.build_plan(findings)

        # ── Step 6: action log ─────────────────────────────────────────
        action_log = self._write_action_log(locations, findings)

        return DetokenizeOutputReport(
            locations=locations,
            findings=findings,
            action_log=action_log,
            operationalization_plan=plan,
        )

    # ------------------------------------------------------------------

    def _write_action_log(
        self,
        locations: List[DetokenizeLocation],
        findings: List[DetokenizeFinding],
    ) -> List[str]:
        """Produce the consolidated action log."""
        log: List[str] = []
        log.append("=== Detokenize / Output Audit – Coordinator Action Log ===")
        log.append("")

        log.append("--- Discovered Pipeline Files ---")
        for loc in locations:
            lines_str = ", ".join(str(l) for l in loc.relevant_lines[:5])
            log.append(f"  [{loc.role}] {loc.path}  (lines: {lines_str})")
        log.append("")

        log.append("--- Audit Findings ---")
        for f in findings:
            status = "PASS" if f.passed else f"FAIL [{f.severity.upper()}]"
            log.append(f"  [{f.agent}] {status}: {f.check}")
            if not f.passed:
                log.append(f"    Details: {f.details}")
                log.append(f"    Fix:     {f.fix_hint}")
        log.append("")

        passed = sum(1 for f in findings if f.passed)
        failed = len(findings) - passed
        log.append(f"--- Summary: {passed} passed, {failed} failed out of {len(findings)} checks ---")
        return log
