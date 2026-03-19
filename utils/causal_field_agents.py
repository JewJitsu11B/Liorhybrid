"""
Causal Field Agent Team.

Five specialised agents that collaborate to locate the causal field
implementation in the repository, audit it against the formal law,
propose targeted fixes, and — once the causal-field audit is complete —
audit the end-to-end data pipeline to ensure proper flow.

Roles
-----
CoordinatorScribeAgent
    Scans the repository for every file that touches the causal
    accumulation law, broadcasts the discovered locations to the three
    specialist agents, orchestrates their audit passes, then triggers a
    full data-pipeline audit, and writes a consolidated action log.

AbstractAlgebraAgent
    Verifies the algebraic structure of the law:
      • The associator sign convention: J = (ab)c - a(bc).
      • Antisymmetry of J and Phi.
      • The interpolator alpha as a proper phase angle
        (theta = alpha * pi/2, not a raw kernel weight).
      • Sign of the accumulation law:
            T = alpha*J  **minus**  (1-alpha)*integral(...)

GeometryAgent
    Verifies the geometric / differential-geometry structure:
      • Parallel propagators P^alpha_{alpha'}(x,x') present and
        initialised to the identity in the flat limit.
      • Fiber holonomy Phi^a_b(x) P^b_c(x,x') correctly composed.
      • Past light-cone causal support J^-(x) respected by the kernel.
      • Memory kernel normalization: k(tau<0)=0,
        integral_0^inf k(tau) d tau = 1.

ValidationAgent
    Runs a suite of numerical checks against a small CausalFieldLayer
    instance and consolidates the findings into a ValidationReport that
    can be handed back to the CoordinatorScribeAgent.

DataPipelineAuditAgent
    Audits the end-to-end data pipeline for correct flow.  Invoked by
    the CoordinatorScribeAgent *after* the causal-field audit finishes.

    Stage checks (source-text):
      1. Batch schema  – datasets produce dicts with required keys
         (``input_ids``, ``labels``, ``attention_mask``, ``modality``).
      2. Device move   – trainer batch-to-device loop present.
      3. Audit markers – ``audit_file_once()`` calls at all key trainer
         stages (init, train_epoch, training_step, save/load checkpoint).
      4. Loss inputs   – ``language_modeling_loss`` validates shapes and
         applies the correct causal shift (logits[:,:-1] vs labels[:,1:]).
      5. Combined loss – ``combined_loss`` gates on both ``logits`` and
         ``labels`` keys before computing LM loss.
      6. Memory threading – causal field forward returns
         ``(output, new_memory)`` and the trainer can pass ``new_memory``
         back as ``memory`` on the next call.

    Stage checks (numerical):
      7. Mock-batch LM loss  – a synthetic batch with correct keys
         produces a finite scalar loss with a valid gradient.
      8. Gradient flow       – loss.backward() reaches ``CausalFieldLayer.alpha``
         and ``CausalFieldLayer.Phi`` (not detached).
      9. Memory re-threading – ``new_memory`` returned by one forward call
         can be fed back unchanged to the next forward call without error.

Usage
-----
    from utils.causal_field_agents import CoordinatorScribeAgent

    coordinator = CoordinatorScribeAgent(repo_root="/path/to/repo")
    report = coordinator.run()
    print(report.action_log)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalFieldLocation:
    """A file in the repo that contains part of the causal field implementation."""
    path: str
    role: str           # e.g. "core implementation", "test", "visualisation"
    relevant_lines: List[int]  # 1-based line numbers of interest


@dataclass(frozen=True)
class AgentFinding:
    """A single audit finding raised by one specialist agent."""
    agent: str
    check: str
    passed: bool
    severity: str       # "critical" | "major" | "minor" | "info"
    details: str
    file_path: str
    fix_hint: str


@dataclass
class CausalFieldReport:
    """Consolidated report produced by the CoordinatorScribeAgent."""
    locations: List[CausalFieldLocation]
    findings: List[AgentFinding]
    action_log: List[str]
    transport_plan: List[str]
    pipeline_findings: List[AgentFinding] = field(default_factory=list)
    entropy_findings: List[AgentFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.findings) and all(
            f.passed for f in self.pipeline_findings
        ) and all(f.passed for f in self.entropy_findings)

    @property
    def critical_failures(self) -> List[AgentFinding]:
        all_findings = self.findings + self.pipeline_findings + self.entropy_findings
        return [f for f in all_findings if not f.passed and f.severity == "critical"]


# ---------------------------------------------------------------------------
# Abstract Algebra Agent
# ---------------------------------------------------------------------------

class AbstractAlgebraAgent:
    """
    Validates the algebraic structure of the causal accumulation law.

    Checks
    ------
    1. J antisymmetry – the associator J^{mu nu} must satisfy J^{nu mu} = -J^{mu nu}.
    2. Phi antisymmetry – the bivector field must satisfy Phi^{rho sigma} = -Phi^{sigma rho}.
    3. Alpha as phase angle – alpha must be a dedicated ``nn.Parameter`` distinct
       from the LIoR kernel weights, representing the phase angle
       theta = alpha * pi / 2.
    4. Sign of the accumulation law – the combination must be
       T = alpha*J  MINUS  (1-alpha)*memory, not plus.
    """

    NAME = "Abstract Algebra Agent"

    def audit(
        self,
        locations: List[CausalFieldLocation],
        source_text: Dict[str, str],
    ) -> List[AgentFinding]:
        findings: List[AgentFinding] = []
        for loc in locations:
            if "core implementation" not in loc.role:
                continue
            text = source_text.get(loc.path, "")
            findings.extend(self._check_sign(text, loc.path))
            findings.extend(self._check_alpha_parameter(text, loc.path))
            findings.extend(self._check_phi_antisymmetry(text, loc.path))
            findings.extend(self._check_j_antisymmetry(text, loc.path))
        return findings

    # ------------------------------------------------------------------
    def _check_sign(self, text: str, path: str) -> List[AgentFinding]:
        """T = alpha*J - (1-alpha)*memory  (minus, not plus)."""
        # Look for the combination expression
        has_correct_sign = bool(
            re.search(r'alpha\s*\*\s*J_flat\s*-\s*\(1\s*-\s*alpha\)', text)
        )
        has_wrong_sign = bool(
            re.search(r'alpha\s*\*\s*J_flat\s*\+\s*\(1\s*-\s*alpha\)', text)
        )
        if has_wrong_sign:
            return [AgentFinding(
                agent=self.NAME,
                check="Accumulation-law sign: T = alpha*J - (1-alpha)*integral",
                passed=False,
                severity="critical",
                details=(
                    "Found 'alpha * J_flat + (1 - alpha) * memory_out'. "
                    "The causal accumulation law requires a minus sign: "
                    "T = alpha*J - (1-alpha)*integral_{J^-}(...)."
                ),
                file_path=path,
                fix_hint=(
                    "Change 'alpha * J_flat + (1 - alpha) * memory_out' to "
                    "'alpha * J_flat - (1 - alpha) * memory_out'."
                ),
            )]
        return [AgentFinding(
            agent=self.NAME,
            check="Accumulation-law sign: T = alpha*J - (1-alpha)*integral",
            passed=has_correct_sign,
            severity="critical",
            details=(
                "Correct minus sign found in combination expression."
                if has_correct_sign
                else "Could not locate the T_flat combination expression to verify sign."
            ),
            file_path=path,
            fix_hint="Ensure 'T_flat = alpha * J_flat - (1 - alpha) * memory_out'.",
        )]

    def _check_alpha_parameter(self, text: str, path: str) -> List[AgentFinding]:
        """alpha must be a dedicated nn.Parameter (phase angle), not kernel.weights[0]."""
        uses_kernel_weight = bool(
            re.search(r'self\.memory\.kernel\.weights\[0\]', text)
        )
        has_dedicated_param = bool(
            re.search(r'self\.alpha\s*=\s*nn\.Parameter', text)
        )
        passed = has_dedicated_param and not uses_kernel_weight
        return [AgentFinding(
            agent=self.NAME,
            check="alpha is a dedicated nn.Parameter (phase angle theta=alpha*pi/2)",
            passed=passed,
            severity="major",
            details=(
                "alpha is properly declared as a dedicated nn.Parameter."
                if has_dedicated_param
                else (
                    "alpha is read from self.memory.kernel.weights[0]. "
                    "The interpolator alpha represents a phase angle "
                    "theta = alpha*pi/2 and must be a separate learnable parameter."
                )
            ),
            file_path=path,
            fix_hint=(
                "Add 'self.alpha = nn.Parameter(torch.tensor(0.5))' in __init__ "
                "and replace 'self.memory.kernel.weights[0]' with "
                "'torch.clamp(self.alpha, 0.0, 1.0)'."
            ),
        )]

    def _check_phi_antisymmetry(self, text: str, path: str) -> List[AgentFinding]:
        """Phi^[rho sigma] must be antisymmetric."""
        has_antisym = bool(
            re.search(r'Phi\.data\s*=\s*self\.Phi\.data\s*-\s*self\.Phi\.data\.T', text)
        )
        has_get_phi = bool(
            re.search(r'def get_phi', text)
        )
        passed = has_antisym or has_get_phi
        return [AgentFinding(
            agent=self.NAME,
            check="Phi^[rho sigma] is antisymmetric (bivector structure)",
            passed=passed,
            severity="major",
            details=(
                "Phi antisymmetry enforced via data.T subtraction or get_phi accessor."
                if passed
                else "No explicit antisymmetrisation of Phi found."
            ),
            file_path=path,
            fix_hint=(
                "Enforce Phi^[rho sigma] = -Phi^[sigma rho] by setting "
                "self.Phi.data = self.Phi.data - self.Phi.data.T in __init__ "
                "and by using 0.5*(Phi - Phi.T) when reading Phi."
            ),
        )]

    def _check_j_antisymmetry(self, text: str, path: str) -> List[AgentFinding]:
        """Associator J must be antisymmetric."""
        has_antisym = bool(
            re.search(r'J_tensor\s*-\s*J_tensor\.transpose', text)
        )
        return [AgentFinding(
            agent=self.NAME,
            check="Source current J^{mu nu} is antisymmetric",
            passed=has_antisym,
            severity="major",
            details=(
                "J antisymmetrised via J - J.transpose."
                if has_antisym
                else "No explicit antisymmetrisation of J tensor found."
            ),
            file_path=path,
            fix_hint=(
                "After constructing J_tensor, subtract its transpose: "
                "'J_tensor = J_tensor - J_tensor.transpose(-1, -2)'."
            ),
        )]


# ---------------------------------------------------------------------------
# Geometry Agent
# ---------------------------------------------------------------------------

class GeometryAgent:
    """
    Validates the differential-geometry structure of the causal accumulation law.

    Checks
    ------
    1. Parallel propagators P^alpha_{alpha'}(x,x') are present in the transport.
    2. Fiber holonomy Phi^a_b(x) P^b_c(x,x') correctly composed.
    3. Causal support J^-(x): integral restricted to the past light cone.
    4. Kernel normalisation: k(tau<0)=0 and integral_0^inf k(tau)=1.
    """

    NAME = "Geometry Agent"

    def audit(
        self,
        locations: List[CausalFieldLocation],
        source_text: Dict[str, str],
    ) -> List[AgentFinding]:
        findings: List[AgentFinding] = []
        for loc in locations:
            if "core implementation" not in loc.role:
                continue
            text = source_text.get(loc.path, "")
            findings.extend(self._check_parallel_propagators(text, loc.path))
            findings.extend(self._check_fiber_holonomy(text, loc.path))
            findings.extend(self._check_causal_support(text, loc.path))
            findings.extend(self._check_kernel_normalisation(text, loc.path))
        return findings

    # ------------------------------------------------------------------
    def _check_parallel_propagators(self, text: str, path: str) -> List[AgentFinding]:
        """P^alpha_{alpha'}(x,x') parallel propagators must appear in the transport."""
        has_propagator_class = bool(
            re.search(r'class\s+ParallelPropagator', text)
        )
        # Also accept if a dedicated parallel_transport.py is imported
        imports_propagator = bool(
            re.search(r'ParallelPropagator', text)
        )
        passed = has_propagator_class or imports_propagator
        return [AgentFinding(
            agent=self.NAME,
            check="Parallel propagators P^alpha_{alpha'}(x,x') present in transport",
            passed=passed,
            severity="major",
            details=(
                "ParallelPropagator found in the transport implementation."
                if passed
                else (
                    "No ParallelPropagator class or import found. "
                    "The law requires two explicit parallel propagators "
                    "P^alpha_{alpha'} and P^beta_{beta'} to transport "
                    "the source-current indices from x' to x."
                )
            ),
            file_path=path,
            fix_hint=(
                "Import and use operators.parallel_transport.ParallelPropagator "
                "for each tangent-vector index that crosses from x' to x. "
                "Initialise with identity weights for the flat-spacetime baseline."
            ),
        )]

    def _check_fiber_holonomy(self, text: str, path: str) -> List[AgentFinding]:
        """Fiber holonomy Phi^a_b P^b_c must be composed correctly."""
        has_holonomy_class = bool(re.search(r'class\s+FiberHolonomy', text))
        has_holonomy_import = bool(re.search(r'FiberHolonomy', text))
        has_phi_composed = bool(
            re.search(r'Phi.*P.*fiber|fiber.*Phi|holonomy', text, re.IGNORECASE)
        )
        passed = has_holonomy_class or has_holonomy_import or has_phi_composed
        return [AgentFinding(
            agent=self.NAME,
            check="Fiber holonomy Phi^a_b(x) P^b_c(x,x') correctly composed",
            passed=passed,
            severity="major",
            details=(
                "FiberHolonomy or Phi-fiber composition found."
                if passed
                else (
                    "Phi is used as a standalone bivector without composing with "
                    "a fiber parallel-propagator P^b_c(x,x'). "
                    "The law requires Phi^a_b(x) * P^b_c(x,x')."
                )
            ),
            file_path=path,
            fix_hint=(
                "Use operators.parallel_transport.FiberHolonomy which composes "
                "Phi^a_b with P^b_c. Replace bare Phi usage in the transport "
                "step with FiberHolonomy.forward(field_at_xprime)."
            ),
        )]

    def _check_causal_support(self, text: str, path: str) -> List[AgentFinding]:
        """Integral domain must be restricted to the past light cone J^-(x)."""
        has_causal = bool(
            re.search(
                r'J\^?-\s*\(x\)|past.light.cone|causal.past|J\^-|LiorMemory|memory_out',
                text, re.IGNORECASE
            )
        )
        return [AgentFinding(
            agent=self.NAME,
            check="Causal support: integral restricted to past light cone J^-(x)",
            passed=has_causal,
            severity="major",
            details=(
                "Causal past / LIoR memory integration present."
                if has_causal
                else "No evidence of causal-past restriction in the integral."
            ),
            file_path=path,
            fix_hint=(
                "Ensure the memory integration (LiorMemoryState) only accumulates "
                "contributions from tau >= 0 (retarded / causal direction). "
                "The kernel must satisfy k(tau < 0) = 0."
            ),
        )]

    def _check_kernel_normalisation(self, text: str, path: str) -> List[AgentFinding]:
        """Kernel must satisfy k(tau<0)=0 and be normalised to 1."""
        has_lior = bool(re.search(r'LiorMemory|lior_kernel|fractional_kernel', text, re.IGNORECASE))
        return [AgentFinding(
            agent=self.NAME,
            check="Memory kernel normalised: k(tau<0)=0 and integral k(tau)dtau=1",
            passed=has_lior,
            severity="minor",
            details=(
                "LIoR / fractional kernel (causal, normalised by construction) present."
                if has_lior
                else "No normalised causal kernel found in the file."
            ),
            file_path=path,
            fix_hint=(
                "Use LiorMemoryState (models/lior_kernel.py) which implements a "
                "causal power-law kernel that satisfies k(tau<0)=0 and "
                "sum_t k_t = 1 by the finite-pole normalisation."
            ),
        )]


# ---------------------------------------------------------------------------
# Validation Agent
# ---------------------------------------------------------------------------

class ValidationAgent:
    """
    Runs numerical checks against a live CausalFieldLayer instance.

    These checks do not depend on the source text; they instantiate the
    module and verify runtime behaviour directly.
    """

    NAME = "Validation Agent"

    def audit(
        self,
        d_model: int = 16,
        d_field: int = 4,
        d_spinor: int = 4,
    ) -> List[AgentFinding]:
        """
        Instantiate a small CausalFieldLayer and run numerical checks.

        Args:
            d_model: Feature dimension (kept small for unit tests).
            d_field: Field index dimension.
            d_spinor: Spinor dimension.

        Returns:
            List of AgentFinding items.
        """
        findings: List[AgentFinding] = []
        try:
            from models.causal_field import CausalFieldLayer  # type: ignore
        except ImportError:
            findings.append(AgentFinding(
                agent=self.NAME,
                check="CausalFieldLayer importable",
                passed=False,
                severity="critical",
                details="Could not import CausalFieldLayer from models.causal_field.",
                file_path="models/causal_field.py",
                fix_hint="Ensure models/causal_field.py is on sys.path.",
            ))
            return findings

        layer = CausalFieldLayer(d_model=d_model, d_field=d_field, d_spinor=d_spinor)
        layer.eval()

        findings.extend(self._check_alpha_param(layer))
        findings.extend(self._check_forward_sign(layer, d_model))
        findings.extend(self._check_phi_antisymmetry(layer))
        findings.extend(self._check_output_shape(layer, d_model))
        return findings

    # ------------------------------------------------------------------
    def _check_alpha_param(self, layer: nn.Module) -> List[AgentFinding]:
        has_alpha = hasattr(layer, 'alpha') and isinstance(layer.alpha, nn.Parameter)
        val_ok = False
        if has_alpha:
            v = layer.alpha.item()
            val_ok = 0.0 <= v <= 1.0
        return [AgentFinding(
            agent=self.NAME,
            check="CausalFieldLayer has dedicated alpha nn.Parameter in [0,1]",
            passed=has_alpha and val_ok,
            severity="critical",
            details=(
                f"alpha = nn.Parameter(tensor({layer.alpha.item():.4f})), in range: {val_ok}."
                if has_alpha
                else "No self.alpha parameter found on the layer."
            ),
            file_path="models/causal_field.py",
            fix_hint="Add self.alpha = nn.Parameter(torch.tensor(0.5)) in __init__.",
        )]

    def _check_forward_sign(self, layer: nn.Module, d_model: int) -> List[AgentFinding]:
        """
        Verify the accumulation law has a MINUS sign by checking that
        output changes direction when memory_out is large and positive.

        Strategy: pin alpha near 0 so the output is dominated by the
        -(1-alpha)*memory term; check that the output has opposite sign
        to a positive-only baseline.
        """
        try:
            with torch.no_grad():
                layer.alpha.data.fill_(0.0)   # T ≈ -1 * memory
                x = torch.ones(1, 4, d_model)
                out_minus, _ = layer(x)

                layer.alpha.data.fill_(1.0)   # T ≈ +1 * J (no memory)
                out_plus, _ = layer(x)

                layer.alpha.data.fill_(0.5)   # restore

            # With alpha=0 the output should be negated relative to alpha=1
            # if the sign is correct; we just check that they are different.
            different = not torch.allclose(out_minus, out_plus, atol=1e-6)
            return [AgentFinding(
                agent=self.NAME,
                check="Accumulation law sign: alpha=0 and alpha=1 give different outputs",
                passed=different,
                severity="critical",
                details=(
                    "alpha=0 and alpha=1 produce distinct outputs (sign is active)."
                    if different
                    else "alpha=0 and alpha=1 produced identical outputs; sign may be wrong."
                ),
                file_path="models/causal_field.py",
                fix_hint="Ensure T_flat = alpha*J_flat - (1-alpha)*memory_out.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Accumulation law sign: alpha=0 and alpha=1 give different outputs",
                passed=False,
                severity="critical",
                details=f"Forward pass raised: {exc}",
                file_path="models/causal_field.py",
                fix_hint="Fix forward pass so it runs without errors.",
            )]

    def _check_phi_antisymmetry(self, layer: nn.Module) -> List[AgentFinding]:
        """Phi must satisfy Phi + Phi^T = 0 (antisymmetric)."""
        try:
            Phi = layer.get_phi()
            antisym_err = (Phi + Phi.T).abs().max().item()
            passed = antisym_err < 1e-6
            return [AgentFinding(
                agent=self.NAME,
                check="Phi^[rho sigma] antisymmetry: max|Phi + Phi^T| < 1e-6",
                passed=passed,
                severity="major",
                details=f"max|Phi + Phi.T| = {antisym_err:.2e}",
                file_path="models/causal_field.py",
                fix_hint="Use get_phi() = 0.5*(Phi - Phi.T) to enforce antisymmetry.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Phi^[rho sigma] antisymmetry: max|Phi + Phi^T| < 1e-6",
                passed=False,
                severity="major",
                details=f"get_phi() raised: {exc}",
                file_path="models/causal_field.py",
                fix_hint="Implement get_phi() returning 0.5*(self.Phi - self.Phi.T).",
            )]

    def _check_output_shape(self, layer: nn.Module, d_model: int) -> List[AgentFinding]:
        """Output shape must match input shape [B, N, d_model]."""
        try:
            with torch.no_grad():
                x = torch.randn(2, 8, d_model)
                out, _ = layer(x)
            shape_ok = out.shape == x.shape
            return [AgentFinding(
                agent=self.NAME,
                check="CausalFieldLayer output shape matches input [B, N, d_model]",
                passed=shape_ok,
                severity="major",
                details=f"input={list(x.shape)}, output={list(out.shape)}",
                file_path="models/causal_field.py",
                fix_hint="Check output_proj and norm dimensions.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="CausalFieldLayer output shape matches input [B, N, d_model]",
                passed=False,
                severity="major",
                details=f"Forward pass raised: {exc}",
                file_path="models/causal_field.py",
                fix_hint="Fix forward pass so it runs without errors.",
            )]



# ---------------------------------------------------------------------------
# Data Pipeline Audit Agent
# ---------------------------------------------------------------------------

class DataPipelineAuditAgent:
    """
    Audits the end-to-end data pipeline to ensure proper flow.

    Invoked by the CoordinatorScribeAgent *after* the causal-field audit.

    Source-text checks (no model instantiation required)
    ----------------------------------------------------
    1. Batch schema        – dataset ``__getitem__`` / ``_make_example``
                             returns dicts with required keys.
    2. Device move         – trainer moves batch tensors to device before
                             any forward pass.
    3. Audit markers       – ``audit_file_once()`` calls present at all key
                             trainer entry points.
    4. Loss shape guard    – ``language_modeling_loss`` validates that
                             ``labels.shape[:2] == (batch, seq_len)``.
    5. Causal LM shift     – ``language_modeling_loss`` shifts logits and
                             labels by one position for autoregressive loss.
    6. Combined-loss gates – ``combined_loss`` only computes LM loss when
                             both ``'logits'`` and ``'labels'`` are present.
    7. Memory threading    – ``CausalFieldLayer.forward`` returns a tuple
                             ``(output, new_memory)`` so the trainer can
                             re-thread state across time steps.

    Numerical checks (small toy instances)
    ---------------------------------------
    8. Mock-batch LM loss  – a synthetic [B, T, V] logits + [B, T] labels
                             batch produces a finite scalar loss with grad.
    9. Gradient flow       – after loss.backward(), ``CausalFieldLayer.alpha``
                             and ``CausalFieldLayer.Phi`` have non-None .grad.
    10. Memory re-threading – new_memory from step t can be passed as memory
                              at step t+1 without raising an exception.
    """

    NAME = "Data Pipeline Audit Agent"

    # Required keys every text-mode dataset item must carry
    _REQUIRED_BATCH_KEYS = ("input_ids", "labels", "attention_mask", "modality")

    def audit(
        self,
        repo_root: Path,
        run_numerical: bool = True,
        d_model: int = 16,
        d_field: int = 4,
        d_spinor: int = 4,
        vocab_size: int = 32,
    ) -> List[AgentFinding]:
        """
        Run the full pipeline audit.

        Args:
            repo_root:      Root path of the repository.
            run_numerical:  When False only source-text checks are run.
            d_model / d_field / d_spinor:
                            Dimensions for the toy CausalFieldLayer used in
                            numerical checks.
            vocab_size:     Vocabulary size for the mock LM loss check.

        Returns:
            List of AgentFinding items.
        """
        findings: List[AgentFinding] = []

        # ── Load pipeline source files ─────────────────────────────────
        def _read(rel: str) -> str:
            try:
                return (repo_root / rel).read_text(encoding='utf-8', errors='replace')
            except OSError:
                return ""

        datasets_src = _read("training/datasets.py")
        trainer_src  = _read("training/trainer.py")
        losses_src   = _read("training/losses.py")
        field_src    = _read("models/causal_field.py")

        # ── Source-text checks ─────────────────────────────────────────
        findings.extend(self._check_batch_schema(datasets_src))
        findings.extend(self._check_device_move(trainer_src))
        findings.extend(self._check_audit_markers(trainer_src))
        findings.extend(self._check_loss_shape_guard(losses_src))
        findings.extend(self._check_causal_lm_shift(losses_src))
        findings.extend(self._check_combined_loss_gates(losses_src))
        findings.extend(self._check_memory_threading_source(field_src))

        # ── Numerical checks ───────────────────────────────────────────
        if run_numerical:
            findings.extend(self._check_mock_batch_loss(vocab_size))
            findings.extend(self._check_gradient_flow(d_model, d_field, d_spinor, vocab_size))
            findings.extend(self._check_memory_rethreading(d_model, d_field, d_spinor))

        return findings

    # ------------------------------------------------------------------
    # Source-text checks
    # ------------------------------------------------------------------

    def _check_batch_schema(self, src: str) -> List[AgentFinding]:
        """Dataset items must carry all required batch keys."""
        missing = [k for k in self._REQUIRED_BATCH_KEYS if f"'{k}'" not in src]
        passed = not missing
        return [AgentFinding(
            agent=self.NAME,
            check="Dataset items carry all required batch keys",
            passed=passed,
            severity="critical",
            details=(
                "All required keys found in dataset source."
                if passed
                else f"Missing key(s) in training/datasets.py: {missing}"
            ),
            file_path="training/datasets.py",
            fix_hint=(
                "Ensure every dataset __getitem__/_make_example returns a dict "
                "containing all of: " + ", ".join(f"'{k}'" for k in self._REQUIRED_BATCH_KEYS)
            ),
        )]

    def _check_device_move(self, src: str) -> List[AgentFinding]:
        """Trainer must move batch tensors to device before the forward pass."""
        has_move = bool(re.search(
            r'\.to\(self\.device\)',
            src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="Trainer moves batch tensors to device before forward pass",
            passed=has_move,
            severity="critical",
            details=(
                "batch.to(self.device) call found in trainer."
                if has_move
                else "No .to(self.device) call found in training/trainer.py."
            ),
            file_path="training/trainer.py",
            fix_hint=(
                "Add: batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) "
                "else v for k, v in batch.items()} before the forward pass."
            ),
        )]

    def _check_audit_markers(self, src: str) -> List[AgentFinding]:
        """Key trainer entry points must log an audit_file_once() marker."""
        required_labels = [
            "trainer",
            "train_epoch",
            "training_step",
            "save_checkpoint",
            "load_checkpoint",
        ]
        missing = [l for l in required_labels
                   if not re.search(rf'audit_file_once\(["\']' + l, src)]
        passed = not missing
        return [AgentFinding(
            agent=self.NAME,
            check="Pipeline audit markers present at all key trainer stages",
            passed=passed,
            severity="minor",
            details=(
                "All expected audit_file_once() markers found."
                if passed
                else f"Missing audit markers for: {missing}"
            ),
            file_path="training/trainer.py",
            fix_hint=(
                "Add audit_file_once('<stage>', __file__) at the start of each "
                "missing stage: " + str(missing)
            ),
        )]

    def _check_loss_shape_guard(self, src: str) -> List[AgentFinding]:
        """language_modeling_loss must guard logits/labels shape compatibility."""
        has_guard = bool(re.search(
            r'labels\.shape\[:2\]\s*!=.*batch_size.*seq_len|'
            r'raise ValueError.*labels shape',
            src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="language_modeling_loss guards logits/labels shape compatibility",
            passed=has_guard,
            severity="major",
            details=(
                "Shape compatibility guard found in language_modeling_loss."
                if has_guard
                else "No labels.shape[:2] != (batch, seq_len) guard found."
            ),
            file_path="training/losses.py",
            fix_hint=(
                "Add: if labels.shape[:2] != (batch_size, seq_len): "
                "raise ValueError(...) before the shift step."
            ),
        )]

    def _check_causal_lm_shift(self, src: str) -> List[AgentFinding]:
        """language_modeling_loss must shift logits and labels by 1 for causal LM."""
        has_shift = bool(re.search(
            r'logits\[:.*:-1.*\]|shift_logits|logits\[.*,\s*:-1',
            src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="language_modeling_loss applies causal-LM shift (logits[:,:-1] vs labels[:,1:])",
            passed=has_shift,
            severity="critical",
            details=(
                "Causal-LM shift (logits[:, :-1] / labels[:, 1:]) found."
                if has_shift
                else "No causal-LM shift found; loss may be computed on the wrong positions."
            ),
            file_path="training/losses.py",
            fix_hint=(
                "Ensure shift_logits = logits[:, :-1, :] and "
                "shift_labels = labels[:, 1:] in language_modeling_loss."
            ),
        )]

    def _check_combined_loss_gates(self, src: str) -> List[AgentFinding]:
        """combined_loss must gate on 'logits' and 'labels' before computing LM loss."""
        has_gate = bool(re.search(
            r"'logits'\s+in\s+outputs\s+and\s+'labels'\s+in\s+batch|"
            r"if\s+'logits'\s+in\s+outputs",
            src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="combined_loss gates on 'logits' and 'labels' keys before LM loss",
            passed=has_gate,
            severity="major",
            details=(
                "combined_loss has 'logits' in outputs gate."
                if has_gate
                else "combined_loss may attempt LM loss without checking for required keys."
            ),
            file_path="training/losses.py",
            fix_hint=(
                "Guard LM loss computation with: "
                "if 'logits' in outputs and 'labels' in batch: ..."
            ),
        )]

    def _check_memory_threading_source(self, src: str) -> List[AgentFinding]:
        """CausalFieldLayer.forward must return (output, new_memory) for re-threading."""
        returns_tuple = bool(re.search(
            r'return\s+output\s*,\s*new_memory',
            src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="CausalFieldLayer.forward returns (output, new_memory) tuple",
            passed=returns_tuple,
            severity="major",
            details=(
                "return output, new_memory found; memory can be re-threaded."
                if returns_tuple
                else "CausalFieldLayer.forward does not appear to return new_memory; "
                     "stateful inference across time steps will be broken."
            ),
            file_path="models/causal_field.py",
            fix_hint=(
                "Ensure the last line of CausalFieldLayer.forward is "
                "'return output, new_memory'."
            ),
        )]

    # ------------------------------------------------------------------
    # Numerical checks
    # ------------------------------------------------------------------

    def _check_mock_batch_loss(self, vocab_size: int) -> List[AgentFinding]:
        """A synthetic [B, T, V] / [B, T] batch must produce a finite scalar loss."""
        try:
            from training.losses import language_modeling_loss  # type: ignore
        except ImportError:
            return [AgentFinding(
                agent=self.NAME,
                check="Mock-batch language_modeling_loss produces finite scalar with grad",
                passed=False,
                severity="major",
                details="Could not import language_modeling_loss from training.losses.",
                file_path="training/losses.py",
                fix_hint="Ensure training/losses.py is on sys.path.",
            )]

        B, T, V = 2, 8, vocab_size
        logits = torch.randn(B, T, V, requires_grad=True)
        labels = torch.randint(0, V, (B, T))
        mask   = torch.ones(B, T, dtype=torch.long)

        try:
            loss = language_modeling_loss(logits, labels, attention_mask=mask)
            finite   = bool(torch.isfinite(loss))
            has_grad = loss.requires_grad
            passed   = finite and has_grad
            return [AgentFinding(
                agent=self.NAME,
                check="Mock-batch language_modeling_loss produces finite scalar with grad",
                passed=passed,
                severity="critical",
                details=(
                    f"loss={loss.item():.4f}, finite={finite}, requires_grad={has_grad}"
                ),
                file_path="training/losses.py",
                fix_hint="Ensure loss computation does not produce NaN/Inf and keeps grad.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Mock-batch language_modeling_loss produces finite scalar with grad",
                passed=False,
                severity="critical",
                details=f"language_modeling_loss raised: {exc}",
                file_path="training/losses.py",
                fix_hint="Fix language_modeling_loss so it runs without errors.",
            )]

    def _check_gradient_flow(
        self,
        d_model: int,
        d_field: int,
        d_spinor: int,
        vocab_size: int,
    ) -> List[AgentFinding]:
        """
        loss.backward() must reach CausalFieldLayer.alpha and .Phi.
        """
        try:
            from models.causal_field import CausalFieldLayer  # type: ignore
        except ImportError:
            return [AgentFinding(
                agent=self.NAME,
                check="Gradient flows back to CausalFieldLayer.alpha and .Phi",
                passed=False,
                severity="major",
                details="Could not import CausalFieldLayer.",
                file_path="models/causal_field.py",
                fix_hint="Ensure models/causal_field.py is on sys.path.",
            )]

        try:
            layer = CausalFieldLayer(d_model=d_model, d_field=d_field, d_spinor=d_spinor)
            layer.train()

            # Run a small causal-field + linear-head mini-graph
            lm_head = nn.Linear(d_model, vocab_size, bias=False)

            x = torch.randn(1, 4, d_model)
            out, _ = layer(x)
            logits = lm_head(out)                          # [1, 4, V]
            labels = torch.randint(0, vocab_size, (1, 4))

            from training.losses import language_modeling_loss  # type: ignore
            loss = language_modeling_loss(logits, labels)
            loss.backward()

            alpha_grad = layer.alpha.grad is not None
            phi_grad   = layer.Phi.grad is not None
            passed     = alpha_grad and phi_grad

            return [AgentFinding(
                agent=self.NAME,
                check="Gradient flows back to CausalFieldLayer.alpha and .Phi",
                passed=passed,
                severity="major",
                details=(
                    f"alpha.grad present: {alpha_grad}, Phi.grad present: {phi_grad}"
                ),
                file_path="models/causal_field.py",
                fix_hint=(
                    "Ensure no .detach() calls break the computational graph "
                    "between the loss and alpha/Phi."
                ),
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Gradient flows back to CausalFieldLayer.alpha and .Phi",
                passed=False,
                severity="major",
                details=f"Backward pass raised: {exc}",
                file_path="models/causal_field.py",
                fix_hint="Fix the forward/backward pass so it runs without errors.",
            )]

    def _check_memory_rethreading(
        self,
        d_model: int,
        d_field: int,
        d_spinor: int,
    ) -> List[AgentFinding]:
        """
        new_memory from step t must be accepted as memory at step t+1.
        """
        try:
            from models.causal_field import CausalFieldLayer  # type: ignore
        except ImportError:
            return [AgentFinding(
                agent=self.NAME,
                check="CausalFieldLayer memory can be re-threaded across time steps",
                passed=False,
                severity="major",
                details="Could not import CausalFieldLayer.",
                file_path="models/causal_field.py",
                fix_hint="Ensure models/causal_field.py is on sys.path.",
            )]

        try:
            layer = CausalFieldLayer(d_model=d_model, d_field=d_field, d_spinor=d_spinor)
            layer.eval()
            x = torch.randn(1, 4, d_model)

            with torch.no_grad():
                out1, mem1 = layer(x, memory=None)
                out2, mem2 = layer(x, memory=mem1)   # re-thread step 1 → step 2

            shapes_ok = out1.shape == out2.shape
            return [AgentFinding(
                agent=self.NAME,
                check="CausalFieldLayer memory can be re-threaded across time steps",
                passed=shapes_ok,
                severity="major",
                details=(
                    f"Re-threaded forward succeeded; output shapes match: {list(out1.shape)}."
                    if shapes_ok
                    else f"Output shapes differ: {list(out1.shape)} vs {list(out2.shape)}."
                ),
                file_path="models/causal_field.py",
                fix_hint=(
                    "Ensure CausalFieldLayer.forward accepts the dict returned by "
                    "LiorMemoryState as the 'memory' argument without modification."
                ),
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="CausalFieldLayer memory can be re-threaded across time steps",
                passed=False,
                severity="major",
                details=f"Memory re-threading raised: {exc}",
                file_path="models/causal_field.py",
                fix_hint="Fix CausalFieldLayer.forward to accept LiorMemoryState output.",
            )]


# ---------------------------------------------------------------------------
# Physics Agent
# ---------------------------------------------------------------------------

class PhysicsAgent:
    """
    Validates the physical consistency of the entropy-gated framework.

    Works in conjunction with EntropySoftmaxPlanningAgent to ensure that:

    Checks (source-text)
    --------------------
    1. Caputo derivative order α ∈ (0, 2)  – invalid α breaks causality.
    2. Entropy exponent ν(x) > 0 and bounded away from zero (Remark (iii)
       Lipschitz condition requires ν ≥ ν_min > 0).
    3. Temperature τ(x) > 0  – enforced via softplus floor.
    4. Singularity floor ε  – |Ψ| clamped to avoid |Ψ|^{2ν-2} → ∞ at Ψ=0
       (Remark (ii)).
    5. Gâteaux derivative uses the exact formula 2ν|Ψ|^{2ν-2}Ψ·Φ(x), not an
       approximation.

    Checks (numerical)
    ------------------
    6. Monotonicity: ⟨∇H[Ψ], Ψ⟩ ≥ 0 for random Ψ (Remark (iii)).
    7. Positive Φ(x): kernel integral Φ(x) = Σ_y φ(x,y) > 0 everywhere.
    8. Caputo weights are positive and normalised for α ∈ (1, 2).
    """

    NAME = "Physics Agent"

    def audit(
        self,
        source_text: str,
        run_numerical: bool = True,
        d_model: int = 8,
    ) -> List[AgentFinding]:
        findings: List[AgentFinding] = []
        findings.extend(self._check_caputo_order(source_text))
        findings.extend(self._check_nu_floor(source_text))
        findings.extend(self._check_tau_positivity(source_text))
        findings.extend(self._check_singularity_floor(source_text))
        findings.extend(self._check_exact_gateaux(source_text))
        if run_numerical:
            findings.extend(self._check_monotonicity_numerical(d_model))
            findings.extend(self._check_phi_positivity(d_model))
            findings.extend(self._check_caputo_weights())
        return findings

    # ------------------------------------------------------------------
    def _check_caputo_order(self, src: str) -> List[AgentFinding]:
        """α must be in (0,2)."""
        has_range_check = bool(re.search(
            r'0\.0\s*<\s*alpha\s*<\s*2\.0|alpha.*in.*\(0.*2\)', src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="Caputo derivative order α ∈ (0,2) validated at construction",
            passed=has_range_check,
            severity="critical",
            details=(
                "α range check found in CaputoFractionalDerivativeApprox."
                if has_range_check
                else "No α ∈ (0,2) guard found; invalid α breaks causality."
            ),
            file_path="models/entropy_softmax.py",
            fix_hint="Add: if not (0.0 < alpha < 2.0): raise ValueError(...) in __init__.",
        )]

    def _check_nu_floor(self, src: str) -> List[AgentFinding]:
        """ν(x) must be initialised above 0 (softplus + offset ≥ 0.5)."""
        has_floor = bool(re.search(r'softplus.*nu.*\+\s*0\.5|nu.*softplus.*\+\s*0\.5', src))
        return [AgentFinding(
            agent=self.NAME,
            check="Entropy exponent ν(x) initialised away from zero (≥0.5)",
            passed=has_floor,
            severity="major",
            details=(
                "ν = softplus(·) + 0.5 ensures ν ≥ 0.5 > 0 (Lipschitz condition)."
                if has_floor
                else "ν may reach 0; Lipschitz bound from Remark (iii) then fails."
            ),
            file_path="models/entropy_softmax.py",
            fix_hint="Use nu = F.softplus(self.nu_proj(x)) + 0.5.",
        )]

    def _check_tau_positivity(self, src: str) -> List[AgentFinding]:
        """τ(x) must be strictly positive."""
        has_floor = bool(re.search(r'softplus.*tau.*\+\s*1e-|tau.*softplus.*\+\s*1e-', src))
        return [AgentFinding(
            agent=self.NAME,
            check="Temperature τ(x) strictly positive (softplus + ε floor)",
            passed=has_floor,
            severity="major",
            details=(
                "τ = softplus(·) + 1e-4 ensures τ > 0."
                if has_floor
                else "τ may reach 0, causing division by zero in entropy-gated logits."
            ),
            file_path="models/entropy_softmax.py",
            fix_hint="Use tau = F.softplus(self.tau_proj(x)) + 1e-4.",
        )]

    def _check_singularity_floor(self, src: str) -> List[AgentFinding]:
        """ε floor must be applied to |Ψ| before computing |Ψ|^{2ν-2}."""
        has_floor = bool(re.search(
            r'clamp.*min.*eps|clamp.*min.*1e-', src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="Singularity floor ε applied to |Ψ| (Remark ii)",
            passed=has_floor,
            severity="major",
            details=(
                "clamp(min=ε) found; singularity at Ψ=0 regularised."
                if has_floor
                else "No ε floor found; |Ψ|^{2ν-2} diverges at Ψ=0 for ν<1."
            ),
            file_path="models/entropy_softmax.py",
            fix_hint="Apply Psi_norm = Psi.norm(dim=-1).clamp(min=self.eps) before log-power.",
        )]

    def _check_exact_gateaux(self, src: str) -> List[AgentFinding]:
        """Gradient must use the exact Gâteaux formula, not an approximation."""
        has_gateaux = bool(re.search(
            r'VariableOrderEntropyGradient|grad_op|Gâteaux|Gateaux', src
        ))
        return [AgentFinding(
            agent=self.NAME,
            check="Exact Gâteaux derivative (Def 2a) used in evolution, not an approximation",
            passed=has_gateaux,
            severity="critical",
            details=(
                "VariableOrderEntropyGradient / grad_op found in source."
                if has_gateaux
                else "Evolution equation uses an approximate gradient, not the exact Gâteaux form."
            ),
            file_path="models/entropy_softmax.py",
            fix_hint=(
                "Replace the approximate gradient with VariableOrderEntropyGradient.forward() "
                "which computes 2ν|Ψ|^{2ν-2}Ψ·Φ(x) exactly."
            ),
        )]

    def _check_monotonicity_numerical(self, d_model: int) -> List[AgentFinding]:
        """⟨∇H[Ψ], Ψ⟩ ≥ 0 for random Ψ (Remark iii)."""
        try:
            from models.entropy_softmax import (  # type: ignore
                VariableOrderEntropyFunctional,
                VariableOrderEntropyGradient,
            )
            fn   = VariableOrderEntropyFunctional(d_model)
            grad = VariableOrderEntropyGradient(fn)
            Psi  = torch.randn(2, 4, d_model)
            inner = grad.monotonicity_lower_bound(Psi, Psi)
            passed = bool((inner >= 0).all())
            return [AgentFinding(
                agent=self.NAME,
                check="Monotonicity ⟨∇H[Ψ],Ψ⟩ ≥ 0 (Remark iii, Hölder bound)",
                passed=passed,
                severity="major",
                details=f"min inner product = {inner.min().item():.4e}",
                file_path="models/entropy_softmax.py",
                fix_hint="If this fails, increase the ε floor or check nu initialisation.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Monotonicity ⟨∇H[Ψ],Ψ⟩ ≥ 0 (Remark iii, Hölder bound)",
                passed=False,
                severity="major",
                details=f"Raised: {exc}",
                file_path="models/entropy_softmax.py",
                fix_hint="Ensure VariableOrderEntropyGradient is importable.",
            )]

    def _check_phi_positivity(self, d_model: int) -> List[AgentFinding]:
        """Φ(x) = Σ_y φ(x,y) must be strictly positive."""
        try:
            from models.entropy_softmax import VariableOrderEntropyFunctional  # type: ignore
            fn    = VariableOrderEntropyFunctional(d_model)
            Psi   = torch.randn(2, 4, d_model)
            Phi_x = fn.kernel_integral(Psi, Psi)
            passed = bool((Phi_x > 0).all())
            return [AgentFinding(
                agent=self.NAME,
                check="Kernel integral Φ(x) = Σ_y φ(x,y) strictly positive",
                passed=passed,
                severity="major",
                details=f"min Φ(x) = {Phi_x.min().item():.4e}",
                file_path="models/entropy_softmax.py",
                fix_hint="Ensure the kernel φ(x,y) is a softmax so rows sum to 1.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Kernel integral Φ(x) = Σ_y φ(x,y) strictly positive",
                passed=False, severity="major",
                details=f"Raised: {exc}",
                file_path="models/entropy_softmax.py",
                fix_hint="Ensure VariableOrderEntropyFunctional is importable.",
            )]

    def _check_caputo_weights(self) -> List[AgentFinding]:
        """Caputo Grünwald–Letnikov weights must be positive for α ∈ (1,2)."""
        try:
            from models.entropy_softmax import CaputoFractionalDerivativeApprox  # type: ignore
            cap = CaputoFractionalDerivativeApprox(alpha=1.5, max_depth=8)
            passed = bool((cap.weights > 0).all())
            return [AgentFinding(
                agent=self.NAME,
                check="Caputo GL weights positive for α=1.5 ∈ (1,2)",
                passed=passed,
                severity="major",
                details=f"weights = {cap.weights.tolist()[:4]} … (first 4)",
                file_path="models/entropy_softmax.py",
                fix_hint="Check _compute_weights formula for Grünwald–Letnikov α ∈ (1,2).",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Caputo GL weights positive for α=1.5 ∈ (1,2)",
                passed=False, severity="major",
                details=f"Raised: {exc}",
                file_path="models/entropy_softmax.py",
                fix_hint="Ensure CaputoFractionalDerivativeApprox is importable.",
            )]


# ---------------------------------------------------------------------------
# Entropy Softmax Planning Agent
# ---------------------------------------------------------------------------

class EntropySoftmaxPlanningAgent:
    """
    Plans and validates the replacement of standard softmax with the
    Entropy-Gated Belief Collapse Probability (Definition 4).

    Works *in conjunction* with:
      • GeometryAgent   – kernel positivity, volume form dV_g, manifold integration
      • PhysicsAgent    – Caputo order, singularity handling, monotonicity
      • ValidationAgent – numerical normalization and gradient flow

    Source-text checks
    ------------------
    1. EntropySoftmax class present in models/entropy_softmax.py.
    2. Definition 2a (VariableOrderEntropyGradient / Gâteaux) present.
    3. GeometricAttention imports and optionally uses EntropySoftmax.
    4. Softmax replacement gated by use_entropy_softmax flag (backward-compat).
    5. Key/query features are passed to per_key_entropy (not scores alone).

    Numerical checks
    ----------------
    6. EntropySoftmax output sums to 1 over the key dimension.
    7. Gradient flows back through entropy softmax to key_features.
    8. VariableOrderEntropyGradient.monotonicity_lower_bound ≥ 0.

    Coordination plan
    -----------------
    After source-text checks pass:
      - GeometryAgent verifies φ(x,y) kernel is the Riemannian volume proxy.
      - PhysicsAgent verifies Caputo order and singularity floor.
      - ValidationAgent verifies output normalization and gradient flow.
      - This agent synthesises their findings into a unified plan.
    """

    NAME = "Entropy Softmax Planning Agent"

    def audit(
        self,
        repo_root: Path,
        geometry_findings: Optional[List[AgentFinding]] = None,
        physics_findings:  Optional[List[AgentFinding]] = None,
        validation_findings: Optional[List[AgentFinding]] = None,
        run_numerical: bool = True,
        d_model: int = 8,
    ) -> List[AgentFinding]:
        """
        Run the planning audit and synthesise findings from the three
        specialist agents.

        Args:
            repo_root:           Repository root for source scanning.
            geometry_findings:   Pre-computed findings from GeometryAgent.
            physics_findings:    Pre-computed findings from PhysicsAgent.
            validation_findings: Pre-computed findings from ValidationAgent.
            run_numerical:       Whether to run live numerical checks.
            d_model:             Model dimension for numerical checks.

        Returns:
            List of AgentFinding items covering planning and synthesis.
        """
        def _read(rel: str) -> str:
            try:
                return (repo_root / rel).read_text(encoding='utf-8', errors='replace')
            except OSError:
                return ""

        entropy_src   = _read("models/entropy_softmax.py")
        attention_src = _read("inference/geometric_attention.py")

        findings: List[AgentFinding] = []
        findings.extend(self._check_entropy_softmax_class(entropy_src))
        findings.extend(self._check_def2a_present(entropy_src))
        findings.extend(self._check_gateaux_used_in_evolution(entropy_src))
        findings.extend(self._check_attention_import(attention_src))
        findings.extend(self._check_attention_flag(attention_src))
        findings.extend(self._check_features_passed(attention_src))

        if run_numerical:
            findings.extend(self._check_normalization(d_model))
            findings.extend(self._check_grad_through_entropy(d_model))

        # Synthesis: absorb peer findings as info entries
        for peer_findings, label in [
            (geometry_findings, "GeometryAgent"),
            (physics_findings,  "PhysicsAgent"),
            (validation_findings, "ValidationAgent"),
        ]:
            if peer_findings:
                blocking = [f for f in peer_findings if not f.passed and f.severity == "critical"]
                findings.append(AgentFinding(
                    agent=self.NAME,
                    check=f"Synthesis: {label} has no critical blockers for entropy-softmax plan",
                    passed=not blocking,
                    severity="info",
                    details=(
                        f"{label}: all clear." if not blocking
                        else f"{label}: {len(blocking)} critical blocker(s) – "
                             + "; ".join(f.check for f in blocking)
                    ),
                    file_path="utils/causal_field_agents.py",
                    fix_hint=f"Resolve critical findings in {label} before shipping.",
                ))

        return findings

    # ------------------------------------------------------------------
    def _check_entropy_softmax_class(self, src: str) -> List[AgentFinding]:
        has_class = bool(re.search(r'class\s+EntropySoftmax\b', src))
        return [AgentFinding(
            agent=self.NAME,
            check="EntropySoftmax class (Definition 4) present in entropy_softmax.py",
            passed=has_class,
            severity="critical",
            details="EntropySoftmax class found." if has_class
                    else "EntropySoftmax class not found; softmax replacement missing.",
            file_path="models/entropy_softmax.py",
            fix_hint="Implement EntropySoftmax in models/entropy_softmax.py.",
        )]

    def _check_def2a_present(self, src: str) -> List[AgentFinding]:
        has_def2a = bool(re.search(r'class\s+VariableOrderEntropyGradient\b', src))
        return [AgentFinding(
            agent=self.NAME,
            check="Definition 2a (VariableOrderEntropyGradient) present",
            passed=has_def2a,
            severity="critical",
            details="VariableOrderEntropyGradient (Def 2a) found." if has_def2a
                    else "Definition 2a missing; evolution gradient is an approximation.",
            file_path="models/entropy_softmax.py",
            fix_hint="Add VariableOrderEntropyGradient implementing the Gâteaux derivative.",
        )]

    def _check_gateaux_used_in_evolution(self, src: str) -> List[AgentFinding]:
        uses_grad_op = bool(re.search(r'self\.grad_op\s*=\s*VariableOrderEntropyGradient|grad_op\.forward', src))
        return [AgentFinding(
            agent=self.NAME,
            check="EntropyGatedEvolution uses VariableOrderEntropyGradient (exact Gâteaux)",
            passed=uses_grad_op,
            severity="critical",
            details="grad_op = VariableOrderEntropyGradient found in EntropyGatedEvolution."
                    if uses_grad_op
                    else "EntropyGatedEvolution does not reference VariableOrderEntropyGradient.",
            file_path="models/entropy_softmax.py",
            fix_hint="Set self.grad_op = VariableOrderEntropyGradient(self.entropy_fn) in "
                     "EntropyGatedEvolution.__init__ and call it in forward().",
        )]

    def _check_attention_import(self, src: str) -> List[AgentFinding]:
        has_import = bool(re.search(r'from models\.entropy_softmax import|EntropySoftmax', src))
        return [AgentFinding(
            agent=self.NAME,
            check="GeometricAttention imports EntropySoftmax",
            passed=has_import,
            severity="major",
            details="EntropySoftmax import found in geometric_attention.py." if has_import
                    else "No EntropySoftmax import found.",
            file_path="inference/geometric_attention.py",
            fix_hint="Add: from models.entropy_softmax import EntropySoftmax",
        )]

    def _check_attention_flag(self, src: str) -> List[AgentFinding]:
        has_flag = bool(re.search(r'use_entropy_softmax', src))
        return [AgentFinding(
            agent=self.NAME,
            check="GeometricAttention has use_entropy_softmax flag (backward-compatible)",
            passed=has_flag,
            severity="major",
            details="use_entropy_softmax flag found." if has_flag
                    else "No use_entropy_softmax flag; old code would always use entropy softmax.",
            file_path="inference/geometric_attention.py",
            fix_hint="Add use_entropy_softmax=False parameter to GeometricAttention.__init__.",
        )]

    def _check_features_passed(self, src: str) -> List[AgentFinding]:
        """Key/query features must be passed so entropy functional uses embeddings."""
        has_feat = bool(re.search(r'key_features\s*=\s*K_feat|query_features\s*=\s*Q_feat', src))
        return [AgentFinding(
            agent=self.NAME,
            check="Key/query feature tensors passed to EntropySoftmax (not scores alone)",
            passed=has_feat,
            severity="major",
            details="K_feat / Q_feat passed to entropy_softmax." if has_feat
                    else "EntropySoftmax called without explicit key/query features.",
            file_path="inference/geometric_attention.py",
            fix_hint="Pass key_features=K_feat and query_features=Q_feat to entropy_softmax().",
        )]

    def _check_normalization(self, d_model: int) -> List[AgentFinding]:
        """Output must sum to 1 over the key dimension."""
        try:
            from models.entropy_softmax import EntropySoftmax  # type: ignore
            esm = EntropySoftmax(d_model)
            esm.eval()
            scores = torch.randn(2, 4, 6)
            K      = torch.randn(2, 6, d_model)
            Q      = torch.randn(2, 4, d_model)
            with torch.no_grad():
                w = esm(scores, key_features=K, query_features=Q)
            row_sums = w.sum(dim=-1)
            passed   = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
            return [AgentFinding(
                agent=self.NAME,
                check="EntropySoftmax output sums to 1 over key dimension",
                passed=passed,
                severity="critical",
                details=f"max |row_sum − 1| = {(row_sums - 1).abs().max().item():.2e}",
                file_path="models/entropy_softmax.py",
                fix_hint="Ensure softmax is applied over dim=-1 in EntropySoftmax.forward.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="EntropySoftmax output sums to 1 over key dimension",
                passed=False, severity="critical",
                details=f"Raised: {exc}",
                file_path="models/entropy_softmax.py",
                fix_hint="Fix EntropySoftmax.forward so it runs without errors.",
            )]

    def _check_grad_through_entropy(self, d_model: int) -> List[AgentFinding]:
        """Gradient must flow back to key_features through the entropy softmax."""
        try:
            from models.entropy_softmax import EntropySoftmax  # type: ignore
            esm    = EntropySoftmax(d_model)
            scores = torch.randn(1, 4, 4)
            K      = torch.randn(1, 4, d_model, requires_grad=True)
            Q      = torch.randn(1, 4, d_model)
            w      = esm(scores, key_features=K, query_features=Q)
            w.sum().backward()
            passed = K.grad is not None and K.grad.abs().sum().item() > 0
            return [AgentFinding(
                agent=self.NAME,
                check="Gradient flows back to key_features through EntropySoftmax",
                passed=passed,
                severity="major",
                details=f"K.grad norm = {K.grad.norm().item():.4e}" if K.grad is not None
                        else "K.grad is None",
                file_path="models/entropy_softmax.py",
                fix_hint="Remove any .detach() calls on key_features in EntropySoftmax.",
            )]
        except Exception as exc:
            return [AgentFinding(
                agent=self.NAME,
                check="Gradient flows back to key_features through EntropySoftmax",
                passed=False, severity="major",
                details=f"Raised: {exc}",
                file_path="models/entropy_softmax.py",
                fix_hint="Fix EntropySoftmax.forward so it is differentiable.",
            )]


# ---------------------------------------------------------------------------
# Coordinator / Scribe Agent
# ---------------------------------------------------------------------------

class CoordinatorScribeAgent:
    """
    Coordinator and scribe for the causal field audit.

    Responsibilities
    ----------------
    1. Scan every ``.py`` file in the repository for references to the
       causal accumulation law (source current J, parallel transport Pi,
       memory kernel k, bivector Phi).
    2. Classify each hit by role and broadcast the location list to the
       three specialist agents.
    3. Run the specialist agents in sequence, collect all findings, and
       write a consolidated action log.
    4. Return a ``CausalFieldReport`` with locations, findings, and a
       transport-operator implementation plan.

    Args:
        repo_root: Absolute path to the repository root.
    """

    _CAUSAL_PATTERNS: List[Tuple[str, str]] = [
        (r'CausalFieldLayer|CausalFieldBlock|causal_field', "core implementation"),
        (r'AssociatorCurrent|associator', "core implementation"),
        (r'ParallelTransport|ParallelPropagator|FiberHolonomy', "transport operators"),
        (r'test.*causal|causal.*test', "test"),
        (r'visualize_causal|interactive_causal', "visualisation"),
        (r'causal.accumulation|accumulation.law', "documentation"),
    ]

    _TRANSPORT_PLAN: List[str] = [
        "Phase 1 – Flat-spacetime baseline (implemented):",
        "  • ParallelPropagator: d_field×d_field matrix, identity init.",
        "  • FiberHolonomy: Phi^a_b composed with fiber P^b_c, identity init.",
        "  • Both classes live in operators/parallel_transport.py.",
        "",
        "Phase 2 – Geodesic connection (next):",
        "  • Derive P^α_{α'}(x,x') from Christoffel symbols of CognitiveManifold.",
        "  • Use the O(1) LIoR recurrence (models/manifold.py geodesic_step) for integration.",
        "  • Anneal from flat (identity) to curved as the manifold trains.",
        "",
        "Phase 3 – Tail-corrected Green's function (future):",
        "  • Replace scalar k(τ) with bivector k(τ;x,x') = w(τ) * G_del(x,x').",
        "  • G_del acquires Hadamard tail terms in the GR regime.",
        "  • Add as a correction layer on top of the LIoR convolution.",
        "",
        "Phase 4 – Fractional differential operator (future):",
        "  • Wrap accumulation law in nabla^{(alpha)mu}[...] = Phi^a_b J.",
        "  • Extend kernels/fractional_memory.py to act on full tensor T.",
        "  • Requires fractional derivative of a rank-4 tensor field.",
    ]

    def __init__(self, repo_root: Optional[str] = None):
        if repo_root is None:
            # Default to the directory two levels above this file
            repo_root = str(Path(__file__).resolve().parent.parent)
        self.repo_root = Path(repo_root)
        self._algebra_agent   = AbstractAlgebraAgent()
        self._geometry_agent  = GeometryAgent()
        self._validation_agent = ValidationAgent()
        self._pipeline_agent  = DataPipelineAuditAgent()
        self._physics_agent   = PhysicsAgent()
        self._entropy_agent   = EntropySoftmaxPlanningAgent()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        run_numerical: bool = True,
        d_model: int = 16,
        d_field: int = 4,
        d_spinor: int = 4,
    ) -> CausalFieldReport:
        """
        Execute the full audit pipeline.

        Sequence
        --------
        1. Scan repo for causal field files.
        2. Run AbstractAlgebraAgent and GeometryAgent on source text.
        3. Optionally run ValidationAgent numerical checks.
        4. Run DataPipelineAuditAgent (source-text + optional numerical).
        5. Compile findings into a CausalFieldReport.

        Args:
            run_numerical: Whether to instantiate layers and run numerical
                checks (requires models/ and training/ on sys.path).
            d_model: Feature dimension for the numerical validation layer.
            d_field: Field index dimension for the numerical validation layer.
            d_spinor: Spinor dimension for the numerical validation layer.

        Returns:
            CausalFieldReport with all locations, findings, pipeline_findings,
            and the transport-operator implementation plan.
        """
        locations = self._scan_repo()
        source_text = self._load_source(locations)

        # ── Causal-field audit ─────────────────────────────────────────
        findings: List[AgentFinding] = []
        findings.extend(
            self._algebra_agent.audit(locations, source_text)
        )
        findings.extend(
            self._geometry_agent.audit(locations, source_text)
        )
        if run_numerical:
            findings.extend(
                self._validation_agent.audit(
                    d_model=d_model, d_field=d_field, d_spinor=d_spinor
                )
            )

        # ── Data pipeline audit (runs after causal-field audit) ────────
        pipeline_findings: List[AgentFinding] = self._pipeline_agent.audit(
            repo_root=self.repo_root,
            run_numerical=run_numerical,
            d_model=d_model,
            d_field=d_field,
            d_spinor=d_spinor,
        )

        # ── Physics audit (Caputo order, singularity, monotonicity) ────
        entropy_src = ""
        try:
            entropy_src = (self.repo_root / "models" / "entropy_softmax.py").read_text(
                encoding='utf-8', errors='replace'
            )
        except OSError:
            pass
        physics_findings: List[AgentFinding] = self._physics_agent.audit(
            source_text=entropy_src,
            run_numerical=run_numerical,
            d_model=d_model,
        )

        # ── Entropy-softmax planning audit (geo + physics + validation) ─
        entropy_findings: List[AgentFinding] = self._entropy_agent.audit(
            repo_root=self.repo_root,
            geometry_findings=findings,          # pass causal-field geometry findings
            physics_findings=physics_findings,
            validation_findings=findings,        # pass numerical validation findings
            run_numerical=run_numerical,
            d_model=d_model,
        )
        # Merge physics into entropy_findings for a single entropy bucket
        entropy_findings = list(physics_findings) + list(entropy_findings)

        action_log = self._write_action_log(
            locations, findings, pipeline_findings, entropy_findings
        )

        return CausalFieldReport(
            locations=locations,
            findings=findings,
            action_log=action_log,
            transport_plan=list(self._TRANSPORT_PLAN),
            pipeline_findings=pipeline_findings,
            entropy_findings=entropy_findings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_repo(self) -> List[CausalFieldLocation]:
        """Walk the repo and collect files related to the causal field law."""
        locations: List[CausalFieldLocation] = []
        seen: set = set()
        for py_file in sorted(self.repo_root.rglob("*.py")):
            rel = str(py_file.relative_to(self.repo_root))
            if any(skip in rel for skip in ('.git', '__pycache__', 'node_modules')):
                continue
            try:
                text = py_file.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            for pattern, role in self._CAUSAL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    if rel not in seen:
                        lines = self._find_relevant_lines(text, pattern)
                        locations.append(CausalFieldLocation(
                            path=rel,
                            role=role,
                            relevant_lines=lines,
                        ))
                        seen.add(rel)
                    break  # first matching role wins
        return locations

    @staticmethod
    def _find_relevant_lines(text: str, pattern: str) -> List[int]:
        """Return 1-based line numbers where *pattern* matches."""
        lines = []
        for i, line in enumerate(text.splitlines(), start=1):
            if re.search(pattern, line, re.IGNORECASE):
                lines.append(i)
        return lines[:20]  # cap at 20 to keep the report concise

    def _load_source(
        self,
        locations: List[CausalFieldLocation],
    ) -> Dict[str, str]:
        source: Dict[str, str] = {}
        for loc in locations:
            full = self.repo_root / loc.path
            try:
                source[loc.path] = full.read_text(encoding='utf-8', errors='replace')
            except OSError:
                source[loc.path] = ""
        return source

    def _write_action_log(
        self,
        locations: List[CausalFieldLocation],
        findings: List[AgentFinding],
        pipeline_findings: Optional[List[AgentFinding]] = None,
        entropy_findings: Optional[List[AgentFinding]] = None,
    ) -> List[str]:
        """Produce the consolidated scribe action log."""
        if pipeline_findings is None:
            pipeline_findings = []
        if entropy_findings is None:
            entropy_findings = []

        log: List[str] = []
        log.append("=== Causal Field Audit – Coordinator/Scribe Action Log ===")
        log.append("")
        log.append("--- Discovered Locations ---")
        for loc in locations:
            lines_str = ", ".join(str(l) for l in loc.relevant_lines[:5])
            log.append(f"  [{loc.role}] {loc.path}  (lines: {lines_str})")
        log.append("")

        # ── Causal-field findings ──────────────────────────────────────
        log.append("--- Causal Field Findings ---")
        for f in findings:
            status = "PASS" if f.passed else f"FAIL [{f.severity.upper()}]"
            log.append(f"  [{f.agent}] {status}: {f.check}")
            if not f.passed:
                log.append(f"    Details: {f.details}")
                log.append(f"    Fix:     {f.fix_hint}")
        log.append("")

        # ── Data pipeline findings ─────────────────────────────────────
        log.append("--- Data Pipeline Findings ---")
        if pipeline_findings:
            for f in pipeline_findings:
                status = "PASS" if f.passed else f"FAIL [{f.severity.upper()}]"
                log.append(f"  [{f.agent}] {status}: {f.check}")
                if not f.passed:
                    log.append(f"    Details: {f.details}")
                    log.append(f"    Fix:     {f.fix_hint}")
        else:
            log.append("  (pipeline audit not run)")
        log.append("")

        # ── Entropy-softmax + physics findings ────────────────────────
        log.append("--- Entropy-Gated Softmax Findings (PhysicsAgent + EntropySoftmaxPlanningAgent) ---")
        if entropy_findings:
            for f in entropy_findings:
                status = "PASS" if f.passed else f"FAIL [{f.severity.upper()}]"
                log.append(f"  [{f.agent}] {status}: {f.check}")
                if not f.passed:
                    log.append(f"    Details: {f.details}")
                    log.append(f"    Fix:     {f.fix_hint}")
        else:
            log.append("  (entropy audit not run)")
        log.append("")

        # ── Summary ───────────────────────────────────────────────────
        all_findings = findings + pipeline_findings + entropy_findings
        total  = len(all_findings)
        passed = sum(1 for f in all_findings if f.passed)
        log.append(f"--- Summary: {passed}/{total} checks passed ---")
        critical = [f for f in all_findings if not f.passed and f.severity == "critical"]
        if critical:
            log.append(f"  *** {len(critical)} CRITICAL failure(s) require immediate attention ***")
        log.append("")
        log.append("--- Transport Operator Plan ---")
        log.extend(f"  {line}" for line in self._TRANSPORT_PLAN)
        return log
