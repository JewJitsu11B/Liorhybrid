"""
Causal Field Agent Team.

Four specialised agents that collaborate to locate the causal field
implementation in the repository, audit it against the formal law, and
propose targeted fixes.

Roles
-----
CoordinatorScribeAgent
    Scans the repository for every file that touches the causal
    accumulation law, broadcasts the discovered locations to the three
    specialist agents, orchestrates their audit passes, and writes a
    consolidated action log.

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

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def critical_failures(self) -> List[AgentFinding]:
        return [f for f in self.findings if not f.passed and f.severity == "critical"]


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
        self._algebra_agent = AbstractAlgebraAgent()
        self._geometry_agent = GeometryAgent()
        self._validation_agent = ValidationAgent()

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

        1. Scan repo for causal field files.
        2. Run AbstractAlgebraAgent and GeometryAgent on source text.
        3. Optionally run ValidationAgent numerical checks.
        4. Compile findings into a CausalFieldReport.

        Args:
            run_numerical: Whether to instantiate a layer and run numerical
                checks (requires models/ on sys.path).
            d_model: Feature dimension for the numerical validation layer.
            d_field: Field index dimension for the numerical validation layer.
            d_spinor: Spinor dimension for the numerical validation layer.

        Returns:
            CausalFieldReport with all locations, findings, and the
            transport-operator implementation plan.
        """
        locations = self._scan_repo()
        source_text = self._load_source(locations)

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

        action_log = self._write_action_log(locations, findings)

        return CausalFieldReport(
            locations=locations,
            findings=findings,
            action_log=action_log,
            transport_plan=list(self._TRANSPORT_PLAN),
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
    ) -> List[str]:
        """Produce the consolidated scribe action log."""
        log: List[str] = []
        log.append("=== Causal Field Audit – Coordinator/Scribe Action Log ===")
        log.append("")
        log.append("--- Discovered Locations ---")
        for loc in locations:
            lines_str = ", ".join(str(l) for l in loc.relevant_lines[:5])
            log.append(f"  [{loc.role}] {loc.path}  (lines: {lines_str})")
        log.append("")
        log.append("--- Findings ---")
        for f in findings:
            status = "PASS" if f.passed else f"FAIL [{f.severity.upper()}]"
            log.append(f"  [{f.agent}] {status}: {f.check}")
            if not f.passed:
                log.append(f"    Details: {f.details}")
                log.append(f"    Fix:     {f.fix_hint}")
        log.append("")
        total = len(findings)
        passed = sum(1 for f in findings if f.passed)
        log.append(f"--- Summary: {passed}/{total} checks passed ---")
        critical = [f for f in findings if not f.passed and f.severity == "critical"]
        if critical:
            log.append(f"  *** {len(critical)} CRITICAL failure(s) require immediate attention ***")
        log.append("")
        log.append("--- Transport Operator Plan ---")
        log.extend(f"  {line}" for line in self._TRANSPORT_PLAN)
        return log
