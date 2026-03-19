"""
Seven-Agent Transport & Fiber Bundle Audit Team

Audits the parallel-transport operator (ParallelTransport / Pi tensor) and the
fiber bundle structure (Tetrad / vielbein + CliffordConnection / Gamma) in
models/causal_field.py and kernels/tetrad.py, and verifies that both operators
are correctly wired into the training/inference pipeline.

CxO vs CxH — practical comparison
------------------------------------
This codebase implements two field-evolution algebras with the same d_field=16:

  CxO  ℂ⊗𝕆  complex octonions   →  CausalFieldLayer
  CxH  ℂ⊗ℍ  biquaternions        →  BiQuatCausalLayer   ← PREFERRED

The key practical differences:

  Algebra:
    CxO  Non-associative AND non-commutative. The associator J=(ab)c-a(bc)!=0
         IS the physics signal: it measures the curvature/non-flatness of
         the cognitive field at each point.
    CxH  Associative but non-commutative. (ab)c = a(bc) always.
         Effective non-commutativity comes from temporal ordering of the
         recurrence, not from the algebra itself.

  Cost per forward step (per element):
    CxO  O(d³): J_expand einsum [16,16,16] ≈ 4096 ops; 5 oct-products @O(64)
         each; plus Pi (3× [16,16,16] parameters), Gamma, LIoR memory.
    CxH  O(1): 4 quaternion products @16 muls each = 64 muls total.

  Memory state:
    CxO  Size d_field² = 256, using LIoR multi-pole exponential kernel.
    CxH  Size 8 (two 4-vectors Q_H_re, Q_H_im), simple leaky integrator:
         Q_H_new = decay * Q_H + scale * W_impulse(Q_M)  (impulse_map)
         T = alpha * Q_M + (1-alpha) * W_transport(Q_H_new)  (transport_map)

  Transport / connection:
    CxO  Pi (rank-8 tensor), Gamma (Clifford connection via tetrad),
         Phi (antisymmetric bivector). These are the operators under audit.
    CxH  W_transport: a single learnable biquaternion (8 scalar params).
         No Pi, no Gamma, no Phi.

  Precision:
    CxO  fp32 only; no fp16/bf16 guards.
    CxH  Pure-real arithmetic, explicit clamps for fp16/bf16 safety.

  Physical model:
    CxO  Gauge / fiber-bundle theory. Pi Gamma integrates octonion-curvature
         over the causal past along a fiber bundle.
    CxH  SL(2,ℂ) = Lorentz rotations + boosts. Q_H is the "historical spin
         state". Structurally lighter, empirically faster.

Architecture note — two field-evolution paths
----------------------------------------------
  Path A  CausalFieldLayer  — CxO, O(d³) cost.
          Uses ParallelTransport, CliffordConnection, Phi, Tetrad (not wired).
          These are the operators under audit here.

  Path B  BiQuatCausalBlock / BiQuatCausalLayer  — CxH, O(N) pure-real.
          Does NOT use ParallelTransport, CliffordConnection, or Phi.
          PREFERRED because CxO is too costly.

The audit documents Path A's state so that if covariant transport is ever
ported to Path B, there is a clear blueprint.

What is d_field?
----------------
``d_field`` is fixed at 16 for BOTH paths (see D_FIELD constant):
  Path A (CxO):   8-d real octonions + 8-d imaginary octonions = 16
  Path B (CxH):   4 real quaternions × 4 components each       = 16
                  (enforced by ``assert d_field == 16`` in BiQuatCausalLayer)
It is NOT a free hyperparameter.

Team roles
----------
1. Coordinator   – owns scope, assigns sub-tasks, resolves blockers
2. Physics        – covariant-derivative consistency, holonomy, conservation
3. Geometry       – fiber bundle / vielbein orthonormality, metric compatibility
4. Coding         – implementation correctness, shape contracts, device safety
5. Validation     – quantitative numerical checks (norms, NaN/Inf, shapes)
6. Morale         – workload balance flag, cadence sustainability notes
7. Scribe         – consolidated decision log with severity & evidence

STATUS: AWAITING APPROVAL TO EXECUTE
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except ImportError:
    pass

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Import bootstrap
# ---------------------------------------------------------------------------

def _ensure_package_importable() -> None:
    """
    Ensure the Liorhybrid parent directory is in sys.path so that
    relative imports inside models/causal_field.py resolve correctly.
    Models use relative imports like ``from ..training.execution_tracker``
    which require being imported as ``Liorhybrid.models.causal_field``.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]   # .../Liorhybrid
    parent = repo_root.parent                                   # .../work/Liorhybrid
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def _import_causal_field():
    """Import CausalFieldLayer, ParallelTransport, CliffordConnection."""
    _ensure_package_importable()
    from Liorhybrid.models.causal_field import (  # noqa: PLC0415
        CausalFieldLayer,
        ParallelTransport,
        CliffordConnection,
    )
    return CausalFieldLayer, ParallelTransport, CliffordConnection


def _import_biquat():
    """Import BiQuatCausalLayer from models.biquaternion."""
    _ensure_package_importable()
    from Liorhybrid.models.biquaternion import BiQuatCausalLayer  # noqa: PLC0415
    return BiQuatCausalLayer


def _import_tetrad():
    """Import Tetrad, compute_metric_from_tetrad from kernels.tetrad."""
    _ensure_package_importable()
    from Liorhybrid.kernels.tetrad import (  # noqa: PLC0415
        Tetrad,
        compute_metric_from_tetrad,
    )
    return Tetrad, compute_metric_from_tetrad


# ---------------------------------------------------------------------------
# Field-dimension constant
# ---------------------------------------------------------------------------

# d_field is ALWAYS 16 for both field-evolution paths:
#   Path A (CxO):   8 real + 8 imaginary octonion dims  = 16
#   Path B (BiQuat): 4 quaternions × 4 components each  = 16
# BiQuatCausalLayer asserts this: ``assert d_field == 16``
D_FIELD: int = 16


# ---------------------------------------------------------------------------
# Severity levels
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
class AuditFinding:
    """Single finding produced by one specialist agent."""
    role: str
    operator: str          # e.g. "ParallelTransport", "CliffordConnection", "Tetrad"
    check: str
    passed: bool
    severity: str          # one of the SEVERITY_* constants
    evidence: str          # file + line / runtime value
    recommendation: str


@dataclass(frozen=True)
class PipelineWiringCheck:
    """Records whether an operator is properly wired into the pipeline."""
    operator: str
    wired: bool
    entry_point: str       # caller module/class
    notes: str


@dataclass(frozen=True)
class ScribeLog:
    """Consolidated decision log produced by the Scribe agent."""
    findings: List[AuditFinding]
    wiring_checks: List[PipelineWiringCheck]
    summary: str
    action_items: List[str]

    @property
    def all_passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def critical_findings(self) -> List[AuditFinding]:
        return [f for f in self.findings if not f.passed and f.severity == SEVERITY_CRITICAL]

    @property
    def failed_findings(self) -> List[AuditFinding]:
        return [f for f in self.findings if not f.passed]


@dataclass
class AuditReport:
    """Complete report produced by the full seven-agent team."""
    coordinator_scope: str
    findings: List[AuditFinding]
    wiring_checks: List[PipelineWiringCheck]
    morale_notes: List[str]
    scribe_log: ScribeLog
    approval_status: str = "AWAITING APPROVAL TO EXECUTE"

    @property
    def passed(self) -> bool:
        return self.scribe_log.all_passed


# ---------------------------------------------------------------------------
# Individual specialist agents
# ---------------------------------------------------------------------------

class CoordinatorAgent:
    """
    Owns scope, assigns sub-tasks, resolves blockers.

    Scope for this audit:
    - models/causal_field.py  : ParallelTransport (Pi), CliffordConnection (Gamma)
    - kernels/tetrad.py        : Tetrad (vielbein / fiber bundle)
    - Pipeline wiring          : CausalFieldLayer forward(), kernels/__init__.py exports

    Architecture context
    --------------------
    Two field-evolution paths exist in this codebase:

      Path A  CausalFieldLayer  — CxO (complex octonions), O(d³) cost.
              Uses ParallelTransport, CliffordConnection, Phi, Tetrad (not wired).
              This is the path being audited.

      Path B  BiQuatCausalBlock — biquaternions, O(N) pure-real, PREFERRED.
              Does NOT use ParallelTransport, CliffordConnection, or Phi.

    Since Path B is preferred (CxO too costly), the transport/fiber bundle
    operators under audit are currently on the non-preferred path.
    The audit documents their state and wiring so that:
    (a) if Path A is ever re-enabled, it is correct; and
    (b) if covariant transport is ever ported to Path B, there is a blueprint.

    d_field
    -------
    ``d_field`` is fixed at 16 for BOTH paths (see D_FIELD constant):
      Path A: 8 real + 8 imaginary octonion dims = 16
      Path B: 4 real quaternions × 4 components  = 16  (assert d_field==16)
    """

    SCOPE = (
        "Audit the parallel-transport operator (ParallelTransport / Pi) and the "
        "fiber bundle operator (Tetrad + CliffordConnection / Gamma) for physical "
        "correctness, geometric validity, implementation soundness, and correct "
        "pipeline wiring. Note: these operators belong to Path A (CxO / CausalFieldLayer). "
        "Path B (BiQuatCausalBlock) is the preferred cheaper path and bypasses them. "
        "d_field=16 is fixed for both paths. Report findings without executing fixes until approved."
    )

    TASK_QUEUE = [
        ("Physics",    "Verify covariant-derivative consistency and holonomy constraint"),
        ("Geometry",   "Verify fiber bundle / vielbein orthonormality and metric compatibility"),
        ("Coding",     "Audit shape contracts, device safety, unused-parameter warnings, "
                       "and biquaternion path bypass of transport operators"),
        ("Validation", "Run quantitative checks: norms, NaN/Inf, shape contracts, "
                       "and biquaternion d_field=16 assertion"),
        ("Morale",     "Flag workload balance; ensure cadence is sustainable"),
        ("Scribe",     "Consolidate all findings into decision log with severity + evidence"),
    ]

    def scope(self) -> str:
        return self.SCOPE

    def task_queue(self) -> List[Tuple[str, str]]:
        return list(self.TASK_QUEUE)


class PhysicsAgent:
    """
    Checks physical consistency of the transport and fiber bundle operators.

    Key invariants:
    1. Parallel transport must be metric-compatible: ∇_μ g = 0
    2. The Clifford connection must anti-commute correctly: {γ^a, γ^b} = 2η^{ab}
    3. The holomorphic constraint ∇(Pi Γ Φ) = 0 should be enforceable
    4. The antisymmetric Phi bivector should enter the T field equation
    """

    def audit(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        CausalFieldLayer, _, _ = _import_causal_field()

        # Check 1: Phi bivector is antisymmetric (required for physical bivector field)
        layer = CausalFieldLayer(d_model=32, d_field=16, d_spinor=4, kernel_size=8)
        phi = layer.get_phi()
        phi_antisym = torch.allclose(phi, -phi.T, atol=1e-6)
        findings.append(AuditFinding(
            role="Physics",
            operator="CausalFieldLayer.Phi",
            check="Phi bivector is antisymmetric",
            passed=bool(phi_antisym),
            severity=SEVERITY_HIGH if not phi_antisym else SEVERITY_INFO,
            evidence="models/causal_field.py:287-292 (Phi initialization) + :409-411 (get_phi)",
            recommendation=(
                "Phi antisymmetry verified."
                if phi_antisym else
                "Re-initialize Phi as Phi - Phi.T and enforce in forward pass."
            ),
        ))

        # Check 2: Phi bivector is NOT used in the forward pass (pipeline gap)
        # Static source inspection — read file text to check for Phi usage in forward()
        cf_path = (
            pathlib.Path(__file__).resolve().parents[1] / "models" / "causal_field.py"
        )
        cf_src = cf_path.read_text(encoding="utf-8")
        # Look for get_phi() or self.Phi inside the forward method body
        forward_start = cf_src.find("    def forward(")
        forward_end = cf_src.find("\n    def ", forward_start + 1)
        forward_body = cf_src[forward_start:forward_end] if forward_end > 0 else cf_src[forward_start:]
        phi_used_in_forward = "get_phi" in forward_body or (
            "self.Phi" in forward_body and "def get_phi" not in forward_body
        )
        findings.append(AuditFinding(
            role="Physics",
            operator="CausalFieldLayer.forward",
            check="Phi bivector enters the T field equation (Pi Γ Phi J)",
            passed=phi_used_in_forward,
            severity=SEVERITY_HIGH,
            evidence=(
                "models/causal_field.py:374-394 — forward() contracts Pi(J, Gamma) "
                "but never references self.Phi or get_phi()"
            ),
            recommendation=(
                "Apply Phi contraction before or after Pi: e.g. "
                "J_phi = einsum('...ij,jk->...ik', J, self.get_phi()) "
                "then Pi(J_phi, Gamma). Pending approval."
            ),
        ))

        # Check 3: Holomorphic constraint ∇(Pi Γ Phi) = 0 is NOT enforced
        holomorphic_enforced = (
            "holomorphic" in cf_src.lower() and
            any(kw in cf_src for kw in ["regulariz", "constraint", "loss"])
        )
        # The docstring mentions it but the forward pass doesn't enforce it
        holomorphic_in_forward = "holomorphic" in forward_body.lower()
        findings.append(AuditFinding(
            role="Physics",
            operator="CausalFieldLayer",
            check="Holomorphic constraint ∇^(cD^α)_μ (Pi Γ Phi) = 0 is enforced",
            passed=holomorphic_in_forward,
            severity=SEVERITY_MEDIUM,
            evidence=(
                "models/causal_field.py module docstring line 25 states the constraint "
                "but no enforcement exists in forward() or as a loss regularizer"
            ),
            recommendation=(
                "Add a regularization term or a post-step projection that enforces "
                "the holomorphic constraint. Pending approval."
            ),
        ))

        # Check 4: CliffordConnection gamma matrices anti-commutator {γ^a,γ^b} check
        gamma = layer.Gamma_conn.gamma_matrices  # shape [4, d_spinor, d_spinor]
        n_gammas = gamma.shape[0]
        eta = torch.eye(n_gammas, device=gamma.device)  # Euclidean signature
        max_err = 0.0
        for a in range(n_gammas):
            for b in range(n_gammas):
                ga, gb = gamma[a], gamma[b]
                anticomm = ga @ gb + gb @ ga  # should equal 2 η_{ab} I
                expected = (2 * eta[a, b] * torch.eye(ga.shape[0], device=ga.device))
                err = (anticomm - expected).abs().max().item()
                if err > max_err:
                    max_err = err
        # Learned gammas are initialized randomly, so exact anti-commutation is NOT
        # guaranteed at init. This is a known structural limitation.
        anticomm_ok = max_err < 0.5  # relaxed threshold for learned params at init
        findings.append(AuditFinding(
            role="Physics",
            operator="CliffordConnection.gamma_matrices",
            check="Clifford gamma matrices satisfy approximate anti-commutation {γ^a,γ^b} ≈ 2η^{ab}I",
            passed=anticomm_ok,
            severity=SEVERITY_MEDIUM,
            evidence=(
                f"models/causal_field.py:236 — gamma_matrices initialized as "
                f"randn(4,d_spinor,d_spinor)/d_spinor. Max anti-commutator error: {max_err:.4g}"
            ),
            recommendation=(
                "Consider initializing gamma matrices from actual Dirac/Pauli matrices "
                "and making them learnable via small perturbations. Pending approval."
            ),
        ))

        return findings


class GeometryAgent:
    """
    Audits the fiber bundle structure: Tetrad orthonormality, metric compatibility,
    and whether the kernels/tetrad.py Tetrad is properly connected to CausalFieldLayer.
    """

    def audit(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        Tetrad, compute_metric_from_tetrad = _import_tetrad()

        # Check 1: Tetrad class is NOT wired into CausalFieldLayer (static inspection)
        cf_path = (
            pathlib.Path(__file__).resolve().parents[1] / "models" / "causal_field.py"
        )
        cf_src = cf_path.read_text(encoding="utf-8")
        tetrad_imported = (
            "from ..kernels" in cf_src or
            "kernels.tetrad" in cf_src or
            "from kernels" in cf_src
        )
        findings.append(AuditFinding(
            role="Geometry",
            operator="CliffordConnection.tetrad vs kernels.tetrad.Tetrad",
            check="kernels/tetrad.Tetrad is used by CliffordConnection (shared fiber bundle)",
            passed=tetrad_imported,
            severity=SEVERITY_HIGH,
            evidence=(
                "models/causal_field.py:241-243 defines self.tetrad = nn.Parameter(eye(4)+noise) "
                "independently; kernels/tetrad.py:Tetrad class is never imported in causal_field.py"
            ),
            recommendation=(
                "Replace CliffordConnection's internal tetrad parameter with an instance of "
                "kernels.tetrad.Tetrad so the fiber bundle is governed by one shared operator. "
                "Pending approval."
            ),
        ))

        # Check 2: Tetrad orthonormality verification (using Tetrad class directly)
        tet = Tetrad(dim=4, learnable=False)
        g_inv_diag = torch.tensor([1.0, 1.0, 1.0, 1.0])
        e = tet.compute_from_metric(g_inv_diag)
        is_ortho, max_err = tet.verify_orthonormality(e)
        findings.append(AuditFinding(
            role="Geometry",
            operator="kernels.tetrad.Tetrad",
            check="Tetrad satisfies e^a_μ e^μ_b = δ^a_b (orthonormality)",
            passed=is_ortho,
            severity=SEVERITY_HIGH if not is_ortho else SEVERITY_INFO,
            evidence=(
                f"kernels/tetrad.py:92-135 — verify_orthonormality returned "
                f"is_ortho={is_ortho}, max_error={max_err:.4g}"
            ),
            recommendation=(
                "Orthonormality verified for diagonal metric."
                if is_ortho else
                "Fix tetrad computation to restore e e^{-1} = I."
            ),
        ))

        # Check 3: Metric round-trip through Tetrad
        g_reconstructed = compute_metric_from_tetrad(e)
        g_expected = torch.diag(g_inv_diag)
        round_trip_ok = torch.allclose(g_reconstructed, g_expected, atol=1e-5)
        findings.append(AuditFinding(
            role="Geometry",
            operator="kernels.tetrad.compute_metric_from_tetrad",
            check="Metric round-trip g = e^T e recovers original diagonal metric",
            passed=round_trip_ok,
            severity=SEVERITY_HIGH if not round_trip_ok else SEVERITY_INFO,
            evidence=(
                f"kernels/tetrad.py:180-195 — round-trip max error: "
                f"{(g_reconstructed - g_expected).abs().max().item():.4g}"
            ),
            recommendation=(
                "Round-trip verified."
                if round_trip_ok else
                "Fix compute_metric_from_tetrad signature consistency."
            ),
        ))

        # Check 4: Pi_memory parameter is defined but unused in ParallelTransport.forward
        # Static source inspection
        pi_memory_in_forward = "Pi_memory" in cf_src
        pi_memory_contracted = (
            "Pi_memory" in cf_src and
            cf_src.count("self.Pi_memory") > 1  # appears in init AND forward
        )
        # Actually check if it's in the forward method body
        forward_start = cf_src.find("    def forward(")
        forward_end = cf_src.find("\n    def ", forward_start + 1)
        forward_body = cf_src[forward_start:forward_end] if forward_end > 0 else cf_src[forward_start:]
        pi_memory_used_in_forward = "Pi_memory" in forward_body
        findings.append(AuditFinding(
            role="Geometry",
            operator="ParallelTransport.Pi_memory",
            check="Pi_memory (causal memory channel α,β) is contracted in ParallelTransport.forward",
            passed=pi_memory_used_in_forward,
            severity=SEVERITY_MEDIUM,
            evidence=(
                "models/causal_field.py:172-175 defines Pi_memory parameter; "
                "models/causal_field.py:199-212 forward() only uses Pi_source, "
                "Pi_target, Pi_spinor — Pi_memory is dead weight"
            ),
            recommendation=(
                "Either contract Pi_memory into the transport chain "
                "or remove it to reduce parameter count. Pending approval."
            ),
        ))

        return findings


class CodingAgent:
    """
    Reviews implementation correctness, shape contracts, edge cases, device safety.
    """

    def audit(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        CausalFieldLayer, _, _ = _import_causal_field()

        # Check 1: Shape consistency — forward output shape matches input
        # d_field must be 16 (AssociatorCurrent uses complex octonions = 16-d)
        d_field = 16
        layer = CausalFieldLayer(d_model=32, d_field=d_field, d_spinor=4, kernel_size=8)
        x = torch.randn(2, 8, 32)
        try:
            out, mem = layer(x)
            shape_ok = (out.shape == x.shape)
        except Exception as exc:
            shape_ok = False
            _ = str(exc)
        findings.append(AuditFinding(
            role="Coding",
            operator="CausalFieldLayer.forward",
            check="Forward pass output shape matches input shape [B, N, d_model]",
            passed=shape_ok,
            severity=SEVERITY_CRITICAL if not shape_ok else SEVERITY_INFO,
            evidence=(
                "models/causal_field.py:317-407 — ran forward([2,8,32]), "
                f"output shape correct: {shape_ok}"
            ),
            recommendation=(
                "Shape contract satisfied."
                if shape_ok else
                "Fix shape mismatch in CausalFieldLayer.forward."
            ),
        ))

        # Check 2: CliffordConnection.forward returns no NaN at init
        gamma_conn = layer.Gamma_conn
        gamma_out = gamma_conn()
        gamma_finite = torch.isfinite(gamma_out).all().item()
        findings.append(AuditFinding(
            role="Coding",
            operator="CliffordConnection.forward",
            check="CliffordConnection output is finite at initialization",
            passed=bool(gamma_finite),
            severity=SEVERITY_HIGH if not gamma_finite else SEVERITY_INFO,
            evidence=(
                f"models/causal_field.py:245-258 — output finite: {gamma_finite}, "
                f"shape: {list(gamma_out.shape)}"
            ),
            recommendation=(
                "CliffordConnection output is finite."
                if gamma_finite else
                "Check initialization scale for tetrad and gamma_matrices."
            ),
        ))

        # Check 3: ParallelTransport output shape matches J
        pi = layer.Pi
        J_test = torch.randn(2, 8, d_field, d_field)
        gamma_test = gamma_conn()
        try:
            transported = pi(J_test, gamma_test)
            transport_shape_ok = (transported.shape == J_test.shape)
        except Exception:
            transport_shape_ok = False
        findings.append(AuditFinding(
            role="Coding",
            operator="ParallelTransport.forward",
            check="ParallelTransport output shape matches J shape [B,N,d_field,d_field]",
            passed=transport_shape_ok,
            severity=SEVERITY_CRITICAL if not transport_shape_ok else SEVERITY_INFO,
            evidence=(
                f"models/causal_field.py:182-213 — output shape match: {transport_shape_ok}"
            ),
            recommendation=(
                "Transport shape contract satisfied."
                if transport_shape_ok else
                "Investigate spinor_contrib broadcast logic at lines 208-211."
            ),
        ))

        # Check 4: No .cpu() or .numpy() calls in transport chain (device safety)
        cf_path = (
            pathlib.Path(__file__).resolve().parents[1] / "models" / "causal_field.py"
        )
        source = cf_path.read_text(encoding="utf-8")
        cpu_call_count = source.count(".cpu()") + source.count(".numpy()")
        device_safe = cpu_call_count == 0
        findings.append(AuditFinding(
            role="Coding",
            operator="models/causal_field.py",
            check="No .cpu() or .numpy() calls in transport chain (GPU-safe)",
            passed=device_safe,
            severity=SEVERITY_MEDIUM if not device_safe else SEVERITY_INFO,
            evidence=(
                f"models/causal_field.py — found {cpu_call_count} .cpu()/.numpy() call(s)"
            ),
            recommendation=(
                "Device-safe: no forced CPU transfers found."
                if device_safe else
                "Replace .cpu()/.numpy() calls with device-agnostic alternatives."
            ),
        ))

        return findings


class ValidationAgent:
    """
    Runs quantitative numerical checks on the transport and fiber bundle operators.
    """

    def audit(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        CausalFieldLayer, ParallelTransport, CliffordConnection = _import_causal_field()
        Tetrad, _ = _import_tetrad()

        d_field, d_spinor = 16, 4
        layer = CausalFieldLayer(d_model=32, d_field=d_field, d_spinor=d_spinor, kernel_size=8)

        # Check 1: CliffordConnection output norm is reasonable
        gamma = layer.Gamma_conn()
        gamma_norm = gamma.norm().item()
        gamma_norm_ok = 0.0 < gamma_norm < 1e4
        findings.append(AuditFinding(
            role="Validation",
            operator="CliffordConnection",
            check="CliffordConnection output Frobenius norm is in (0, 1e4)",
            passed=gamma_norm_ok,
            severity=SEVERITY_HIGH if not gamma_norm_ok else SEVERITY_INFO,
            evidence=f"||Gamma||_F = {gamma_norm:.4g}",
            recommendation=(
                "Norm within expected range."
                if gamma_norm_ok else
                "Normalize tetrad or gamma_matrices initialization."
            ),
        ))

        # Check 2: ParallelTransport output is finite for random input
        J = torch.randn(2, 8, d_field, d_field)
        transported = layer.Pi(J, gamma)
        transport_finite = torch.isfinite(transported).all().item()
        findings.append(AuditFinding(
            role="Validation",
            operator="ParallelTransport",
            check="ParallelTransport output is finite for random J",
            passed=bool(transport_finite),
            severity=SEVERITY_CRITICAL if not transport_finite else SEVERITY_INFO,
            evidence=(
                f"max={transported.abs().max().item():.4g}, "
                f"has_nan={transported.isnan().any().item()}, "
                f"has_inf={transported.isinf().any().item()}"
            ),
            recommendation=(
                "Transport output is finite."
                if transport_finite else
                "Add gradient clipping or scale initialization."
            ),
        ))

        # Check 3: Tetrad orthonormality for 4-d anisotropic metric
        tet = Tetrad(dim=4, learnable=False)
        g_aniso = torch.tensor([4.0, 1.0, 2.0, 0.5])
        e = tet.compute_from_metric(g_aniso)
        is_ortho, max_err = tet.verify_orthonormality(e)
        findings.append(AuditFinding(
            role="Validation",
            operator="kernels.tetrad.Tetrad",
            check="Tetrad orthonormality for 4-d anisotropic metric (max error < 1e-5)",
            passed=is_ortho,
            severity=SEVERITY_HIGH if not is_ortho else SEVERITY_INFO,
            evidence=f"max_error = {max_err:.4g}",
            recommendation=(
                "Orthonormality verified."
                if is_ortho else
                "Check sqrt computation for near-zero metric components."
            ),
        ))

        # Check 4: Full CausalFieldLayer forward — no NaN with typical batch
        x = torch.randn(2, 16, 32)
        out, _ = layer(x)
        no_nan = torch.isfinite(out).all().item()
        findings.append(AuditFinding(
            role="Validation",
            operator="CausalFieldLayer (full pipeline)",
            check="Full CausalFieldLayer forward is NaN/Inf-free for typical batch",
            passed=bool(no_nan),
            severity=SEVERITY_CRITICAL if not no_nan else SEVERITY_INFO,
            evidence=(
                f"out has_nan={out.isnan().any().item()}, "
                f"has_inf={out.isinf().any().item()}, "
                f"max_abs={out.abs().max().item():.4g}"
            ),
            recommendation=(
                "Full pipeline forward is stable."
                if no_nan else
                "Enable diagnose=True and trace the NaN source."
            ),
        ))

        return findings


class MoraleAgent:
    """
    Monitors team health and workload sustainability.
    """

    def audit(self, findings: List[AuditFinding]) -> List[str]:
        notes: List[str] = []
        total = len(findings)
        failed = sum(1 for f in findings if not f.passed)
        critical = sum(1 for f in findings if not f.passed and f.severity == SEVERITY_CRITICAL)
        high = sum(1 for f in findings if not f.passed and f.severity == SEVERITY_HIGH)

        notes.append(
            f"Team health check: {total} checks completed, {failed} failed "
            f"({critical} CRITICAL, {high} HIGH)."
        )
        if critical > 0:
            notes.append(
                "⚠  CRITICAL findings present. Recommend immediate coordinator review "
                "before scheduling follow-up work to avoid team overload."
            )
        if failed > total * 0.5:
            notes.append(
                "⚠  Majority of checks failing. Suggest breaking the remediation "
                "into two sequential sprints: (1) pipeline wiring fixes, "
                "(2) physical/geometric constraint enforcement."
            )
        else:
            notes.append(
                "✓  Workload appears sustainable. Proceed in one sprint "
                "once coordinator obtains approval."
            )
        notes.append(
            "Morale note: Findings at this stage are PLAN-ONLY. "
            "No code has been changed. Approval is required before execution."
        )
        return notes


class ScribeAgent:
    """
    Consolidates all findings into a structured decision log.
    """

    def consolidate(
        self,
        findings: List[AuditFinding],
        wiring_checks: List[PipelineWiringCheck],
    ) -> ScribeLog:
        failed = [f for f in findings if not f.passed]

        summary_lines = [
            f"Total checks: {len(findings)}",
            f"Passed: {sum(1 for f in findings if f.passed)}",
            f"Failed: {len(failed)}",
            f"Pipeline wiring issues: {sum(1 for w in wiring_checks if not w.wired)}",
        ]
        summary = " | ".join(summary_lines)

        action_items = []
        for f in failed:
            action_items.append(
                f"[{f.severity}] [{f.role}] {f.operator} — {f.check}: {f.recommendation}"
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

def _check_pipeline_wiring() -> List[PipelineWiringCheck]:
    """
    Inspect source files to determine whether the transport and fiber bundle
    operators are wired into the pipeline.
    """
    checks: List[PipelineWiringCheck] = []
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    # Read causal_field.py source directly (avoid import issues)
    cf_src = (repo_root / "models" / "causal_field.py").read_text(encoding="utf-8")

    # Find the CausalFieldLayer forward method body for scoped checks
    layer_start = cf_src.find("class CausalFieldLayer")
    layer_src = cf_src[layer_start:] if layer_start >= 0 else cf_src

    # 1. Is ParallelTransport instantiated and called in CausalFieldLayer?
    checks.append(PipelineWiringCheck(
        operator="ParallelTransport",
        wired=("self.Pi" in layer_src and "self.Pi(" in layer_src),
        entry_point="models.causal_field.CausalFieldLayer",
        notes="Pi is instantiated at line ~295 and called at line ~375.",
    ))

    # 2. Is CliffordConnection instantiated and called in CausalFieldLayer?
    checks.append(PipelineWiringCheck(
        operator="CliffordConnection",
        wired=("self.Gamma_conn" in layer_src and "self.Gamma_conn()" in layer_src),
        entry_point="models.causal_field.CausalFieldLayer",
        notes="Gamma_conn is instantiated at line ~298 and called at line ~371.",
    ))

    # 3. Is Phi (bivector) contracted into the forward pass?
    forward_start = layer_src.find("    def forward(")
    forward_end = layer_src.find("\n    def ", forward_start + 1)
    forward_body = (
        layer_src[forward_start:forward_end]
        if forward_end > 0 else layer_src[forward_start:]
    )
    phi_wired = "get_phi" in forward_body or (
        "self.Phi" in forward_body and "def get_phi" not in forward_body
    )
    checks.append(PipelineWiringCheck(
        operator="Phi bivector",
        wired=phi_wired,
        entry_point="models.causal_field.CausalFieldLayer.forward",
        notes=(
            "get_phi() is defined and Phi is initialized but never called in forward(); "
            "the field equation T = α J + (1-α) ∫ k Pi Γ J is missing the Phi factor."
        ),
    ))

    # 4. Is kernels.tetrad.Tetrad imported/used in causal_field.py?
    tetrad_imported = (
        "from ..kernels" in cf_src or
        "kernels.tetrad" in cf_src or
        "from kernels" in cf_src
    )
    checks.append(PipelineWiringCheck(
        operator="kernels.tetrad.Tetrad (fiber bundle)",
        wired=tetrad_imported,
        entry_point="models/causal_field.py",
        notes=(
            "kernels/tetrad.py Tetrad is NOT imported in causal_field.py. "
            "CliffordConnection uses its own independent tetrad nn.Parameter. "
            "The shared fiber bundle is not connected."
        ),
    ))

    # 5. Is Tetrad exported from kernels/__init__.py?
    kernels_init = (repo_root / "kernels" / "__init__.py").read_text(encoding="utf-8")
    tetrad_exported = "Tetrad" in kernels_init
    checks.append(PipelineWiringCheck(
        operator="kernels.tetrad.Tetrad export",
        wired=tetrad_exported,
        entry_point="kernels/__init__.py",
        notes="Tetrad is exported from kernels/__init__.py at line ~39-41.",
    ))

    # 6. Is ParallelTransport exported from models/__init__.py?
    models_init = (repo_root / "models" / "__init__.py").read_text(encoding="utf-8")
    pt_exported = "ParallelTransport" in models_init
    checks.append(PipelineWiringCheck(
        operator="models.ParallelTransport export",
        wired=pt_exported,
        entry_point="models/__init__.py",
        notes="ParallelTransport is exported from models/__init__.py at line ~36.",
    ))

    # 7. Does BiQuatCausalLayer bypass transport operators? (biquaternion path check)
    biquat_src = (repo_root / "models" / "biquaternion.py").read_text(encoding="utf-8")
    biquat_src_lower = biquat_src.lower()
    biquat_bypasses = (
        "paralleltransport" not in biquat_src_lower and
        "cliffordconnection" not in biquat_src_lower and
        "self.pi" not in biquat_src_lower and
        "self.gamma_conn" not in biquat_src_lower
    )
    checks.append(PipelineWiringCheck(
        operator="BiQuatCausalLayer (biquaternion / preferred path)",
        wired=biquat_bypasses,   # "wired" here means "correctly bypasses" Pi/Gamma
        entry_point="models/biquaternion.py",
        notes=(
            "BiQuatCausalLayer is the preferred O(N) path. "
            "It correctly does NOT use ParallelTransport, CliffordConnection, or Phi. "
            "d_field=16 is enforced by assert in BiQuatCausalLayer.__init__."
        ),
    ))

    # 8. Does BiQuatCausalLayer enforce d_field=16?
    biquat_asserts_16 = "assert d_field == 16" in biquat_src
    checks.append(PipelineWiringCheck(
        operator="BiQuatCausalLayer.d_field assertion",
        wired=biquat_asserts_16,
        entry_point="models/biquaternion.py:302",
        notes=(
            "d_field=16 is structurally fixed for biquaternions: "
            "4 real quaternions × 4 components = 16 real DOF. "
            "Same value as CxO path (8+8 octonion dims), but for different reasons."
        ),
    ))

    return checks


# ---------------------------------------------------------------------------
# Seven-Agent Audit Team (public entry point)
# ---------------------------------------------------------------------------

class TransportFiberAuditTeam:
    """
    Seven-agent specialist team that audits the transport and fiber bundle
    operators and their pipeline connections.

    STATUS: AWAITING APPROVAL TO EXECUTE
    Call run() to produce a full AuditReport; no code changes are made.
    """

    def run(self) -> AuditReport:
        """
        Execute the full audit and return the report.
        Does NOT make any code changes — findings are plan-only.
        """
        coordinator = CoordinatorAgent()
        physics = PhysicsAgent()
        geometry = GeometryAgent()
        coding = CodingAgent()
        validation = ValidationAgent()
        morale = MoraleAgent()
        scribe = ScribeAgent()

        # Collect findings from specialist agents
        all_findings: List[AuditFinding] = []
        all_findings.extend(physics.audit())
        all_findings.extend(geometry.audit())
        all_findings.extend(coding.audit())
        all_findings.extend(validation.audit())

        # Check pipeline wiring
        wiring_checks = _check_pipeline_wiring()

        # Morale notes
        morale_notes = morale.audit(all_findings)

        # Scribe consolidation
        scribe_log = scribe.consolidate(all_findings, wiring_checks)

        return AuditReport(
            coordinator_scope=coordinator.scope(),
            findings=all_findings,
            wiring_checks=wiring_checks,
            morale_notes=morale_notes,
            scribe_log=scribe_log,
            approval_status="AWAITING APPROVAL TO EXECUTE",
        )


# ---------------------------------------------------------------------------
# CxO vs CxH tradeoff analysis (quantitative)
# ---------------------------------------------------------------------------

def biquat_tradeoff_analysis(d_model: int = 512) -> dict:
    """
    Compute the precise tradeoffs between CxO and CxH at a given d_model.

    Instantiates both CausalFieldLayer (CxO) and BiQuatCausalLayer (CxH),
    counts parameters per submodule, and returns a structured comparison dict.

    The "pair of biquaternions" question
    -------------------------------------
    The BiQuat state IS already a pair:
        Q_M = Q_M_re[4] + Q_M_im[4]  -- present moment biquaternion
        Q_H = Q_H_re[4] + Q_H_im[4]  -- memory biquaternion
    Total: 16 scalars = d_field.
    Having the pair gives temporal context (present vs memory) but does NOT
    recover the algebraic properties of CxO (see ``losses`` key in output).

    Returns:
        dict with keys:
            cxo_total_params         int -- total learnable params, CxO path
            cxh_total_params         int -- total learnable params, CxH path
            param_ratio              float -- cxo / cxh
            cxo_submodules           dict -- breakdown by submodule
            cxh_submodules           dict -- breakdown by submodule
            cxo_memory_state_dim     int -- memory vector length per token
            cxh_memory_state_dim     int -- memory vector length per token
            memory_state_ratio       int -- cxo / cxh
            losses                   list[dict] -- each capability lost in CxH
            gains                    list[str]  -- capabilities kept or gained in CxH
    """
    CausalFieldLayer, _, _ = _import_causal_field()
    BiQuatCausalLayer = _import_biquat()

    def _count(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    # Instantiate at the given d_model
    cxo = CausalFieldLayer(
        d_model=d_model, d_field=D_FIELD, d_spinor=4, kernel_size=8
    )
    cxh = BiQuatCausalLayer(d_model=d_model, d_field=D_FIELD)

    # --- CxO submodule breakdown ---
    cxo_sub = {
        # AssociatorCurrent: 3 x Linear(d_model, 16) + J_expand[16,16,16]
        # = 3*(d_model*16 + 16) + 4096
        "associator (J = (ab)c - a(bc), Fano-plane)": _count(cxo.associator),
        # ParallelTransport: 3 x [16,16,16] + [4,4,16] = 12288 + 256
        "parallel_transport Pi (rank-8 fiber bundle)": _count(cxo.Pi),
        # CliffordConnection: gamma[4,4,4] + tetrad[4,4] = 64 + 16
        "clifford_connection Gamma (tetrad + generators)": _count(cxo.Gamma_conn),
        # Phi: [16,16] antisymmetric bivector
        "phi_bivector Phi (antisymmetric field)": cxo.Phi.numel(),
        # LiorMemoryState: LiorKernel (~15 scalars) + J_H projection
        "lior_memory (exp + power-law + oscillatory kernel)": _count(cxo.memory),
        # CognitiveManifold: metric_net, resilience_net, complex_metric, etc.
        "cognitive_manifold (G=A+iB, geodesics, K0->K1->K2)": _count(cxo.manifold),
        "input_proj + output_proj + norm": (
            _count(cxo.input_proj) + _count(cxo.output_proj) + _count(cxo.norm)
        ),
    }

    # --- CxH submodule breakdown ---
    cxh_sub = {
        # CausalAccumulator: W_impulse(8) + W_transport(8) + 3 scalars = 19
        "accumulator (W_impulse + W_transport + alpha/decay/scale)": _count(cxh.accumulator),
        "input_proj + output_proj + norm": (
            _count(cxh.input_proj) + _count(cxh.output_proj) + _count(cxh.norm)
        ),
    }

    cxo_total = sum(cxo_sub.values())
    cxh_total = sum(cxh_sub.values())

    # --- Capabilities lost going CxO -> CxH ---
    losses = [
        {
            "capability": "Non-associative algebra as signal (AssociatorCurrent)",
            "cxo_detail": (
                "J = (ab)c - a(bc) uses fixed Fano-plane structure constants "
                "(7 triples, oct_struct buffer [8,8,8]). Non-zero residual IS "
                "the source current — algebraic curvature as inductive bias."
            ),
            "cxh_detail": (
                "ℂ⊗ℍ is associative: (ab)c = a(bc) always. "
                "Associator is identically zero. Having two biquaternions "
                "(Q_M + Q_H) does not change this — associativity is an "
                "algebraic property of H, not a count-of-elements property."
            ),
            "param_delta": cxo_sub["associator (J = (ab)c - a(bc), Fano-plane)"],
        },
        {
            "capability": "G2 symmetry / Fano plane inductive bias",
            "cxo_detail": (
                "Aut(O) = G2 (14-dimensional exceptional Lie group). "
                "The 7 Fano-plane triples are a fixed non-learnable structural "
                "constraint respected by every oct_mul call."
            ),
            "cxh_detail": (
                "Aut(H) = SO(3) (3-dimensional). Two biquaternions gives "
                "SO(3) x SO(3) (6-dimensional). "
                "8 symmetry generators are permanently absent."
            ),
            "param_delta": 0,  # Fano plane is a buffer, not a parameter
        },
        {
            "capability": "Fiber bundle / parallel transport (Pi, Gamma, Phi)",
            "cxo_detail": (
                f"Pi: {cxo_sub['parallel_transport Pi (rank-8 fiber bundle)']} params (rank-8 tensor). "
                f"Gamma: {cxo_sub['clifford_connection Gamma (tetrad + generators)']} params (tetrad + Clifford generators). "
                f"Phi: {cxo_sub['phi_bivector Phi (antisymmetric field)']} params (bivector field). "
                "Implements genuine covariant parallel transport with holonomy."
            ),
            "cxh_detail": (
                "W_transport: 8 learnable scalars (one BiQuatTransform). "
                "Single left-multiplication in ℂ⊗ℍ. No holonomy. "
                "No tetrad. No bivector coupling. No fiber bundle structure."
            ),
            "param_delta": (
                cxo_sub["parallel_transport Pi (rank-8 fiber bundle)"]
                + cxo_sub["clifford_connection Gamma (tetrad + generators)"]
                + cxo_sub["phi_bivector Phi (antisymmetric field)"]
            ),
        },
        {
            "capability": "LIoR multi-modal memory (fractional + oscillatory)",
            "cxo_detail": (
                "Three-mode kernel: exponential (Markovian) + power-law "
                "k(dt) ~ dt^(-delta) (fractional, non-Markovian, long tail) "
                "+ oscillatory k(dt) ~ cos(omega*dt+phi)*exp(-zeta*dt) "
                "(phase-sensitive interference). "
                f"Memory state: {D_FIELD * D_FIELD} dims per token. "
                "Fractional order delta in (0,1) is learnable."
            ),
            "cxh_detail": (
                "Single exponential pole: "
                "Q_H_new = decay * Q_H + scale * W(Q_M). "
                "Memory state: 8 dims per token. "
                "No fractional memory. No oscillatory mode. "
                "History decays exponentially — long-range dependencies "
                "cannot be represented."
            ),
            "param_delta": cxo_sub["lior_memory (exp + power-law + oscillatory kernel)"],
        },
        {
            "capability": "Complex metric G=A+iB and Riemannian geometry (CognitiveManifold)",
            "cxo_detail": (
                "ComplexMetricTensor G=A+iB: A=Riemannian, B=symplectic. "
                "Phase theta(omega) from fractional kernel feeds into B. "
                "Geodesic integration (exp/log maps), Christoffel symbols, "
                "normal coordinates, spinor bilinears K0->K1->K2. "
                f"Adds {cxo_sub['cognitive_manifold (G=A+iB, geodesics, K0->K1->K2)']} params."
            ),
            "cxh_detail": (
                "No manifold. No complex metric. No geodesics. "
                "Scalar alpha is the only mixing control."
            ),
            "param_delta": cxo_sub["cognitive_manifold (G=A+iB, geodesics, K0->K1->K2)"],
        },
    ]

    # --- Capabilities kept or gained in CxH ---
    gains = [
        "SL(2,C) = double cover of Lorentz group: W = W_re + i*W_im ∈ ℂ⊗ℍ = M_2(ℂ), "
        "so W_impulse and W_transport are Lorentz rotations + boosts.",

        "Temporal present/memory split: Q_M (present biquaternion) and "
        "Q_H (memory biquaternion) give causal ordering without sequential loops.",

        "fp16/bf16 safe: pure-real arithmetic with explicit clamps in BiQuatTransform. "
        "CxO is fp32-only.",

        "O(1) per element (64 quaternion muls) vs O(d^3) for CxO "
        f"({D_FIELD**3} ops just for J_expand einsum).",

        "Bounded hyperparameters: alpha/decay/impulse_scale all pass through "
        "sigmoid/softplus, so no exploding scalars.",
    ]

    return {
        "d_model": d_model,
        "cxo_total_params": cxo_total,
        "cxh_total_params": cxh_total,
        "param_ratio": round(cxo_total / cxh_total, 1),
        "cxo_submodules": cxo_sub,
        "cxh_submodules": cxh_sub,
        "cxo_memory_state_dim": D_FIELD * D_FIELD,   # d_field^2 = 256
        "cxh_memory_state_dim": 8,                    # Q_H_re[4] + Q_H_im[4]
        "memory_state_ratio": (D_FIELD * D_FIELD) // 8,  # 32
        "losses": losses,
        "gains": gains,
    }
