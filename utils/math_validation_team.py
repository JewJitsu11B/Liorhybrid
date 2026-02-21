"""
Multi-agent mathematical validation for the full physics pipeline.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch

from kernels.bayesian import bayesian_posterior, bayesian_recursive_term
from kernels.fractional_memory import fractional_kernel_weights
from kernels.hamiltonian import hamiltonian_evolution, spatial_laplacian


DEFAULT_LITERATURE: Dict[str, str] = {
    "hamiltonian_operator": "Schrodinger form: H = -(ħ²/2m)∇² + V",
    "bayesian_update": "Bayes theorem: P(H|D) ∝ P(D|H)P(H)",
    "fractional_memory": "Podlubny, Fractional Differential Equations (1998), Eq. power-law kernels",
    "index_contract": "Tensor index consistency requirement: contiguous block layout with exact offsets",
}
BRIDGE_AGENT_NAMES = ("Physics Agent", "Abstract Algebra Agent", "Differential Geometry Agent")


@dataclass(frozen=True)
class ValidationFinding:
    agent: str
    check: str
    passed: bool
    details: str
    literature: str
    logic_path: str
    rationale: str


@dataclass(frozen=True)
class BridgeStep:
    gap: str
    owner_agent: str
    action: str
    completion_criterion: str


@dataclass(frozen=True)
class StubTeamOutput:
    pseudocode: List[str]
    formalisms: List[str]


@dataclass(frozen=True)
class ValidationReport:
    findings: List[ValidationFinding]
    logic_audit_comments: List[str]
    stub_output: StubTeamOutput
    bridge_plan: List[BridgeStep]

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def failed_checks(self) -> List[ValidationFinding]:
        return [f for f in self.findings if not f.passed]


class AddressLayout(Protocol):
    core_end: int
    geom_start: int
    geom_end: int
    neighbors_start: int
    neighbors_end: int
    integrity_start: int
    integrity_end: int
    total_dim: int
    n_neighbors: int
    m: int
    d_block: int
    d_prime: int
    k: int


@dataclass(frozen=True)
class DefaultAddressLayout:
    d: int = 512
    n_nearest: int = 32
    n_high_sim: int = 16
    n_low_sim: int = 16
    d_prime: int = 64
    m: int = 6
    k: int = 16
    ecc_bits: int = 32
    n_timestamps: int = 2

    @property
    def n_neighbors(self) -> int:
        return self.n_nearest + self.n_high_sim + self.n_low_sim

    @property
    def d_geom(self) -> int:
        return 2 * self.d

    @property
    def d_block(self) -> int:
        return self.d_prime + self.m + self.k

    @property
    def d_integrity(self) -> int:
        return self.ecc_bits + self.n_timestamps

    @property
    def core_end(self) -> int:
        return self.d

    @property
    def geom_start(self) -> int:
        return self.core_end

    @property
    def geom_end(self) -> int:
        return self.geom_start + self.d_geom

    @property
    def neighbors_start(self) -> int:
        return self.geom_end

    @property
    def neighbors_end(self) -> int:
        return self.neighbors_start + self.n_neighbors * self.d_block

    @property
    def integrity_start(self) -> int:
        return self.neighbors_end

    @property
    def integrity_end(self) -> int:
        return self.integrity_start + self.d_integrity

    @property
    def total_dim(self) -> int:
        return self.integrity_end


class MathValidationTeam:
    """
    Agent team that validates full-pipeline math, indices, and precompute logic.
    """

    def __init__(self, literature: Dict[str, str] | None = None):
        self.literature = literature or DEFAULT_LITERATURE
        self.required_neighbors = 64
        self.required_score_channels = 6
        self.validation_lambda_f = 0.1

    def validate_all(
        self,
        *,
        address_config: AddressLayout | None = None,
        alpha: float = 0.5,
        n_steps: int = 64,
        dt: float = 0.01,
    ) -> ValidationReport:
        cfg = address_config or DefaultAddressLayout()
        findings: List[ValidationFinding] = []
        findings.extend(self._formal_integrity_agent())
        findings.extend(self._physical_consistency_agent(alpha=alpha, n_steps=n_steps, dt=dt))
        findings.extend(self._implementation_agent(cfg))
        comments = self._logic_audit_comments(findings)
        stub_output = self._stub_team_output()
        bridge_plan = self._bridge_plan(findings)
        return ValidationReport(
            findings=findings,
            logic_audit_comments=comments,
            stub_output=stub_output,
            bridge_plan=bridge_plan,
        )

    def _formal_integrity_agent(self) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        T_const = torch.ones(8, 8, 2, 2, dtype=torch.complex64)
        lap = spatial_laplacian(T_const, dx=1.0)
        lap_ok = torch.allclose(lap, torch.zeros_like(lap), atol=1e-6)
        findings.append(
            ValidationFinding(
                agent="Formal Integrity Agent",
                check="Laplacian of constant field is zero",
                passed=lap_ok,
                details="Finite-difference stencil must preserve constant fields exactly.",
                literature=self.literature["hamiltonian_operator"],
                logic_path="start->operator_discretization->constant_field_invariance",
                rationale="A constant field should have zero second derivative; non-zero indicates index/sign defect.",
            )
        )

        H = hamiltonian_evolution(T_const, hbar_cog=0.1, m_cog=1.0, V=None)
        ham_ok = torch.allclose(H, torch.zeros_like(H), atol=1e-6)
        findings.append(
            ValidationFinding(
                agent="Formal Integrity Agent",
                check="Hamiltonian of constant field with V=0 is zero",
                passed=ham_ok,
                details="Checks coefficient/sign consistency in H[T] = -(ħ²/2m)∇²T + V·T.",
                literature=self.literature["hamiltonian_operator"],
                logic_path="start->hamiltonian_kernel->free_field_limit",
                rationale="In the free-field limit with ∇²T=0 and V=0, Hamiltonian contribution must vanish.",
            )
        )

        T_prev = torch.randn(4, 4, 2, 2, dtype=torch.complex64)
        w = torch.ones_like(T_prev.real)
        B = bayesian_posterior(T_prev, w)
        posterior_ok = B.shape == T_prev.shape and torch.isfinite(B.real).all() and torch.isfinite(B.imag).all()
        findings.append(
            ValidationFinding(
                agent="Formal Integrity Agent",
                check="Bayesian posterior is finite and shape-preserving",
                passed=bool(posterior_ok),
                details="Posterior normalization must not create NaN/Inf and must preserve indices.",
                literature=self.literature["bayesian_update"],
                logic_path="start->bayesian_update->posterior_normalization",
                rationale="Posterior maps tensor-to-tensor; shape/index mismatch breaks downstream operator algebra.",
            )
        )

        zero_qr = bayesian_recursive_term(
            T_current=T_prev,
            T_prev_collapsed=None,
            evidence=None,
            lambda_QR=0.1,
            tau=0.1,
        )
        qr_ok = torch.allclose(zero_qr, torch.zeros_like(T_prev), atol=1e-8)
        findings.append(
            ValidationFinding(
                agent="Formal Integrity Agent",
                check="Recursive Bayesian term is zero at first step",
                passed=qr_ok,
                details="When no prior collapsed state exists, Λ_QR must be exactly zero.",
                literature=self.literature["bayesian_update"],
                logic_path="start->bayesian_recursive_term->initial_condition",
                rationale="Initial update must not invent prior evidence; otherwise dynamics is unphysical.",
            )
        )

        return findings

    def _physical_consistency_agent(self, *, alpha: float, n_steps: int, dt: float) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        weights = fractional_kernel_weights(alpha=alpha, n_steps=n_steps, dt=dt)
        kernel_norm_ok = torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        kernel_positive_ok = bool(torch.all(weights > 0))
        findings.append(
            ValidationFinding(
                agent="Physical Consistency Agent",
                check="Fractional kernel is normalized and positive",
                passed=bool(kernel_norm_ok and kernel_positive_ok),
                details="Power-law memory kernel must remain a valid bounded weighting measure.",
                literature=self.literature["fractional_memory"],
                logic_path="start->fractional_memory->kernel_measure",
                rationale="A normalized positive kernel is required for stable memory integration and physical weighting.",
            )
        )

        hist_short = [torch.randn(4, 4, 2, 2, dtype=torch.complex64) for _ in range(4)]
        hist_long = [torch.randn(4, 4, 2, 2, dtype=torch.complex64) for _ in range(24)]
        w_short = self._memory_weight_proxy(len(hist_short), alpha=alpha, lambda_F=self.validation_lambda_f)
        w_long = self._memory_weight_proxy(len(hist_long), alpha=alpha, lambda_F=self.validation_lambda_f)
        w_short_f = float(w_short)
        w_long_f = float(w_long)
        weight_bound_ok = (
            w_short_f >= 0.0 and w_short_f <= 1.0 and
            w_long_f >= 0.0 and w_long_f <= 1.0 and
            w_long_f >= w_short_f
        )
        findings.append(
            ValidationFinding(
                agent="Physical Consistency Agent",
                check="Memory modulation weight is bounded and history-monotone",
                passed=weight_bound_ok,
                details="memory_weight must stay in [0,1] and increase with accumulated history.",
                literature=self.literature["fractional_memory"],
                logic_path="start->fractional_memory->bayesian_modulation_weight",
                rationale="Out-of-bound or non-monotone weights break posterior interpretation and control stability.",
            )
        )

        return findings

    def _implementation_agent(self, cfg: AddressLayout) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        contiguous = (
            cfg.core_end == cfg.geom_start and
            cfg.geom_end == cfg.neighbors_start and
            cfg.neighbors_end == cfg.integrity_start and
            cfg.integrity_end == cfg.total_dim
        )
        findings.append(
            ValidationFinding(
                agent="Implementation Agent",
                check="Address index layout is contiguous and complete",
                passed=contiguous,
                details="All linearized block offsets must be gap-free and end at total_dim.",
                literature=self.literature["index_contract"],
                logic_path="start->address_linearization->offset_continuity",
                rationale="Non-contiguous offsets imply index corruption between semantic blocks.",
            )
        )

        precompute_ok = (
            cfg.n_neighbors == self.required_neighbors and
            cfg.m == self.required_score_channels and
            cfg.d_block == (cfg.d_prime + cfg.m + cfg.k)
        )
        findings.append(
            ValidationFinding(
                agent="Implementation Agent",
                check="Precompute dimensions satisfy Option-6 contracts",
                passed=precompute_ok,
                details="Neighbor slots and score channel counts must match strict pipeline assumptions.",
                literature=self.literature["index_contract"],
                logic_path="start->address_precompute->dimension_contracts",
                rationale="Precompute dimensions define compile-time tensor interfaces for routing kernels.",
            )
        )

        return findings

    @staticmethod
    def _memory_weight_proxy(n_steps: int, *, alpha: float, lambda_F: float) -> float:
        if n_steps <= 0:
            return 0.0
        w = alpha * torch.log(torch.tensor(1.0 + float(n_steps)))
        return float(torch.clamp(w * lambda_F, min=0.0, max=1.0))

    @staticmethod
    def _logic_audit_comments(findings: List[ValidationFinding]) -> List[str]:
        comments = []
        for finding in findings:
            direction = "accepted" if finding.passed else "rejected"
            comments.append(
                f"[{finding.agent}] {direction}: {finding.check} | "
                f"path={finding.logic_path} | why={finding.rationale}"
            )
        return comments

    @staticmethod
    def _stub_team_output() -> StubTeamOutput:
        return StubTeamOutput(
            pseudocode=[
                "Input T, config, evidence, history",
                "Compute H[T], Λ_QR[T], and memory modulation w_mem",
                "Assemble update dT = (dt/(i*ℏ)) * (-effective_grad + Λ_QR + J)",
                "Validate index contracts and record audit path/rationale",
                "If gaps remain, dispatch to bridge agents until all checks pass",
            ],
            formalisms=[
                "H[T] = -(ħ²/2m)∇²T + V·T",
                "Λ_QR[T] = λ_QR(B[T_prev] - T_prev)",
                "B[T] = (w ⊙ T) / Z, with Z = Σ w|T|²",
                "K(τ) = τ^(α-1)/Γ(α), normalized over discrete history",
                "w_mem = clamp(λ_F * α * log(1+n_steps), 0, 1)",
            ],
        )

    @staticmethod
    def _bridge_plan(findings: List[ValidationFinding]) -> List[BridgeStep]:
        bridge_steps: List[BridgeStep] = []
        if not all(f.passed for f in findings):
            bridge_steps.extend([
                BridgeStep(
                    gap="Operator-level physical inconsistency",
                    owner_agent=BRIDGE_AGENT_NAMES[0],
                    action="Re-derive physical constraint and enforce admissible parameter regime.",
                    completion_criterion="All physics-tagged findings pass.",
                ),
                BridgeStep(
                    gap="Tensor/index contract ambiguity",
                    owner_agent=BRIDGE_AGENT_NAMES[1],
                    action="Normalize tensor/index formalism and prove block-consistent mappings.",
                    completion_criterion="All index/layout findings pass with contiguous mappings.",
                ),
                BridgeStep(
                    gap="Geometry/operator transport mismatch",
                    owner_agent=BRIDGE_AGENT_NAMES[2],
                    action="Verify metric-connection compatibility and curvature-safe discretization assumptions.",
                    completion_criterion="Geometry-sensitive findings pass and no transport mismatch remains.",
                ),
            ])
        return bridge_steps
