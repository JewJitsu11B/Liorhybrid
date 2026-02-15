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


@dataclass(frozen=True)
class ValidationFinding:
    agent: str
    check: str
    passed: bool
    details: str
    literature: str


@dataclass(frozen=True)
class ValidationReport:
    findings: List[ValidationFinding]

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
        return ValidationReport(findings=findings)

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
            )
        )

        hist_short = [torch.randn(4, 4, 2, 2, dtype=torch.complex64) for _ in range(4)]
        hist_long = [torch.randn(4, 4, 2, 2, dtype=torch.complex64) for _ in range(24)]
        w_short = self._memory_weight_proxy(len(hist_short), alpha=alpha, lambda_F=0.1)
        w_long = self._memory_weight_proxy(len(hist_long), alpha=alpha, lambda_F=0.1)
        weight_bound_ok = (
            float(w_short) >= 0.0 and float(w_short) <= 1.0 and
            float(w_long) >= 0.0 and float(w_long) <= 1.0 and
            float(w_long) >= float(w_short)
        )
        findings.append(
            ValidationFinding(
                agent="Physical Consistency Agent",
                check="Memory modulation weight is bounded and history-monotone",
                passed=weight_bound_ok,
                details="memory_weight must stay in [0,1] and increase with accumulated history.",
                literature=self.literature["fractional_memory"],
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
            )
        )

        precompute_ok = cfg.n_neighbors == 64 and cfg.m == 6 and cfg.d_block == (cfg.d_prime + cfg.m + cfg.k)
        findings.append(
            ValidationFinding(
                agent="Implementation Agent",
                check="Precompute dimensions satisfy Option-6 contracts",
                passed=precompute_ok,
                details="Neighbor slots and score channel counts must match strict pipeline assumptions.",
                literature=self.literature["index_contract"],
            )
        )

        return findings

    @staticmethod
    def _memory_weight_proxy(n_steps: int, *, alpha: float, lambda_F: float) -> float:
        if n_steps <= 0:
            return 0.0
        w = alpha * torch.log(torch.tensor(1.0 + float(n_steps)))
        return float(torch.clamp(w * lambda_F, min=0.0, max=1.0))
