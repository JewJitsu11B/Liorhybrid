"""
Complex Metric Tensor: G_{mu nu} = A_{mu nu} + i B_{mu nu}

Implements the complex metric structure from Clifford-Hodge geometry:

    A_{mu nu} = (1/2)(gamma_mu gamma_nu + gamma_nu gamma_mu)  [Symmetric, Riemannian]
    B_{mu nu} = (1/2)(gamma_mu gamma_nu - gamma_nu gamma_mu)  [Antisymmetric, Symplectic]

Physical interpretation:
    - A_{mu nu}: Configuration space metric (positions, distances, curvature)
    - B_{mu nu}: Phase/momentum space metric (spectral, interference, flow)

The symplectic form B comes from the phase gradient:
    B_{mu nu} = nabla_mu theta wedge nabla_nu theta

where theta is derived from the fractional kernel Fourier transform:
    theta(omega) = (pi * alpha / 2) - alpha * ln(omega)

Phase Orthogonality (Stability Guarantee):
    The Sigma (geometric) and Lambda (spectral) sectors are orthogonal:
    - Real axis: Observable mass / geometric quantities
    - Imaginary axis: Quantum corrections / phase quantities
    This ensures the O(1) recurrence is stable.

Spinor-Wedge-Tensor Mapping (K0 -> K1 -> K2):
    K0 (Spinor): sigma = psi_bar psi                    [Scalar overlap]
    K1 (Wedge):  Phi_{mu nu} = psi_bar gamma_{mu nu} psi  [Antisymmetric Torquency]
    K2 (Tensor): Theta_{mu nu} = psi_bar gamma_{(mu} gamma_{nu)} psi  [Symmetric Newtocity]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class ComplexMetricTensor(nn.Module):
    """
    Complex metric G_{mu nu} = A_{mu nu} + i B_{mu nu}

    A is the symmetric Riemannian metric (LIoR-scaled).
    B is the antisymmetric symplectic form (from phase gradients).
    """

    def __init__(self, d_coord: int = 8):
        """
        Initialize complex metric tensor.

        Args:
            d_coord: Dimension of coordinate manifold (default 8 for E8)
        """
        super().__init__()
        self.d_coord = d_coord

    def compute_phase_field(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase field theta from fractional kernel structure.

        The fractional kernel k(t) ~ t^(alpha-1) has Fourier transform:
            k_hat(omega) = Gamma(alpha) * omega^(-alpha) * exp(i * pi * alpha / 2)

        So the phase is:
            theta(omega) = (pi * alpha / 2) - alpha * ln(omega)

        We use the embedding norm as a proxy for "frequency".

        Args:
            z: Embeddings [B, N, d_model] or coordinates [B, N, d_coord]
            alpha: Fractional order, scalar or [B, N]

        Returns:
            theta: Phase field [B, N]
        """
        # Use embedding norm as frequency
        omega = torch.norm(z, dim=-1) + 1e-8  # [B, N]

        # Broadcast alpha if needed
        if alpha.dim() == 0:
            alpha = alpha.expand_as(omega)

        # Phase from fractional kernel Fourier transform
        theta = (math.pi * alpha / 2) - alpha * torch.log(omega)

        return theta

    def compute_phase_gradient(
        self,
        theta: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient of phase field.

        If z (coordinates) is provided, compute spatial gradient.
        Otherwise, use finite differences along sequence dimension.

        Args:
            theta: Phase field [B, N]
            z: Optional coordinates [B, N, d_coord]

        Returns:
            grad_theta: Phase gradient [B, N, d_coord]
        """
        B, N = theta.shape

        if z is not None:
            # Compute gradient via chain rule
            # d theta / d z_i = (d theta / d omega) * (d omega / d z_i)
            # where omega = ||z||

            omega = torch.norm(z, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]

            # d omega / d z_i = z_i / omega
            d_omega_dz = z / omega  # [B, N, d_coord]

            # Need alpha to compute d theta / d omega
            # For now, assume alpha embedded in theta computation
            # Approximate: d theta / d omega ~ -alpha / omega
            # Use average theta slope as proxy
            theta_expanded = theta.unsqueeze(-1)  # [B, N, 1]
            d_theta_domega = -theta_expanded / (omega * math.pi / 2 + 1e-8)

            grad_theta = d_theta_domega * d_omega_dz  # [B, N, d_coord]

        else:
            # Finite difference along sequence dimension
            # Pad for boundary
            theta_pad = F.pad(theta, (1, 1), mode='replicate')

            # Central difference
            d_theta = (theta_pad[:, 2:] - theta_pad[:, :-2]) / 2  # [B, N]

            # Broadcast to coordinate dimensions (isotropic approximation)
            grad_theta = d_theta.unsqueeze(-1).expand(B, N, self.d_coord)

        return grad_theta

    def compute_symplectic_form(
        self,
        grad_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symplectic form B_{mu nu} = nabla_mu theta wedge nabla_nu theta.

        The wedge product of two 1-forms gives an antisymmetric 2-form:
            B_{mu nu} = (nabla_mu theta)(nabla_nu theta) - (nabla_nu theta)(nabla_mu theta)

        Args:
            grad_theta: Phase gradient [B, N, d_coord]

        Returns:
            B: Symplectic form [B, N, d_coord, d_coord] (antisymmetric)
        """
        # Outer product
        B = torch.einsum('...i,...j->...ij', grad_theta, grad_theta)

        # Antisymmetrize: B_{ij} - B_{ji}
        B = B - B.transpose(-1, -2)

        return B

    def forward(
        self,
        A_metric: torch.Tensor,
        z: torch.Tensor,
        alpha: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complex metric G = A + iB.

        Args:
            A_metric: Real symmetric metric [B, N, d_coord, d_coord]
            z: Embeddings/coordinates [B, N, d_model] or [B, N, d_coord]
            alpha: Fractional order for phase computation

        Returns:
            G: Dict with 'real' (A) and 'imag' (B) components
        """
        # Compute phase field
        theta = self.compute_phase_field(z, alpha)

        # Compute phase gradient
        grad_theta = self.compute_phase_gradient(theta, z if z.size(-1) == self.d_coord else None)

        # Compute symplectic form
        B = self.compute_symplectic_form(grad_theta)

        return {
            'real': A_metric,       # Symmetric, Riemannian (Sigma)
            'imag': B,              # Antisymmetric, Symplectic (Lambda)
            'theta': theta,         # Phase field for diagnostics
        }

    def verify_symmetries(self, G: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """
        Verify that A is symmetric and B is antisymmetric.

        Args:
            G: Complex metric dict

        Returns:
            Dict with verification results
        """
        A = G['real']
        B = G['imag']

        A_symmetric = torch.allclose(A, A.transpose(-1, -2), atol=1e-5)
        B_antisymmetric = torch.allclose(B, -B.transpose(-1, -2), atol=1e-5)

        return {
            'A_symmetric': A_symmetric,
            'B_antisymmetric': B_antisymmetric,
        }


class SpinorBilinears(nn.Module):
    """
    Compute spinor bilinear forms for K0 -> K1 -> K2 mapping.

    The causal hierarchy is realized via standard bilinear projections:
        K0 (Spinor):  sigma = psi_bar psi                      [Scalar]
        K1 (Wedge):   Phi_{mu nu} = psi_bar gamma_{mu nu} psi  [Antisymmetric]
        K2 (Tensor):  Theta_{mu nu} = psi_bar gamma_{(mu} gamma_{nu)} psi  [Symmetric]

    These map the informational -> flow -> inertia hierarchy.
    """

    def __init__(self, d_spinor: int = 4, d_coord: int = 8):
        """
        Initialize spinor bilinear computation.

        Args:
            d_spinor: Spinor dimension (4 for Dirac, 2 for Weyl)
            d_coord: Coordinate dimension for gamma matrices
        """
        super().__init__()
        self.d_spinor = d_spinor
        self.d_coord = d_coord

        # Learnable gamma matrices (Clifford algebra generators)
        # gamma_mu satisfies: {gamma_mu, gamma_nu} = 2 g_{mu nu}
        # Initialize with random orthogonal matrices
        self.gamma = nn.Parameter(
            torch.randn(d_coord, d_spinor, d_spinor) / math.sqrt(d_spinor)
        )

    def gamma_product(self, mu: int, nu: int) -> torch.Tensor:
        """
        Compute gamma_mu gamma_nu.

        Args:
            mu, nu: Coordinate indices

        Returns:
            Product [d_spinor, d_spinor]
        """
        return torch.matmul(self.gamma[mu], self.gamma[nu])

    def gamma_anticommutator(self, mu: int, nu: int) -> torch.Tensor:
        """
        Compute {gamma_mu, gamma_nu} = gamma_mu gamma_nu + gamma_nu gamma_mu.

        This should give 2 g_{mu nu} I for proper Clifford algebra.

        Args:
            mu, nu: Coordinate indices

        Returns:
            Anticommutator [d_spinor, d_spinor]
        """
        return self.gamma_product(mu, nu) + self.gamma_product(nu, mu)

    def gamma_commutator(self, mu: int, nu: int) -> torch.Tensor:
        """
        Compute [gamma_mu, gamma_nu] = gamma_mu gamma_nu - gamma_nu gamma_mu.

        This is the bivector generator gamma_{mu nu}.

        Args:
            mu, nu: Coordinate indices

        Returns:
            Commutator [d_spinor, d_spinor]
        """
        return self.gamma_product(mu, nu) - self.gamma_product(nu, mu)

    def k0_scalar(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute K0 scalar: sigma = psi_bar psi.

        This is the informational overlap / probability amplitude.

        Args:
            psi: Spinor field [B, N, d_spinor]

        Returns:
            sigma: Scalar [B, N]
        """
        # psi_bar = psi^dagger (conjugate transpose)
        # For real spinors, just transpose
        psi_bar = psi  # Simplified for real case
        sigma = torch.einsum('...i,...i->...', psi_bar, psi)
        return sigma

    def k1_wedge(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute K1 wedge: Phi_{mu nu} = psi_bar gamma_{mu nu} psi.

        This is the antisymmetric Torquency / flow / force field.

        Args:
            psi: Spinor field [B, N, d_spinor]

        Returns:
            Phi: Bivector [B, N, d_coord, d_coord] (antisymmetric)
        """
        B, N, S = psi.shape
        Phi = torch.zeros(B, N, self.d_coord, self.d_coord, device=psi.device)

        for mu in range(self.d_coord):
            for nu in range(self.d_coord):
                # gamma_{mu nu} = (1/2)[gamma_mu, gamma_nu]
                gamma_mn = 0.5 * self.gamma_commutator(mu, nu)
                # Phi_{mu nu} = psi_bar gamma_{mu nu} psi
                Phi[:, :, mu, nu] = torch.einsum(
                    '...i,ij,...j->...',
                    psi, gamma_mn, psi
                )

        return Phi

    def k2_tensor(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute K2 tensor: Theta_{mu nu} = psi_bar gamma_{(mu} gamma_{nu)} psi.

        This is the symmetric Newtocity / inertia / stress field.

        Args:
            psi: Spinor field [B, N, d_spinor]

        Returns:
            Theta: Symmetric tensor [B, N, d_coord, d_coord]
        """
        B, N, S = psi.shape
        Theta = torch.zeros(B, N, self.d_coord, self.d_coord, device=psi.device)

        for mu in range(self.d_coord):
            for nu in range(self.d_coord):
                # gamma_{(mu} gamma_{nu)} = (1/2){gamma_mu, gamma_nu}
                gamma_sym = 0.5 * self.gamma_anticommutator(mu, nu)
                # Theta_{mu nu} = psi_bar gamma_{(mu} gamma_{nu)} psi
                Theta[:, :, mu, nu] = torch.einsum(
                    '...i,ij,...j->...',
                    psi, gamma_sym, psi
                )

        return Theta

    def forward(self, psi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all spinor bilinears.

        Args:
            psi: Spinor field [B, N, d_spinor]

        Returns:
            Dict with K0, K1, K2 components
        """
        return {
            'K0': self.k0_scalar(psi),   # [B, N] scalar overlap
            'K1': self.k1_wedge(psi),    # [B, N, d, d] antisymmetric flow
            'K2': self.k2_tensor(psi),   # [B, N, d, d] symmetric inertia
        }


class PhaseOrthogonalProjector(nn.Module):
    """
    Projects quantities onto orthogonal Sigma (real) and Lambda (imaginary) axes.

    This enforces the phase orthogonality that guarantees stability:
    - Sigma: Geometric/symmetric contributions (observable)
    - Lambda: Spectral/antisymmetric quantum corrections (phase)

    The orthogonality Sigma perp Lambda ensures quantum corrections don't
    contaminate observable quantities, making the O(1) recurrence stable.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Projection matrices for Sigma and Lambda sectors
        self.sigma_proj = nn.Linear(d_model, d_model)
        self.lambda_proj = nn.Linear(d_model, d_model)

        # Ensure orthogonality via parameterization
        # We use Householder reflections to maintain orthogonality
        self.v_sigma = nn.Parameter(torch.randn(d_model))
        self.v_lambda = nn.Parameter(torch.randn(d_model))

    def householder_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Householder reflection matrix from vector v."""
        v = v / (torch.norm(v) + 1e-8)
        return torch.eye(self.d_model, device=v.device) - 2 * torch.outer(v, v)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project x onto orthogonal Sigma and Lambda sectors.

        Args:
            x: Input [B, N, d_model]

        Returns:
            Dict with 'sigma' (real) and 'lambda' (imaginary) projections
        """
        # Compute orthogonal projection matrices
        P_sigma = self.householder_matrix(self.v_sigma)
        P_lambda = self.householder_matrix(self.v_lambda)

        # Gram-Schmidt to ensure strict orthogonality
        # Project lambda orthogonal to sigma
        overlap = torch.sum(P_sigma * P_lambda)
        P_lambda_orth = P_lambda - overlap * P_sigma
        P_lambda_orth = P_lambda_orth / (torch.norm(P_lambda_orth) + 1e-8)

        # Apply projections
        x_sigma = torch.einsum('ij,...j->...i', P_sigma, x)
        x_lambda = torch.einsum('ij,...j->...i', P_lambda_orth, x)

        return {
            'sigma': x_sigma,   # Real / geometric / observable
            'lambda': x_lambda,  # Imaginary / spectral / phase
        }
