"""
Cognitive Manifold: Coordinate Spacetime from Geometry

Provides coordinate spacetime derived from LIoR metric and geodesic structure,
rather than hard-coded tree scaffolding.

Key components:
1. Metric g_{mu nu}(z) - learned metric on embedding space
2. LIoR metric: tilde_g = R(x)^2 * g - resilience-weighted effective metric
3. Complex metric: G = A + iB where A is Riemannian, B is symplectic
4. Geodesic integration for exp/log maps
5. Normal coordinates centered at origin (COG of dataset)
6. Spinor bilinears for K0->K1->K2 mapping

The tree/graph structure becomes DERIVED from this geometry, not fundamental.

Architecture Constants (from CDGT):
    r = 8: Branching factor (cluster concepts per level)
    H ~ 13: Hierarchy depth (log_8 N)
    D_sem = 512: Semantic embedding dimension

References:
- Exponential map: exp_p(v) = gamma(1) where gamma is geodesic
- Normal coordinates: Christoffel symbols vanish at origin
- LIoR functional: path integral of R(x) * sqrt(g_{mu nu} dx^mu dx^nu)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .complex_metric import ComplexMetricTensor, SpinorBilinears, PhaseOrthogonalProjector
from .lior_kernel import LiorKernel


class CognitiveManifold(nn.Module):
    """
    Cognitive manifold providing coordinate spacetime from geometry.

    Instead of hard-coding a tree structure, coordinates emerge from:
    1. A learned metric g_{mu nu}(z)
    2. Scalar resilience R(x) giving effective metric tilde_g = R^2 * g
    3. Complex metric G = A + iB with phase orthogonality
    4. Geodesic integration defining the exponential map
    5. Normal coordinates via log map

    Attributes:
        d_embed: Dimension of embedding space
        d_coord: Dimension of coordinate manifold (default 8)
        d_spinor: Dimension of spinor space (default 4)
        origin: Origin point for normal coordinates (COG of dataset)
    """

    def __init__(
        self,
        d_embed: int,
        d_coord: int = 8,
        d_spinor: int = 4,
        learnable_metric: bool = True
    ):
        super().__init__()

        self.d_embed = d_embed
        self.d_coord = d_coord
        self.d_spinor = d_spinor

        # === Coordinate projection ===
        self.coord_proj = nn.Linear(d_embed, d_coord)

        # === Spinor projection ===
        self.spinor_proj = nn.Linear(d_embed, d_spinor)

        # === Learnable metric basis ===
        # g_{mu nu} = L @ L.T to ensure positive definiteness
        if learnable_metric:
            self.L = nn.Parameter(
                torch.eye(d_coord) + 0.1 * torch.randn(d_coord, d_coord)
            )
        else:
            self.register_buffer('L', torch.eye(d_coord))

        # Position-dependent metric perturbation
        self.metric_net = nn.Sequential(
            nn.Linear(d_coord, d_coord * 2),
            nn.GELU(),
            nn.Linear(d_coord * 2, d_coord * d_coord),
        )

        # === Resilience field R(x) ===
        self.resilience_net = nn.Sequential(
            nn.Linear(d_coord, d_coord),
            nn.GELU(),
            nn.Linear(d_coord, 1),
            nn.Softplus(),  # R(x) > 0
        )

        # === Complex metric ===
        self.complex_metric = ComplexMetricTensor(d_coord)

        # === Spinor bilinears ===
        self.spinor_bilinears = SpinorBilinears(d_spinor, d_coord)

        # === Phase orthogonal projector ===
        self.phase_projector = PhaseOrthogonalProjector(d_embed)

        # === LIoR kernel for fractional order ===
        self.lior_kernel = LiorKernel()

        # === Origin for normal coordinates ===
        self.register_buffer('origin', torch.zeros(d_coord))
        self.origin_ema_decay = 0.99

    def to_coords(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to coordinate space.

        Args:
            z: Embeddings [B, N, d_embed] or [B, d_embed]

        Returns:
            Coordinates [B, N, d_coord] or [B, d_coord]
        """
        return self.coord_proj(z)

    def to_spinor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to spinor space.

        Args:
            z: Embeddings [B, N, d_embed] or [B, d_embed]

        Returns:
            Spinor [B, N, d_spinor] or [B, d_spinor]
        """
        return self.spinor_proj(z)

    def base_metric(self) -> torch.Tensor:
        """
        Get base metric g_0 = L @ L.T (position-independent part).

        Returns:
            [d_coord, d_coord] positive definite matrix
        """
        return self.L @ self.L.T

    def metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute metric g_{mu nu}(x) at coordinate x.

        The metric is position-dependent:
            g(x) = g_0 + h(x)
        where g_0 = L @ L.T and h(x) is a learned perturbation.

        Args:
            x: Coordinates [B, N, d_coord] or [B, d_coord]

        Returns:
            Metric tensor [B, N, d_coord, d_coord] or [B, d_coord, d_coord]
        """
        g0 = self.base_metric()

        # Position-dependent perturbation
        h_flat = self.metric_net(x)
        h = h_flat.view(*x.shape[:-1], self.d_coord, self.d_coord)

        # Symmetrize and scale
        h = 0.5 * (h + h.transpose(-1, -2))
        h = 0.1 * torch.tanh(h)

        # Full metric with regularization
        g = g0 + h + 1e-4 * torch.eye(self.d_coord, device=x.device)

        return g

    def scalar_resilience(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar resilience R(x) at coordinate x.

        R(x) = sqrt((1/n^2) g_{mu rho} g_{nu sigma} R^{mu nu rho sigma})

        For now, learned directly as positive scalar field.

        Args:
            x: Coordinates [B, N, d_coord] or [B, d_coord]

        Returns:
            Scalar resilience [B, N] or [B]
        """
        R = self.resilience_net(x).squeeze(-1)
        R = torch.clamp(R, min=0.1, max=10.0)
        return R

    def lior_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LIoR-weighted effective metric (A component).

        tilde_g_{mu nu}(x) = R(x)^2 * g_{mu nu}(x)

        Args:
            x: Coordinates [B, N, d_coord]

        Returns:
            Effective metric [B, N, d_coord, d_coord]
        """
        g = self.metric(x)
        R = self.scalar_resilience(x)
        R_sq = (R ** 2).unsqueeze(-1).unsqueeze(-1)
        return R_sq * g

    def full_complex_metric(
        self,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full complex metric G = A + iB.

        Args:
            z: Embeddings [B, N, d_embed]

        Returns:
            Dict with 'real' (A), 'imag' (B), 'theta' (phase field)
        """
        # Project to coordinates
        x = self.to_coords(z)

        # A = R^2 * g (LIoR metric)
        A = self.lior_metric(x)

        # Get fractional order from kernel
        alpha = self.lior_kernel.fractional_order

        # Compute complex metric (A + iB)
        G = self.complex_metric(A, x, alpha)

        return G

    def spinor_decomposition(
        self,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute K0 -> K1 -> K2 spinor bilinear decomposition.

        Args:
            z: Embeddings [B, N, d_embed]

        Returns:
            Dict with K0 (scalar), K1 (wedge), K2 (tensor)
        """
        psi = self.to_spinor(z)
        return self.spinor_bilinears(psi)

    def christoffel(self, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Compute Christoffel symbols via finite differences.

        Gamma^lambda_{mu nu} = (1/2) g^{lambda rho} (
            partial_mu g_{nu rho} + partial_nu g_{mu rho} - partial_rho g_{mu nu}
        )

        Args:
            x: Coordinates [B, N, d_coord]
            eps: Step size for finite differences

        Returns:
            Christoffel symbols [B, N, d_coord, d_coord, d_coord]
        """
        d = self.d_coord
        shape = x.shape[:-1]
        device = x.device

        # Get metric and inverse
        g = self.lior_metric(x)
        g_inv = torch.linalg.inv(g)

        # Compute metric derivatives
        dg = torch.zeros(*shape, d, d, d, device=device)

        for rho in range(d):
            x_plus = x.clone()
            x_plus[..., rho] += eps
            g_plus = self.lior_metric(x_plus)

            x_minus = x.clone()
            x_minus[..., rho] -= eps
            g_minus = self.lior_metric(x_minus)

            dg[..., rho, :, :] = (g_plus - g_minus) / (2 * eps)

        # Christoffel symbols
        Gamma = torch.zeros(*shape, d, d, d, device=device)

        for lam in range(d):
            for mu in range(d):
                for nu in range(d):
                    for rho in range(d):
                        Gamma[..., lam, mu, nu] += 0.5 * g_inv[..., lam, rho] * (
                            dg[..., mu, nu, rho] +
                            dg[..., nu, mu, rho] -
                            dg[..., rho, mu, nu]
                        )

        return Gamma

    def geodesic_step(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of geodesic integration (RK2).

        Geodesic equation: d^2 x^lambda / dt^2 + Gamma^lambda_{mu nu} v^mu v^nu = 0

        Args:
            x: Position [B, N, d_coord]
            v: Velocity [B, N, d_coord]
            dt: Time step

        Returns:
            (x_new, v_new)
        """
        Gamma = self.christoffel(x)

        # Acceleration
        a = -torch.einsum('...lmn,...m,...n->...l', Gamma, v, v)

        # RK2 midpoint
        x_mid = x + 0.5 * dt * v
        v_mid = v + 0.5 * dt * a

        Gamma_mid = self.christoffel(x_mid)
        a_mid = -torch.einsum('...lmn,...m,...n->...l', Gamma_mid, v_mid, v_mid)

        x_new = x + dt * v_mid
        v_new = v + dt * a_mid

        return x_new, v_new

    def exp_map(
        self,
        origin: torch.Tensor,
        tangent: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Exponential map: tangent vector at origin -> point on manifold.

        exp_p(v) = gamma(1) where gamma is geodesic with gamma(0)=p, gamma'(0)=v

        Args:
            origin: Origin point [d_coord] or [B, d_coord]
            tangent: Tangent vector [B, N, d_coord]
            steps: Number of integration steps

        Returns:
            Point on manifold [B, N, d_coord]
        """
        if origin.dim() == 1:
            origin = origin.unsqueeze(0).unsqueeze(0).expand_as(tangent)
        elif origin.dim() == 2:
            origin = origin.unsqueeze(0).expand_as(tangent)

        x = origin.clone()
        v = tangent.clone()
        dt = 1.0 / steps

        for _ in range(steps):
            x, v = self.geodesic_step(x, v, dt)

        return x

    def log_map_approx(
        self,
        origin: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate log map (linear approximation for small distances).

        Args:
            origin: Origin [d_coord] or [B, d_coord]
            target: Target [B, N, d_coord]

        Returns:
            Tangent vector [B, N, d_coord]
        """
        if origin.dim() == 1:
            origin = origin.unsqueeze(0).unsqueeze(0)
        elif origin.dim() == 2:
            origin = origin.unsqueeze(1)

        return target - origin

    def normal_coords(
        self,
        z: torch.Tensor,
        origin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Express embeddings in normal coordinates centered at origin.

        Args:
            z: Embeddings [B, N, d_embed]
            origin: Center [d_coord], uses self.origin if None

        Returns:
            Normal coordinates [B, N, d_coord]
        """
        x = self.to_coords(z)

        if origin is None:
            origin = self.origin

        return self.log_map_approx(origin, x)

    def update_origin(self, z: torch.Tensor):
        """
        Update origin as EMA of batch COG.

        Args:
            z: Embeddings [B, N, d_embed]
        """
        with torch.no_grad():
            x = self.to_coords(z)
            batch_cog = x.mean(dim=(0, 1))

            self.origin = (
                self.origin_ema_decay * self.origin +
                (1 - self.origin_ema_decay) * batch_cog
            )

    def lior_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Compute LIoR distance between two points.

        LIoR[gamma] = integral R(x) sqrt(g_{mu nu} dx^mu dx^nu) dtau

        Args:
            x1: Start [B, d_coord]
            x2: End [B, d_coord]
            num_samples: Samples along path

        Returns:
            Distance [B]
        """
        t = torch.linspace(0, 1, num_samples, device=x1.device).view(1, -1, 1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        path = x1 + t * (x2 - x1)
        dx = (x2 - x1) / num_samples

        R = self.scalar_resilience(path)
        g = self.lior_metric(path)

        ds_sq = torch.einsum('bni,bnij,bnj->bn', dx.expand_as(path), g, dx.expand_as(path))
        ds = torch.sqrt(ds_sq.clamp(min=1e-8))

        integrand = R * ds
        distance = torch.trapezoid(integrand, dx=1.0/num_samples, dim=1)

        return distance

    def forward(
        self,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full manifold computation.

        Args:
            z: Embeddings [B, N, d_embed]

        Returns:
            Dict with all geometric quantities
        """
        # Coordinates
        x = self.to_coords(z)

        # Complex metric
        G = self.full_complex_metric(z)

        # Spinor decomposition
        K = self.spinor_decomposition(z)

        # Normal coordinates
        normal = self.normal_coords(z)

        # Resilience
        R = self.scalar_resilience(x)

        # Phase projection
        phase_split = self.phase_projector(z)

        return {
            'coords': x,
            'normal_coords': normal,
            'metric_real': G['real'],      # A_{mu nu}
            'metric_imag': G['imag'],      # B_{mu nu}
            'phase': G['theta'],
            'resilience': R,
            'K0': K['K0'],                 # Scalar overlap
            'K1': K['K1'],                 # Wedge (Torquency)
            'K2': K['K2'],                 # Tensor (Newtocity)
            'sigma': phase_split['sigma'],   # Real/geometric
            'lambda': phase_split['lambda'], # Imaginary/spectral
        }
